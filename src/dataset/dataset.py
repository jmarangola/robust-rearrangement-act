from pathlib import Path
import numpy as np
import torch
from typing import Dict, Union, List

from src.dataset.normalizer import LinearNormalizer
from src.dataset.zarr import combine_zarr_datasets
from src.common.control import ControlMode

from src.common.tasks import furniture2idx
import src.common.geometry as G
import furniture_bench.controllers.control_utils as C

from ipdb import set_trace as bp


def create_sample_indices(
    episode_ends: np.ndarray,
    sequence_length: int,
    pad_before: int = 0,
    pad_after: int = 0,
):
    indices = list()
    for i in range(len(episode_ends)):
        start_idx = 0
        if i > 0:
            start_idx = episode_ends[i - 1]
        end_idx = episode_ends[i]
        episode_length = end_idx - start_idx

        min_start = -pad_before
        max_start = episode_length - sequence_length + pad_after

        # range stops one idx before end
        for idx in range(min_start, max_start + 1):
            buffer_start_idx = max(idx, 0) + start_idx
            buffer_end_idx = min(idx + sequence_length, episode_length) + start_idx
            start_offset = buffer_start_idx - (idx + start_idx)
            end_offset = (idx + sequence_length + start_idx) - buffer_end_idx
            sample_start_idx = 0 + start_offset
            sample_end_idx = sequence_length - end_offset
            indices.append(
                [buffer_start_idx, buffer_end_idx, sample_start_idx, sample_end_idx, i]
            )
    indices = np.array(indices)
    return indices


def sample_sequence(
    train_data: Dict[str, torch.Tensor],
    sequence_length: int,
    buffer_start_idx: int,
    buffer_end_idx: int,
    sample_start_idx: int,
    sample_end_idx: int,
) -> Dict[str, torch.Tensor]:
    result = dict()
    # TODO: Implement the performance improvement (particularly for image-based training):
    # https://github.com/real-stanford/diffusion_policy/blob/main/diffusion_policy/common/sampler.py#L130-L138
    for key, input_arr in train_data.items():
        sample = input_arr[buffer_start_idx:buffer_end_idx]
        data = sample
        if (sample_start_idx > 0) or (sample_end_idx < sequence_length):
            data = torch.zeros(
                size=(sequence_length,) + input_arr.shape[1:], dtype=input_arr.dtype
            )
            if sample_start_idx > 0:
                data[:sample_start_idx] = sample[0]
            if sample_end_idx < sequence_length:
                data[sample_end_idx:] = sample[-1]
            data[sample_start_idx:sample_end_idx] = sample
        result[key] = data
    return result


class FurnitureImageDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        dataset_paths: Union[List[str], str],
        pred_horizon: int,
        obs_horizon: int,
        action_horizon: int,
        data_subset: int = None,
        predict_past_actions: bool = False,
        control_mode: ControlMode = ControlMode.delta,
        pad_after: bool = True,
        max_episode_count: Union[dict, None] = None,
        minority_class_power: bool = False,
    ):
        self.pred_horizon = pred_horizon
        self.action_horizon = action_horizon
        self.obs_horizon = obs_horizon
        self.control_mode = control_mode
        self.minority_class_power = minority_class_power

        self.normalizer = LinearNormalizer()

        # Read from zarr dataset
        combined_data, metadata = combine_zarr_datasets(
            dataset_paths,
            [
                "color_image1",
                "color_image2",
                "robot_state",
                f"action/{control_mode}",
                "skill",
            ],
            max_episodes=data_subset,
            max_ep_cnt=max_episode_count,
        )

        # (N, D)
        # Get only the first data_subset episodes
        self.episode_ends = combined_data["episode_ends"]
        self.metadata = metadata
        print(f"Loading dataset of {len(self.episode_ends)} episodes:")
        for path, data in metadata.items():
            print(
                f"  {path}: {data['n_episodes_used']} episodes, {data['n_frames_used']}"
            )

        self.train_data = {
            "robot_state": torch.from_numpy(combined_data["robot_state"]),
            "action": torch.from_numpy(combined_data[f"action/{control_mode}"]),
        }

        # Fit the normalizer to the data
        self.normalizer.fit(self.train_data)

        # Normalize data to [-1,1]
        for key in self.normalizer.keys():
            self.train_data[key] = self.normalizer(
                self.train_data[key], key, forward=True
            )

        # Add the color images to the train_data (it's not supposed to be normalized)
        # and move the channels to the front
        self.train_data["color_image1"] = torch.from_numpy(
            combined_data["color_image1"]
        ).permute(0, 3, 1, 2)
        self.train_data["color_image2"] = torch.from_numpy(
            combined_data["color_image2"]
        ).permute(0, 3, 1, 2)

        # compute start and end of each state-action sequence
        # also handles padding
        self.sequence_length = (
            pred_horizon if predict_past_actions else obs_horizon + pred_horizon - 1
        )
        self.indices = create_sample_indices(
            episode_ends=self.episode_ends,
            sequence_length=self.sequence_length,
            pad_before=obs_horizon - 1,
            pad_after=action_horizon - 1 if pad_after else 0,
        )

        self.n_samples = len(self.indices)

        self.task_idxs = np.array(
            [furniture2idx[f] for f in combined_data["furniture"]]
        )
        self.successes = combined_data["success"].astype(np.uint8)
        self.skills = combined_data["skill"].astype(np.uint8)
        self.failure_idx = combined_data["failure_idx"]
        self.domain = combined_data["domain"]

        # Add action and observation dimensions to the dataset
        self.action_dim = self.train_data["action"].shape[-1]
        self.robot_state_dim = self.train_data["robot_state"].shape[-1]

        # Set the limits for the action indices based on wether we predict past actions or not
        # First action refers to the first action we predict, not necessarily the first action executed
        self.first_action_idx = 0 if predict_past_actions else self.obs_horizon - 1
        self.final_action_idx = self.first_action_idx + self.pred_horizon

        if self.minority_class_power:
            # Upsample the minority class

            sim_indices = []
            real_indices = []

            for i, (_, _, _, _, demo_idx) in enumerate(self.indices):
                if self.domain[demo_idx] == 0:
                    sim_indices.append(i)
                else:
                    real_indices.append(i)

            sim_indices = np.array(sim_indices)
            real_indices = np.array(real_indices)

            # Calculate the number of samples for each class in self.indices
            sim_samples = len(sim_indices)
            real_samples = len(real_indices)
            class_samples = np.array([sim_samples, real_samples])
            total_samples = len(self.indices)

            print(
                f"Ratio of real to sim samples before upsampling: {real_samples/sim_samples:.2f}"
            )

            # Calculate the desired number of samples for each class based on cube root of class sizes
            class_weights = np.power(class_samples, 1 / minority_class_power)
            class_weights = class_weights / np.sum(class_weights)
            desired_class_samples = total_samples * class_weights

            print(
                f"Ratio of real to sim samples after upsampling: {desired_class_samples[1]/desired_class_samples[0]:.2f}"
            )

            # Identify the minority class
            minority_class = np.argmin(class_samples)

            # Calculate the number of additional samples needed for the minority class
            additional_samples_needed = int(
                desired_class_samples[minority_class] - class_samples[minority_class]
            )

            if additional_samples_needed > 0:

                # Randomly select minority samples to duplicate
                additional_indices = np.random.choice(
                    real_indices,
                    size=additional_samples_needed,
                    replace=True,
                )

                # Create additional samples in self.indices for minority class samples
                additional_samples = self.indices[additional_indices]
                self.indices = np.concatenate((self.indices, additional_samples))

    def set_normalizer(self, normalizer: LinearNormalizer):
        self.normalizer.load_state_dict(normalizer.state_dict())

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        # get the start/end indices for this datapoint
        (
            buffer_start_idx,
            buffer_end_idx,
            sample_start_idx,
            sample_end_idx,
            demo_idx,
        ) = self.indices[idx]

        # get normalized data using these indices
        nsample = sample_sequence(
            train_data=self.train_data,
            sequence_length=self.sequence_length,
            buffer_start_idx=buffer_start_idx,
            buffer_end_idx=buffer_end_idx,
            sample_start_idx=sample_start_idx,
            sample_end_idx=sample_end_idx,
        )

        # Discard unused observations
        # TODO: This is where a performance improvement can be made, i.e., don't load
        # the full image sequence if we're only going to use a subset of it
        nsample["color_image1"] = nsample["color_image1"][: self.obs_horizon, :]
        nsample["color_image2"] = nsample["color_image2"][: self.obs_horizon, :]
        nsample["robot_state"] = nsample["robot_state"][: self.obs_horizon, :]

        # Discard unused actions
        nsample["action"] = nsample["action"][
            self.first_action_idx : self.final_action_idx, :
        ]

        # Add the task index and success flag to the sample
        nsample["task_idx"] = torch.LongTensor([self.task_idxs[demo_idx]])
        nsample["success"] = torch.IntTensor([self.successes[demo_idx]])
        nsample["domain"] = torch.IntTensor([self.domain[demo_idx]])

        return nsample

    def train(self):
        pass

    def eval(self):
        pass


class FurnitureStateDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        dataset_paths: Union[List[Path], Path],
        pred_horizon: int,
        obs_horizon: int,
        action_horizon: int,
        data_subset: int = None,
        predict_past_actions: bool = False,
        control_mode: ControlMode = ControlMode.delta,
        pad_after: bool = True,
        max_episode_count: Union[dict, None] = None,
        task: str = None,
        add_relative_pose: bool = False,
        normalizer: LinearNormalizer = None,
        include_future_obs: bool = False,
        parts_poses_in_robot_frame: bool = False,
    ):
        assert control_mode == ControlMode.relative or not parts_poses_in_robot_frame

        self.pred_horizon = pred_horizon
        self.action_horizon = action_horizon
        self.obs_horizon = obs_horizon
        self.predict_past_actions = predict_past_actions
        self.control_mode = control_mode
        self.include_future_obs = include_future_obs
        self.parts_poses_in_robot_frame = parts_poses_in_robot_frame

        # Read from zarr dataset
        # The dataset only has `delta/pos` control modes, use pos if `relative` is selected
        control_mode = "pos" if control_mode == ControlMode.relative else control_mode
        combined_data, metadata = combine_zarr_datasets(
            dataset_paths,
            [
                "parts_poses",
                "robot_state",
                f"action/{control_mode}",
                # "skill",
                # "reward",
            ],
            max_episodes=data_subset,
            max_ep_cnt=max_episode_count,
        )

        # (N, D)
        # Get only the first data_subset episodes
        self.episode_ends: np.ndarray = combined_data["episode_ends"]
        self.metadata = metadata
        print(f"Loading dataset of {len(self.episode_ends)} episodes:")
        for path, data in metadata.items():
            print(
                f"  {path}: {data['n_episodes_used']} episodes, {data['n_frames_used']}"
            )

        # Get the data and convert to torch tensors
        robot_state = torch.from_numpy(combined_data["robot_state"])
        action = torch.from_numpy(combined_data[f"action/{control_mode}"])
        parts_poses = torch.from_numpy(combined_data["parts_poses"])

        if self.parts_poses_in_robot_frame:
            from furniture_bench import config

            # Parts are by default in april frame, convert to robot frame
            base_tag_from_robot_mat = torch.tensor(
                config["robot"]["tag_base_from_robot_base"]
            )

            part_poses_mat = C.pose2mat_batched(
                parts_poses[:, :, :3], parts_poses[:, :, 3:7]
            )

            robot_coord_poses_mat = base_tag_from_robot_mat @ part_poses_mat
            parts_poses = torch.cat(C.mat2pose_batched(robot_coord_poses_mat), dim=-1)

        self.train_data = {
            "parts_poses": parts_poses,
            "robot_state": robot_state,
            "action": action,
        }

        # Fit the normalizer to the data
        self.normalizer = LinearNormalizer()
        if normalizer is None:
            self.normalizer.fit(self.train_data)
        else:
            self.normalizer.load_state_dict(normalizer.state_dict())
            self.normalizer.cpu()

        if task == "place-tabletop":
            self._make_tabletop_goal()

        # If control mode is relative we need a different normalization scheme
        # I.e., we only normalize the position part of the poses
        if self.control_mode == ControlMode.relative:
            from torch import nn

            # NOTE: For now, only implement relative poses for the action
            # === Normalize the parts poses and robot state ===
            # # We need the ee position and parts position limits to be the
            # # same for the relative pose calculation
            # # Startby finding the limits for the robot state
            # ee_pos_min = self.train_data["robot_state"][:, :3].min(dim=0).values
            # ee_pos_max = self.train_data["robot_state"][:, :3].max(dim=0).values

            # # Find the limits for the parts poses, but first reshape so that
            # # the parts poses are (N, P, 7)
            # parts_pos = (
            #     self.train_data["parts_poses"]
            #     .view(self.train_data["parts_poses"].shape[0], -1, 7)[:, :, :3]
            #     .view(-1, 3)
            # )
            # parts_pos_min = parts_pos.min(dim=0).values
            # parts_pos_max = parts_pos.max(dim=0).values

            # # Find the overall min and max for the positions
            # pos_min = torch.min(ee_pos_min, parts_pos_min)
            # pos_max = torch.max(ee_pos_max, parts_pos_max)

            # # Update the robot_state in the normalizer with the new limits
            # self.normalizer.stats.robot_state.min[:3] = pos_min
            # self.normalizer.stats.robot_state.max[:3] = pos_max

            # # Set the orientation part of the robot state to [-1, 1] (rot_6d, no normalization)
            # self.normalizer.stats.robot_state.min[3:9] = -1.0
            # self.normalizer.stats.robot_state.max[3:9] = 1.0

            # # Make a view into the normalizer stats for the parts poses
            # parts_pose_min = self.normalizer.stats.parts_poses.min.view(-1, 7)
            # parts_pose_max = self.normalizer.stats.parts_poses.max.view(-1, 7)

            # # Set the position part of the parts poses to the overall min and max
            # parts_pose_min[:, :3] = pos_min
            # parts_pose_max[:, :3] = pos_max

            # # Set the orientation part of the parts poses to [-1, 1] (quat, no normalization)
            # parts_pose_min[:, 3:7] = -1.0
            # parts_pose_max[:, 3:7] = 1.0

            # === Normalize the actions ===
            # Normalize the position part of the actions (for now, no normalization at all)
            self.normalizer.stats.action.min[:] = -1.0
            self.normalizer.stats.action.max[:] = 1.0

        # Normalize data to [-1,1]
        for key in self.normalizer.keys():
            self.train_data[key] = self.normalizer(
                self.train_data[key], key, forward=True
            )

        # Concatenate the robot_state and parts_poses into a single observation
        self.train_data["obs"] = torch.cat(
            [self.train_data["robot_state"], self.train_data["parts_poses"]], dim=-1
        )

        # Add parts poses relative to the end-effector as a new key in the train_data
        if add_relative_pose:
            parts_poses, robot_state = (
                self.train_data["parts_poses"],
                self.train_data["robot_state"],
            )
            # Parts is (N, P * 7)
            N = parts_poses.shape[0]
            n_parts = parts_poses.shape[1] // 7

            # Get the robot state end-effector pose
            ee_pos = robot_state[:, None, :3]
            ee_quat_xyzw = G.rot_6d_to_isaac_quat(robot_state[:, 3:9]).view(N, 1, 4)
            ee_pose = torch.cat([ee_pos, ee_quat_xyzw], dim=-1)

            # Reshape the parts poses into (N, P, 7)
            parts_pose = parts_poses.view(N, n_parts, 7)

            # Concatenate the relative pose into a single tensor
            rel_pose = G.pose_error(ee_pose, parts_pose)

            # Flatten the relative pose tensor and add it to the train_data
            self.train_data["rel_poses"] = rel_pose.view(N, -1)

            self.train_data["obs"] = torch.cat(
                [self.train_data["obs"], self.train_data["rel_poses"]], dim=-1
            )

        # Recalculate the rewards and returns
        rewards = torch.zeros_like(self.train_data["robot_state"][:, 0])
        rewards[self.episode_ends - 1] = 1.0

        gamma = 0.99
        returns = []
        ee = [0] + self.episode_ends.tolist()
        for start, end in zip(ee[:-1], ee[1:]):
            ep_rewards = rewards[start:end]
            timesteps = torch.arange(len(ep_rewards), device=ep_rewards.device)
            discounts = gamma**timesteps
            ep_returns = (
                torch.flip(
                    torch.cumsum(torch.flip(ep_rewards * discounts, dims=[0]), dim=0),
                    dims=[0],
                )
                / discounts
            )
            returns.append(ep_returns)

        # Concatenate the returns for all episodes into a single tensor
        returns = torch.cat(returns)
        self.train_data["returns"] = returns

        # compute start and end of each state-action sequence
        # also handles padding
        # If we only predict the future, we need to make sure we have enough actions to predict
        self.sequence_length = (
            pred_horizon if predict_past_actions else obs_horizon + pred_horizon - 1
        )
        self.indices = create_sample_indices(
            episode_ends=self.episode_ends,
            sequence_length=self.sequence_length,
            pad_before=obs_horizon - 1,
            pad_after=action_horizon - 1 if pad_after else 0,
        )

        self.n_samples = len(self.indices)

        self.task_idxs = np.array(
            [furniture2idx[f] for f in combined_data["furniture"]]
        )
        self.successes = combined_data["success"].astype(np.uint8)
        # self.skills = combined_data["skill"].astype(np.uint8)
        self.failure_idx = combined_data["failure_idx"]

        # Add action, robot_state, and parts_poses dimensions to the dataset
        self.action_dim = self.train_data["action"].shape[-1]
        self.robot_state_dim = self.train_data["robot_state"].shape[-1]
        self.parts_poses_dim = self.train_data["parts_poses"].shape[-1]
        self.obs_dim = (self.robot_state_dim + self.parts_poses_dim) * self.obs_horizon

        # Set the limits for the action indices based on wether we predict past actions or not
        # First action refers to the first action we predict, not necessarily the first action executed
        self.first_action_idx = 0 if predict_past_actions else self.obs_horizon - 1
        self.final_action_idx = self.first_action_idx + self.pred_horizon
        self.last_obs = (
            self.obs_horizon if not self.include_future_obs else self.sequence_length
        )

        del self.train_data["robot_state"]
        del self.train_data["parts_poses"]
        if add_relative_pose:
            del self.train_data["rel_poses"]

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        # get the start/end indices for this datapoint
        (
            buffer_start_idx,
            buffer_end_idx,
            sample_start_idx,
            sample_end_idx,
            demo_idx,
        ) = self.indices[idx]

        # get normalized data using these indices
        nsample = sample_sequence(
            train_data=self.train_data,
            sequence_length=self.sequence_length,
            buffer_start_idx=buffer_start_idx,
            buffer_end_idx=buffer_end_idx,
            sample_start_idx=sample_start_idx,
            sample_end_idx=sample_end_idx,
        )

        # From the diffusion_policy code:
        # E.g., obs=2, pred=16, act=8:
        # |o|o|                             observations:       2
        # | |a|a|a|a|a|a|a|a|               actions executed:   8
        # |p|p|p|p|p|p|p|p|p|p|p|p|p|p|p|p| actions predicted: 16

        # This is the logic for the indices if we only predict future actions
        # E.g., obs=2, pred=4, act=4:
        # |o|o|                             observations:       2
        # | |a|a|a|a|                       actions executed:   4
        # | |p|p|p|p|                       actions predicted:  4

        # This is the logic for the indices if we predict past actions
        # E.g., obs=2, pred=4, act=3:
        # |o|o|                             observations:       2
        # | |a|a|a|                         actions executed:   3
        # |p|p|p|p|                         actions predicted:  4

        # Discard unused actions
        nsample["action"] = nsample["action"][
            self.first_action_idx : self.final_action_idx, :
        ]

        # Discard unused observations
        nsample["obs"] = nsample["obs"][: self.last_obs, :]

        if self.control_mode == ControlMode.relative:
            # Each action in the chunk will be relative to the current EE pose
            curr_ee_pos = nsample["obs"][-1, :3]
            curr_ee_6d = nsample["obs"][-1, 3:9]
            curr_ee_quat = C.rotation_6d_to_quaternion(curr_ee_6d)

            # Calculate the relative pos action (the actions are absolute poses to begin with)
            nsample["action"][:, :3] = nsample["action"][:, :3] - curr_ee_pos

            # Calculate the relative rot action
            action_quat = C.rotation_6d_to_quaternion(nsample["action"][:, 3:9])

            # Want a quaternion such that if it's applied to the current EE pose, it will result in the action (absolute pose)
            # This is the same as the relative rotation between the current EE pose and the action
            # curr_quat * rel_quat = action_quat -> rel_quat = curr_quat^-1 * action_quat
            action_quat = C.quaternion_multiply(
                C.quaternion_invert(curr_ee_quat), action_quat
            )

            nsample["action"][:, 3:9] = C.quaternion_to_rotation_6d(action_quat)

        # Sum up the returns accrued during the action chunk
        # Double check if this should be calculated only for executed actions
        nsample["returns"] = nsample["returns"][
            self.first_action_idx : self.final_action_idx
        ].sum()

        return nsample

    def train(self):
        pass

    def eval(self):
        pass

    def _make_tabletop_goal(self):
        ee = np.array([0] + self.episode_ends.tolist())
        tabletop_goal = torch.tensor([0.0819, 0.2866, -0.0157])
        new_episode_starts = []
        new_episode_ends = []
        curr_cumulate_timesteps = 0
        self.episode_ends = []
        for prev_ee, curr_ee in zip(ee[:-1], ee[1:]):
            # Find the first index at which the tabletop goal is reached (if at all)
            for i in range(prev_ee, curr_ee):
                if torch.allclose(
                    self.train_data["parts_poses"][i, :3], tabletop_goal, atol=1e-2
                ):
                    new_episode_starts.append(prev_ee)
                    end = i + 10
                    new_episode_ends.append(end)
                    curr_cumulate_timesteps += end - prev_ee
                    self.episode_ends.append(curr_cumulate_timesteps)
                    break

        # Slice the train_data using the new episode starts and ends
        for key in self.train_data:
            data_slices = [
                self.train_data[key][start:end]
                for start, end in zip(new_episode_starts, new_episode_ends)
            ]
            self.train_data[key] = torch.cat(data_slices)

        self.episode_ends = torch.tensor(self.episode_ends)
