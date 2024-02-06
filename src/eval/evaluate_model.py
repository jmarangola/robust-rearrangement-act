import argparse
import time
from typing import List
import furniture_bench
from src.behavior.base import Actor  # noqa
import torch
from omegaconf import OmegaConf, DictConfig
from src.eval.rollout import calculate_success_rate
from src.behavior import get_actor
from src.common.tasks import furniture2idx, task_timeout
from src.common.files import trajectory_save_dir
from src.gym import get_env
from src.dataset import get_normalizer

from ipdb import set_trace as bp  # noqa
from wandb import Api
from wandb.sdk.wandb_run import Run

api = Api()


def validate_args(args: argparse.Namespace):
    assert (
        sum(
            [
                args.run_id is not None,
                args.sweep_id is not None,
                args.project_id is not None,
            ]
        )
        == 1
    ), "Exactly one of run-id, sweep-id, project-id must be provided"
    assert args.run_state is None or all(
        [
            state in ["running", "finished", "failed", "crashed"]
            for state in args.run_state
        ]
    ), (
        "Invalid run-state: "
        f"{args.run_state}. Valid options are: None, running, finished, failed, crashed"
    )
    assert (
        not args.continuous_mode
        or args.sweep_id is not None
        or args.project_id is not None
    ), "Continuous mode is only supported when sweep_id is provided"

    assert not args.leaderboard, "Leaderboard mode is not supported as of now"


def get_runs(args: argparse.Namespace) -> List[Run]:
    if args.sweep_id:
        runs: List[Run] = list(api.sweep(f"robot-rearrangement/{args.sweep_id}").runs)
    elif args.run_id:
        runs: List[Run] = [api.run(f"{args.run_id}")]
    elif args.project_id:
        runs: List[Run] = list(api.runs(f"robot-rearrangement/{args.project_id}"))
    else:
        raise ValueError("Exactly one of run-id, sweep-id, project-id must be provided")

        # Filter out the runs based on the run state
    if args.run_state:
        runs = [run for run in runs if run.state in args.run_state]
    return runs


def vision_encoder_field_hotfix(run, config):
    if isinstance(config.vision_encoder, str):
        # Read in the vision encoder config from the `vision_encoder` config group and set it
        OmegaConf.set_readonly(config, False)
        config.vision_encoder = OmegaConf.load(
            f"src/config/vision_encoder/{config.vision_encoder}.yaml"
        )
        OmegaConf.set_readonly(config, True)

        # Write it back to the run
        run.config = OmegaConf.to_container(config, resolve=True)
        run.update()


def convert_state_dict(state_dict):
    if not any(k.startswith("encoder1.0") for k in state_dict.keys()) and not any(
        k.startswith("encoder1.model.nets.3") for k in state_dict.keys()
    ):
        print("Dict already in the correct format")
        return

    # Change all instances of "encoder1.0" to "encoder1" and "encoder2.0" to "encoder2"
    # and all instances of "encoder1.1" to encoder1_proj and "encoder2.1" to "encoder2_proj"
    for k in list(state_dict.keys()):
        if k.startswith("encoder1.0"):
            new_k = k.replace("encoder1.0", "encoder1")
            state_dict[new_k] = state_dict.pop(k)
        elif k.startswith("encoder2.0"):
            new_k = k.replace("encoder2.0", "encoder2")
            state_dict[new_k] = state_dict.pop(k)
        elif k.startswith("encoder1.1"):
            new_k = k.replace("encoder1.1", "encoder1_proj")
            state_dict[new_k] = state_dict.pop(k)
        elif k.startswith("encoder2.1"):
            new_k = k.replace("encoder2.1", "encoder2_proj")
            state_dict[new_k] = state_dict.pop(k)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", type=str, required=False)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--n-envs", type=int, default=1)
    parser.add_argument("--n-rollouts", type=int, default=1)
    parser.add_argument("--randomness", type=str, default="low")
    parser.add_argument(
        "--furniture",
        "-f",
        type=str,
        choices=["one_leg", "lamp", "round_table", "desk", "square_table", "cabinet"],
        required=True,
    )
    parser.add_argument("--n-parts-assemble", type=int, default=None)

    parser.add_argument("--save-rollouts", action="store_true")
    parser.add_argument("--save-failures", action="store_true")

    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--leaderboard", action="store_true")

    # Define what should be done if the success rate fields are already present
    parser.add_argument(
        "--if-exists",
        type=str,
        choices=["skip", "overwrite", "append", "error"],
        default="error",
    )
    parser.add_argument(
        "--run-state",
        type=str,
        default=None,
        choices=["running", "finished", "failed", "crashed"],
        nargs="*",
    )

    # For batch evaluating runs from a sweep or a project
    parser.add_argument("--sweep-id", type=str, default=None)
    parser.add_argument("--project-id", type=str, default=None)

    parser.add_argument("--continuous-mode", action="store_true")
    parser.add_argument(
        "--continuous-interval",
        type=int,
        default=60,
        help="Pause interval before next evaluation",
    )
    parser.add_argument("--ignore-currently-evaluating-flag", action="store_true")

    parser.add_argument("--visualize", action="store_true")

    # Parse the arguments
    args = parser.parse_args()

    # Validate the arguments
    validate_args(args)

    # Make the device
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

    # Set the timeout
    rollout_max_steps = task_timeout(args.furniture, n_parts=args.n_parts_assemble)

    # Get the environment
    # TODO: This needs to be changed to enable recreation the env for each run
    print(
        "Creating the environment (this needs to be changed to enable recreation the env for each run)"
    )
    env = get_env(
        gpu_id=args.gpu,
        furniture=args.furniture,
        num_envs=args.n_envs,
        randomness=args.randomness,
        max_env_steps=rollout_max_steps,
        resize_img=False,
        act_rot_repr="rot_6d",
        ctrl_mode="osc",
        action_type="delta",
        verbose=False,
        headless=args.visualize,
    )

    # Start the evaluation loop
    print(f"Starting evaluation loop in continuous mode: {args.continuous_mode}")
    try:
        while True:
            # Get the run(s) to test
            runs = get_runs(args)

            print(f"Found {len(runs)} runs to evaluate")
            for run in runs:
                # First, we must flush the api and request the run again in case the information is stale
                api.flush()
                run = api.run("/".join([run.project, run.id]))

                # Check if the run is currently being evaluated
                if (
                    run.config.get("currently_evaluating", False)
                    and not args.ignore_currently_evaluating_flag
                ):
                    print(f"Run: {run.name} is currently being evaluated, skipping")
                    continue

                # Check if the run has already been evaluated
                how_update = "overwrite"
                if run.summary.get("success_rate", None) is not None:
                    if args.if_exists == "skip":
                        print(f"Run: {run.name} has already been evaluated, skipping")
                        continue
                    elif args.if_exists == "error":
                        raise ValueError(f"Run: {run.name} has already been evaluated")
                    elif args.if_exists == "overwrite":
                        print(
                            f"Run: {run.name} has already been evaluated, overwriting"
                        )
                        how_update = "overwrite"
                    elif args.if_exists == "append":
                        print(f"Run: {run.name} has already been evaluated, appending")
                        how_update = "append"

                # If in overwrite set the currently_evaluating flag to true runs can cooperate better in skip mode
                if how_update == "overwrite" and args.wandb:
                    print(
                        f"Setting currently_evaluating flag to true for run: {run.name}"
                    )
                    run.config["currently_evaluating"] = True
                    run.update()

                model_file = [f for f in run.files() if f.name.endswith(".pt")][0]
                model_path = model_file.download(exist_ok=True).name

                # Get the current `test_epoch_loss` from the run
                test_epoch_loss = run.summary.get("test_epoch_loss", None)
                print(
                    f"Evaluating run: {run.name} at test_epoch_loss: {test_epoch_loss}"
                )

                # Create the config object with the project name and make it read-only
                config: DictConfig = OmegaConf.create(
                    {**run.config, "project_name": run.project},
                    flags={"readonly": True},
                )

                # Get the normalizer
                normalizer_type = config.get("data", {}).get("normalization", "min_max")
                normalizer = get_normalizer(
                    normalizer_type=normalizer_type, control_mode="delta"
                )

                # TODO: Fix this properly, but for now have an ugly escape hatch
                vision_encoder_field_hotfix(run, config)

                print(OmegaConf.to_yaml(config))

                # Make the actor
                actor: Actor = get_actor(
                    config=config, normalizer=normalizer, device=device
                )

                # Load the model weights
                state_dict = torch.load(model_path)
                convert_state_dict(state_dict)

                actor.load_state_dict(state_dict)
                actor.eval()

                save_dir = trajectory_save_dir(
                    environment="sim",
                    task=args.furniture,
                    demo_source="rollout",
                    randomness=args.randomness,
                    create=False,
                )

                # Perform the rollouts
                actor.set_task(furniture2idx[args.furniture])
                rollout_stats = calculate_success_rate(
                    actor=actor,
                    env=env,
                    n_rollouts=args.n_rollouts,
                    rollout_max_steps=rollout_max_steps,
                    epoch_idx=0,
                    gamma=config.discount,
                    rollout_save_dir=save_dir,
                    save_failures=args.save_failures,
                    n_parts_assemble=args.n_parts_assemble,
                )
                success_rate = rollout_stats.success_rate

                print(
                    f"Success rate: {success_rate:.2%} ({rollout_stats.n_success}/{rollout_stats.n_rollouts})"
                )

                if args.wandb:
                    print("Writing to wandb...")

                    # Set the summary fields
                    if how_update == "overwrite":
                        run.summary["success_rate"] = success_rate
                        run.summary["n_success"] = rollout_stats.n_success
                        run.summary["n_rollouts"] = rollout_stats.n_rollouts
                    elif how_update == "append":
                        run.summary["n_success"] += rollout_stats.n_success
                        run.summary["n_rollouts"] += rollout_stats.n_rollouts
                        run.summary["success_rate"] = (
                            run.summary["n_success"] / run.summary["n_rollouts"]
                        )
                    else:
                        raise ValueError(f"Invalid how_update: {how_update}")

                    # Set the currently_evaluating flag to false
                    run.config["currently_evaluating"] = False

                    # Update the run to save the summary fields
                    run.update()

                else:
                    print("Not writing to wandb")

            # If not in continuous mode, break
            if not args.continuous_mode:
                break

            # Sleep for the interval
            print(
                f"Sleeping for {args.continuous_interval} seconds before checking for new runs..."
            )
            time.sleep(args.continuous_interval)
    finally:
        # Unset the "currently_evaluating" flag
        if args.wandb:
            print("Exiting the evaluation loop")
            print("Unsetting the currently_evaluating flag")
            run.config["currently_evaluating"] = False
            run.update()
