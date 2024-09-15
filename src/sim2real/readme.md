Steps for sim to real
- Convert `.obj` files into `.usd` files. You can use the `convert_parts_to_usd.py` script for this (may have to hack it a bit to point the paths in the right place)
- Update the `part_config_dict` dictionary in `part_config_render.py` with a new dictionary for any new tasks. See the existing ones for the information that should be included, and see `isaac_lab_rerender.py` for how this information is used (mainly to know where the `.usd` assets are and to have some name for the prims that are loaded into the scene)
- Ensure Isaac Sim and IsaacLab are set up (easiest way is to follow the `pip install` instructions on the [Isaac Lab documentation](https://isaac-sim.github.io/IsaacLab/source/setup/installation/pip_installation.html) - this will require a different `conda` environment with `python=3.10`)
- Run `isaac_lab_rerender.py` with all the relevant information specified (mainly, what task to re-render, what folder to load `.pkl` trajectories from, whether domain randomization should be used or not (there is still a bug in the script that does not parse these flags properly - manually set domain randomization on/off in the script if needed until it's resolved), and whether to save the new data + where to save it). See examples in the `scripts` folder.

Then, follow the same procedure you did at the beginning with your demos -- process the pickle files into `zarr`s and train a model!

You will find in this folder both `isaac_lab_rerender.py` and `isaac_sim_raytrace.py`. These are very similar. The former is newer and has been updated to utilize Isaac Sim 4.1 and Isaac Lab, but is still under development to work out some final kinks. The latter has been used and tested with Isaac Sim 2022.2.1 and `Orbit` (the tool that preceded Isaac Lab), neither of which appears to be available to install anymore. We include both for reference, but moving forward, the plan is to use the Isaac Lab version (as it is easier to install with `pip` on remote machines, and has been updated with more features that will be useful to incorporate in the future like tiled image rendering). 