import os
import numpy as np
import robosuite, mujoco
from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv

bench = benchmark.get_benchmark_dict()["libero_object"]()
task = bench.get_task(0)
bddl = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
env = OffScreenRenderEnv(bddl_file_name=bddl, camera_heights=256, camera_widths=256)
env.seed(0)
env.reset()
obs = env.set_init_state(bench.get_task_init_states(0)[0])
before = obs["robot0_eef_pos"].copy()
for _ in range(15):
    obs, reward, done, info = env.step([0.2, 0, 0, 0, 0, 0, 0])
dx_mm = (obs["robot0_eef_pos"] - before)[0] * 1000
print("OK", obs["agentview_image"].shape, f"dx_mm={dx_mm:.2f}", robosuite.__version__, mujoco.__version__, np.__version__)

