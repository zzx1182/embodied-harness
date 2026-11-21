"""「LIBERO 仿真环境与第一条可执行动作」附件的公共函数：建环境、过滤观测、检查动作、读末端、写账本。

四个脚本都只用这里的函数。改动作值、步数、任务时改命令行参数，不改这个文件。
"""
import json
import math
import os
from pathlib import Path

import numpy as np

# LIBERO 的初态文件是老版本的 torch pickle，torch 2.6 起 torch.load 默认拒读，这里放开。只读 LIBERO 自带的文件。
os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")

# 交给动作发出方的观测白名单，一共七项。其余键一律不出这个文件。
WHITELIST = (
    "agentview_image",
    "robot0_eye_in_hand_image",
    "robot0_joint_pos",
    "robot0_joint_vel",
    "robot0_eef_pos",
    "robot0_eef_quat",
    "robot0_gripper_qpos",
)

# OSC_POSE 控制器的单步量程：位置 ±0.05 m，姿态 ±0.5 rad。
POS_SCALE_M = 0.05


def make_env(suite="libero_object", task_id=0, seed=0, init_index=0, size=256):
    """建一个 LIBERO 环境，reset 到固定初态，返回 (env, obs, 任务指令)。"""
    os.environ.setdefault("MUJOCO_GL", "egl")
    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv

    bench = benchmark.get_benchmark_dict()[suite]()
    task = bench.get_task(task_id)
    bddl = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
    env = OffScreenRenderEnv(bddl_file_name=bddl, camera_heights=size, camera_widths=size)
    env.seed(seed)
    env.reset()
    obs = env.set_init_state(bench.get_task_init_states(task_id)[init_index])
    return env, obs, task.language


def split_obs(obs):
    """把观测字典拆成 (白名单内, 被过滤掉的键名列表)。"""
    kept = {k: obs[k] for k in WHITELIST if k in obs}
    dropped = sorted(k for k in obs if k not in WHITELIST)
    missing = [k for k in WHITELIST if k not in obs]
    if missing:
        raise KeyError(f"观测里缺少白名单字段: {missing}")
    return kept, dropped


def check_action(action):
    """三条断言：维度、有限数、范围。顺序有讲究，先查有限数再查范围。"""
    assert len(action) == 7, f"动作应为 7 维，收到 {len(action)} 维"
    for i, v in enumerate(action):
        assert math.isfinite(v), f"第 {i} 维不是有限数，收到 {v}"
        assert -1.0 <= v <= 1.0, f"第 {i} 维超出 [-1, 1]，收到 {v}"
    return action


def eef(obs):
    """末端位置（米，基座系）和姿态四元数，四舍五入到 0.01 mm。"""
    return {
        "pos": [round(float(x), 5) for x in obs["robot0_eef_pos"]],
        "quat": [round(float(x), 5) for x in obs["robot0_eef_quat"]],
    }


def save_image(array, path):
    """把仿真器给的 RGB 数组存成 PNG。robosuite 离屏渲染的画面是倒的，存之前转 180 度。"""
    from PIL import Image

    Image.fromarray(np.ascontiguousarray(array[::-1, ::-1])).save(path)


def jsonable(x):
    if isinstance(x, np.ndarray):
        return [jsonable(v) for v in x.tolist()]
    if isinstance(x, (np.floating, float)):
        return round(float(x), 6)
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, dict):
        return {k: jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [jsonable(v) for v in x]
    return x


class Ledger:
    """JSON 行账本，一条请求一行，只追加不改写。"""

    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def rows(self):
        if not self.path.exists():
            return []
        with open(self.path, encoding="utf-8") as f:
            return [json.loads(line) for line in f if line.strip()]

    def next_request_id(self):
        return f"{len(self.rows()) + 1:04d}"

    def append(self, row):
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(jsonable(row), ensure_ascii=False) + "\n")
        return row
