"""环境适配器。LiberoEnv 包真仿真，FakeEnv 给回归用例用，两个接口一样。"""
import hashlib
import os
import re

import numpy as np

# LIBERO 的初态文件是老版本的 torch pickle，torch 2.6 起 torch.load 默认拒读，这里放开。只读 LIBERO 自带的文件。
os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")

from .contracts import IMAGE_KEYS


def read_bddl_language(path):
    """取 bddl 文件里 (:language ...) 那一行的内容，没有就返回 None。"""
    try:
        with open(path, encoding="utf-8") as f:
            for line in f:
                m = re.search(r"\(:language\s+(.*?)\)\s*$", line.strip())
                if m:
                    return m.group(1).strip()
    except OSError:
        return None
    return None


class LiberoEnv:
    """LIBERO 环境。step 接 7 维归一化命令，返回观测字典；check_success 读环境原生判定。"""

    def __init__(self, suite="libero_object", task_id=0, seed=0, init_index=0, size=256, settle_steps=10):
        os.environ.setdefault("MUJOCO_GL", "egl")
        from libero.libero import benchmark, get_libero_path
        from libero.libero.envs import OffScreenRenderEnv

        bench = benchmark.get_benchmark_dict()[suite]()
        task = bench.get_task(task_id)
        bddl = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
        self.env = OffScreenRenderEnv(bddl_file_name=bddl, camera_heights=size, camera_widths=size)
        self.env.seed(seed)
        self.env.reset()
        self.init_state = np.asarray(bench.get_task_init_states(task_id)[init_index])
        self.obs = self.env.set_init_state(self.init_state)
        # 让夹爪张开、机械臂静止几步再交给决策方，和作者探针一致
        for _ in range(settle_steps):
            self.obs, *_ = self.env.step([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0])
        # LIBERO 的 task.language 是从 bddl 文件名推出来的。LIBERO-Pro 的 Task 扰动只改 bddl 里的
        # (:language ...) 和 (:goal ...)，文件名不变，所以 _task 套件要读 bddl 里那一行，否则发出去的是旧指令。
        bddl_language = read_bddl_language(bddl)
        self.instruction = bddl_language if (suite.endswith("_task") and bddl_language) else task.language
        self.meta = {"suite": suite, "task_id": task_id, "seed": seed, "init_state_index": init_index,
                     "settle_steps": settle_steps, "size": size, "instruction": self.instruction,
                     "filename_language": task.language, "bddl_language": bddl_language, "bddl_file": task.bddl_file,
                     # 初态数组按 float64 取 sha256：同一 suite、task、init 的哈希相同，拿它证明几种模式用的是同一个初态
                     "init_state_sha256": hashlib.sha256(np.ascontiguousarray(self.init_state, dtype=np.float64).tobytes()).hexdigest()}

    def observe(self):
        return self.obs

    def step(self, action):
        self.obs, reward, done, info = self.env.step([float(a) for a in action])
        return self.obs

    def check_success(self):
        return bool(self.env.check_success())

    def goal_progress(self):
        """逐条评估 bddl 里的目标谓词，只给评分用，不进模型上下文。"""
        dom = self.env.env
        out = []
        for g in dom.parsed_problem["goal_state"]:
            out.append({"predicate": " ".join(str(x) for x in g), "value": bool(dom._eval_predicate(g))})
        return out

    def images(self):
        """两路画面，转 180 度和策略约定一致。"""
        return {k: np.ascontiguousarray(self.obs[k][::-1, ::-1]) for k in IMAGE_KEYS}

    def close(self):
        self.env.close()


class FakeEnv:
    """没有 MuJoCo 的假环境。每步末端向目标靠近一半，成功由测试自己设。"""

    def __init__(self, extra_keys=None, success_after_steps=None):
        self.eef_pos = np.array([-0.15, 0.0, 0.25])
        self.eef_quat = np.array([1.0, 0.0, 0.0, 0.0])
        self.gripper = np.array([0.04, -0.04])
        self.n = 0
        self.extra_keys = dict(extra_keys or {})
        self.success_after_steps = success_after_steps
        self.instruction = "fake task: move the end effector"
        self.target = None
        self.meta = {"suite": "fake", "task_id": 0, "seed": 0, "init_state_index": 0}

    def observe(self):
        img = np.zeros((8, 8, 3), dtype=np.uint8)
        obs = {"agentview_image": img, "robot0_eye_in_hand_image": img,
               "robot0_joint_pos": np.zeros(7), "robot0_joint_vel": np.zeros(7),
               "robot0_eef_pos": self.eef_pos.copy(), "robot0_eef_quat": self.eef_quat.copy(),
               "robot0_gripper_qpos": self.gripper.copy()}
        obs.update(self.extra_keys)
        return obs

    def step(self, action):
        a = np.asarray(action, dtype=float)
        self.eef_pos = self.eef_pos + np.clip(a[:3], -1, 1) * 0.05 * 0.5
        self.n += 1
        return self.observe()

    def check_success(self):
        return self.success_after_steps is not None and self.n >= self.success_after_steps

    def goal_progress(self):
        return [{"predicate": "in fake_goal_1 box_1_contain_region", "value": self.check_success()}]

    def images(self):
        o = self.observe()
        return {k: o[k] for k in IMAGE_KEYS}

    def close(self):
        pass
