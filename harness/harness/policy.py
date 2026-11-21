"""π₀.₅ 策略客户端。通过 openpi 的 WebSocket 协议向策略服务要候选动作块。

服务端是 openpi 的 websocket_policy_server（见「π₀.₅、Direct 与 Hybrid 控制」第一节），客户端包在
openpi 仓库的 packages/openpi-client/src 里，路径用 HARNESS_OPENPI_CLIENT_SRC 指定，
别直接 pip install 那个包：它的依赖写着 numpy<2，会把 .venv 里的 numpy 降级。
"""
import os
import sys
import time

import numpy as np
from scipy.spatial.transform import Rotation


# 预处理版本。改任何一处（缩放、旋转、state 拼接顺序）都要换版本号，manifest 和每条 episode 的策略身份里都记它。
PREPROCESS_VERSIONS = {
    "libero-v1": "图像转 180 度，resize_with_pad 到 224×224；state = 末端位置 3 + 旋转向量 3 + 夹爪开度 2",
    # 下面两个是故意做错的版本，只给冻结快照的故障实验用：服务端照样返回 (10, 7)，什么都不报
    "fault-no-rotate": "忘了转 180 度（把执行器转过的图再转回去），其余同 libero-v1",
    # pi05_libero 这份权重不读 state（Pi0Config(pi05=True, discrete_state_input=False)），这个故障对它没有影响，是个空实验
    "fault-state-order": "state 里旋转向量排在位置前面，其余同 libero-v1",
}


class PolicyClient:
    def __init__(self, host="127.0.0.1", port=43861, image_size=224, client_src=None, preprocess="libero-v1"):
        if preprocess not in PREPROCESS_VERSIONS:
            raise ValueError(f"未知的预处理版本 {preprocess}，可选 {list(PREPROCESS_VERSIONS)}")
        self.preprocess = preprocess
        src = client_src or os.environ.get("HARNESS_OPENPI_CLIENT_SRC")
        if src and src not in sys.path:
            sys.path.insert(0, src)
        from openpi_client.websocket_client_policy import WebsocketClientPolicy
        from openpi_client import image_tools

        self._image_tools = image_tools
        self.client = WebsocketClientPolicy(host=host, port=port)
        self.image_size = image_size
        self.identity = {"kind": "openpi_websocket", "host": host, "port": port,
                         "preprocess_version": preprocess, "preprocess": PREPROCESS_VERSIONS[preprocess]}
        try:
            self.identity["server_metadata"] = self.client.get_server_metadata()
        except Exception:
            pass

    def element(self, images, proprio, instruction):
        """把白名单观测变成策略输入。图像已经在执行器里转过 180 度。"""
        it = self._image_tools
        pos = np.asarray(proprio["robot0_eef_pos"], dtype=np.float32)
        rotvec = Rotation.from_quat(proprio["robot0_eef_quat"]).as_rotvec().astype(np.float32)
        grip = np.asarray(proprio["robot0_gripper_qpos"], dtype=np.float32)
        state = np.concatenate((rotvec, pos, grip) if self.preprocess == "fault-state-order" else (pos, rotvec, grip))
        view, wrist = images["agentview_image"], images["robot0_eye_in_hand_image"]
        if self.preprocess == "fault-no-rotate":
            view, wrist = view[::-1, ::-1], wrist[::-1, ::-1]
        return {"observation/image": it.resize_with_pad(np.ascontiguousarray(view), self.image_size, self.image_size),
                "observation/wrist_image": it.resize_with_pad(np.ascontiguousarray(wrist), self.image_size, self.image_size),
                "observation/state": state, "prompt": instruction}

    def infer(self, images, proprio, instruction):
        """返回 (候选动作块 ndarray[n,7], 耗时秒)。候选是原生归一化命令，未做任何修改。"""
        started = time.monotonic()
        out = self.client.infer(self.element(images, proprio, instruction))
        candidate = np.asarray(out["actions"], dtype=float)
        elapsed = time.monotonic() - started
        if candidate.ndim != 2 or candidate.shape[1] != 7 or len(candidate) < 1:
            raise ValueError(f"策略返回的候选形状不对: {candidate.shape}")
        return candidate, elapsed

    def close(self):
        ws = getattr(self.client, "_ws", None)
        if ws is not None:
            try:
                ws.close()
            except Exception:
                pass
