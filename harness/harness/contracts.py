"""四个契约。每种结构固定由一方生成，其他方只读。"""
import hashlib
import json
from dataclasses import dataclass, field, asdict
from typing import Any, Optional

# 交给模型的观测白名单，一共七项，递归扫描时名单外的键一律算泄漏。
WHITELIST = (
    "agentview_image",
    "robot0_eye_in_hand_image",
    "robot0_joint_pos",
    "robot0_joint_vel",
    "robot0_eef_pos",
    "robot0_eef_quat",
    "robot0_gripper_qpos",
)
IMAGE_KEYS = ("agentview_image", "robot0_eye_in_hand_image")
PROPRIO_KEYS = tuple(k for k in WHITELIST if k not in IMAGE_KEYS)


@dataclass
class Spec:
    """执行边界。改数字改这里，不改校验代码。"""
    base_frame: str = "robot0_base"
    max_segment_distance_m: float = 0.05
    max_segment_rotation_rad: float = 0.35
    max_horizon_steps: int = 5
    max_prefix_steps: int = 10          # 候选动作块最多执行的前缀长度
    workspace_min: tuple = (-0.6, -0.6, 0.0)
    workspace_max: tuple = (0.6, 0.6, 1.2)
    controller: str = "OSC_POSE"
    control_hz: float = 20.0
    pos_scale_m: float = 0.05
    rot_scale_rad: float = 0.5
    image_size: int = 256
    history_segments: int = 2
    max_control_steps: int = 300
    max_model_calls: int = 40
    max_policy_calls: Optional[int] = None   # VLA-only 默认只受控制步预算约束；需要时另给策略调用上限
    max_wall_s: float = 1800.0
    max_consecutive_rejections: int = 5
    query_budget_s: float = 30.0
    model_error_backoff_s: float = 60.0   # 接口容量不足、网络断开时等多久再调，这类失败不算模型的拒绝
    max_model_errors: int = 30            # 连续这么多次接口失败（约半小时），记为基础设施失败停机
    max_skill_chunks: int = 5             # 编排模式里一次 run_skill 最多执行几个动作块

    def hash(self):
        return hashlib.sha256(json.dumps(asdict(self), sort_keys=True).encode()).hexdigest()[:16]


@dataclass
class ObservationBundle:
    """执行器生成，模型只读。图像数组不进这个结构，另存文件后放路径。"""
    obs_seq: int
    episode_id: str
    executor_generation: int
    control_step: int
    sim_time_s: float
    instruction: str
    proprio: dict                       # 五项本体字段
    image_paths: dict = field(default_factory=dict)
    last_receipt_summary: Optional[dict] = None


@dataclass
class DecisionProposal:
    """模型生成，runtime 校验。身份字段模型一个都不填。"""
    tool: str
    expected_obs_seq: Optional[int]
    target: Optional[dict] = None        # {frame_id, position_m, quaternion_xyzw, gripper}
    horizon_steps: Optional[int] = None
    reason: str = ""
    reason_category: Optional[str] = None  # give_up 用
    candidate: Optional[list] = None      # accept_vla 用，runtime 从策略拿到后贴上，模型不填
    candidate_binding: Optional[dict] = None  # accept_vla 用，执行器登记候选时发的凭据，模型不填
    raw: Any = None                      # 模型原始返回，原样留档


@dataclass
class ActionRequest:
    """runtime 生成，执行器再校验。"""
    request_id: str
    episode_id: str
    executor_generation: int
    obs_seq: int
    proposal: dict
    payload_hash: str = ""

    def payload(self):
        return {"episode_id": self.episode_id, "executor_generation": self.executor_generation,
                "obs_seq": self.obs_seq, "proposal": self.proposal}

    def compute_hash(self):
        self.payload_hash = payload_hash(self.payload())
        return self.payload_hash


@dataclass
class ExecutionReceipt:
    """执行器生成，模型和评测器只读。"""
    request_id: str
    status: str                          # COMPLETED / REJECTED / UNKNOWN / RUNNING
    steps_requested: int = 0
    steps_executed: int = 0
    eef_before: Optional[dict] = None
    eef_after: Optional[dict] = None
    obs_seq_before: Optional[int] = None
    obs_seq_after: Optional[int] = None
    native_success: Optional[bool] = None
    rejection: Optional[dict] = None
    sim_time_s: Optional[float] = None

    def summary(self):
        """回给模型的公开摘要，不含任何真值。"""
        s = {"request_id": self.request_id, "status": self.status,
             "steps_executed": self.steps_executed}
        if self.eef_before:
            s["eef_before"] = self.eef_before
        if self.eef_after:
            s["eef_after"] = self.eef_after
        if self.rejection:
            s["rejection"] = self.rejection
        return s


def canonical_json(obj):
    """键排序、浮点固定格式，任何字段变了摘要就变。"""
    def norm(x):
        if isinstance(x, float):
            return format(x, ".6f")
        if isinstance(x, dict):
            return {k: norm(v) for k, v in sorted(x.items())}
        if isinstance(x, (list, tuple)):
            return [norm(v) for v in x]
        return x
    return json.dumps(norm(obj), sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def payload_hash(obj):
    return hashlib.sha256(canonical_json(obj).encode("utf-8")).hexdigest()


def policy_hash(identity):
    """策略身份（服务地址、checkpoint、预处理版本等）的短摘要，候选靠它绑定到出它的那个策略。"""
    return payload_hash(identity or {})[:16]


def to_dict(x):
    return asdict(x) if hasattr(x, "__dataclass_fields__") else x
