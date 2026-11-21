"""对建议做四道校验，顺序固定：数值有限、坐标系、限幅、观测新鲜度。

返回 None 表示通过，否则返回一条拒绝记录（dict）。执行器不改数、不裁剪。
"""
import math

import numpy as np
from scipy.spatial.transform import Rotation

from .contracts import Spec

GRIPPER = {"open": -1.0, "close": 1.0}
TOOLS = ("move_eef", "give_up", "accept_vla")


def reject(code, **detail):
    return {"code": code, **detail}


def _finite_list(x, n):
    return isinstance(x, (list, tuple)) and len(x) == n and all(
        isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v) for v in x)


def validate(proposal, obs_seq, eef_pos, eef_quat, spec: Spec, session=None):
    """proposal 是 DecisionProposal。session 是 {episode_id, executor_generation}，用于第 8 条用例。"""
    if proposal.tool not in TOOLS:
        return reject("BAD_TOOL", got=proposal.tool)
    # 会话身份：来自别的 episode 或旧代际的建议，代号和观测过期区分开
    if session is not None and proposal.raw and isinstance(proposal.raw, dict):
        claimed = proposal.raw.get("session")
        if claimed and (claimed.get("episode_id") != session["episode_id"]
                        or claimed.get("executor_generation") != session["executor_generation"]):
            return reject("BAD_SESSION", got=claimed, expected=session)
    if proposal.tool == "give_up":
        if proposal.expected_obs_seq != obs_seq:
            return reject("OBSERVATION_EXPIRED", expected=proposal.expected_obs_seq, current=obs_seq)
        return None
    if proposal.tool == "accept_vla":
        c = proposal.candidate
        if c is None:
            return reject("NO_CANDIDATE")
        arr = np.asarray(c, dtype=float)
        if arr.ndim != 2 or arr.shape[1] != 7 or len(arr) < 1 or not np.isfinite(arr).all():
            return reject("BAD_CANDIDATE", shape=list(arr.shape))
        h = proposal.horizon_steps
        limit = min(spec.max_prefix_steps, len(arr))
        if not isinstance(h, int) or isinstance(h, bool) or not 1 <= h <= limit:
            return reject("BAD_HORIZON", got=h, limit=limit)
        if proposal.expected_obs_seq != obs_seq:
            return reject("OBSERVATION_EXPIRED", expected=proposal.expected_obs_seq, current=obs_seq)
        return None

    t = proposal.target or {}
    # 1. 数值有限
    if not _finite_list(t.get("position_m"), 3):
        return reject("NON_FINITE", field="position_m", got=t.get("position_m"))
    if not _finite_list(t.get("quaternion_xyzw"), 4):
        return reject("NON_FINITE", field="quaternion_xyzw", got=t.get("quaternion_xyzw"))
    qn = float(np.linalg.norm(t["quaternion_xyzw"]))
    if qn < 1e-6 or abs(qn - 1.0) > 0.05:
        return reject("BAD_QUATERNION", norm=qn)
    h = proposal.horizon_steps
    if not isinstance(h, int) or isinstance(h, bool) or not 1 <= h <= spec.max_horizon_steps:
        return reject("BAD_HORIZON", got=h, limit=spec.max_horizon_steps)
    if t.get("gripper") not in GRIPPER:
        return reject("BAD_ENUM", field="gripper", got=t.get("gripper"), allowed=list(GRIPPER))
    # 2. 坐标系
    if t.get("frame_id") != spec.base_frame:
        return reject("BAD_FRAME", got=t.get("frame_id"), expected=spec.base_frame)
    # 3. 限幅与工作域
    d = float(np.linalg.norm(np.asarray(t["position_m"]) - np.asarray(eef_pos)))
    if d > spec.max_segment_distance_m + 1e-9:
        return reject("LIMIT", field="position_m", got=round(d, 4), limit=spec.max_segment_distance_m,
                      current=[round(float(v), 4) for v in eef_pos])
    ang = float(np.linalg.norm((Rotation.from_quat(t["quaternion_xyzw"]) * Rotation.from_quat(eef_quat).inv()).as_rotvec()))
    if ang > spec.max_segment_rotation_rad + 1e-9:
        return reject("LIMIT", field="quaternion_xyzw", got=round(ang, 4), limit=spec.max_segment_rotation_rad)
    lo, hi = np.asarray(spec.workspace_min), np.asarray(spec.workspace_max)
    if not (np.all(np.asarray(t["position_m"]) >= lo) and np.all(np.asarray(t["position_m"]) <= hi)):
        return reject("WORKSPACE", got=t["position_m"], min=list(lo), max=list(hi))
    # 4. 观测新鲜度
    if proposal.expected_obs_seq != obs_seq:
        return reject("OBSERVATION_EXPIRED", expected=proposal.expected_obs_seq, current=obs_seq)
    return None
