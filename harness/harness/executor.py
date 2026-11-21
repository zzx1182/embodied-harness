"""单写执行器。一条 episode 只有它能对环境步进和写账本。"""
import json
import threading
import time
import uuid

import numpy as np
from scipy.spatial.transform import Rotation

from .contracts import Spec, ExecutionReceipt, PROPRIO_KEYS, ObservationBundle, payload_hash, policy_hash
from .ledger import Ledger
from .validator import validate, GRIPPER


class ExecutorCrash(RuntimeError):
    """测试用：在 ACCEPTED 落盘之后、COMPLETED 落盘之前让执行器退出。"""


class Executor:
    def __init__(self, env, ledger_path, spec: Spec = None, episode_id=None, run_id=None,
                 fault_crash_after_accept=False, pace_s=0.0):
        self.env = env
        self.spec = spec or Spec()
        self.ledger = Ledger(ledger_path)
        self.lock = threading.Lock()
        self.fault_crash_after_accept = fault_crash_after_accept
        self.pace_s = pace_s          # 每步之后睡多久；ROS2 网关演示用它模拟实时控制，平时为 0
        # 重启：没有终态的请求标 UNKNOWN，代际加一，旧代际的一切失效
        unknown = self.ledger.mark_unknown_unfinished()
        self.generation = int(self.ledger.meta("executor_generation", 0)) + 1
        self.halted = bool(unknown)
        self.halt_reason = "UNKNOWN" if unknown else None
        self.episode_id = episode_id or self.ledger.meta("episode_id") or uuid.uuid4().hex[:12]
        self.run_id = run_id or self.ledger.meta("run_id") or uuid.uuid4().hex[:12]
        self.control_step = 0
        self.obs_seq = 0
        self.snapshot = None
        self.last_receipt = None
        self.policy_identity = None     # runtime 接入策略服务时设置
        self.candidates = {}            # candidate_id -> 登记信息；执行器重启后登记表清空，旧候选全部作废
        self.ledger.set_meta(run_id=self.run_id, episode_id=self.episode_id, executor_generation=self.generation,
                             controller_spec_hash=self.spec.hash(), spec=self.spec.__dict__,
                             budgets={"max_control_steps": self.spec.max_control_steps,
                                      "max_model_calls": self.spec.max_model_calls, "max_wall_s": self.spec.max_wall_s},
                             env=getattr(env, "meta", {}))
        self._new_snapshot()
        self._score()

    # ---- observation ----
    def _new_snapshot(self):
        obs = self.env.observe()
        self.obs_seq += 1
        self.snapshot = {"obs_seq": self.obs_seq, "obs": obs, "images": self.env.images(),
                         "control_step": self.control_step}

    def _score(self):
        """真值只写进评分表，runtime 和模型都不读它。"""
        progress = getattr(self.env, "goal_progress", None)
        if progress is not None:
            self.ledger.score(self.episode_id, self.snapshot["obs_seq"], self.control_step, progress())

    def goal_progress(self):
        progress = getattr(self.env, "goal_progress", None)
        return progress() if progress is not None else []

    @property
    def sim_time_s(self):
        return self.control_step / self.spec.control_hz

    def eef(self):
        o = self.snapshot["obs"]
        return {"pos": [round(float(x), 5) for x in o["robot0_eef_pos"]],
                "quat": [round(float(x), 5) for x in o["robot0_eef_quat"]]}

    def get_observation(self):
        """同一物理状态重复取到同一份快照，序号不变。"""
        s = self.snapshot
        proprio = {k: np.asarray(s["obs"][k]).round(5).tolist() for k in PROPRIO_KEYS}
        return ObservationBundle(obs_seq=s["obs_seq"], episode_id=self.episode_id,
                                 executor_generation=self.generation, control_step=self.control_step,
                                 sim_time_s=self.sim_time_s, instruction=self.env.instruction,
                                 proprio=proprio, image_paths={},
                                 last_receipt_summary=self.last_receipt.summary() if self.last_receipt else None)

    def get_images(self):
        return self.snapshot["images"]

    def raw_obs_for_test(self):
        return self.snapshot["obs"]

    def advance_for_test(self, steps=1):
        """测试用：让环境在没有请求的情况下变化，产生新快照。"""
        for _ in range(steps):
            self.env.step([0, 0, 0, 0, 0, 0, 0])
            self.control_step += 1
        self._new_snapshot()

    # ---- candidates ----
    def set_policy(self, identity):
        with self.lock:
            self.policy_identity = identity

    def register_candidate(self, rows):
        """策略每出一块候选就登记一次，绑定当前快照序号、执行器代际和策略身份。返回的凭据随 accept_vla 请求一起提交。"""
        with self.lock:
            cid = uuid.uuid4().hex[:12]
            binding = {"candidate_id": cid, "obs_seq": self.snapshot["obs_seq"], "executor_generation": self.generation,
                       "policy_hash": policy_hash(self.policy_identity)}
            self.candidates[cid] = {**binding, "rows_hash": payload_hash(np.asarray(rows, dtype=float).tolist())}
            return dict(binding)

    def _check_candidate(self, req, proposal):
        """accept_vla 只能执行本执行器登记过、在当前快照上生成、来自当前策略的那一块候选，而且不能超出剩余控制步。"""
        b = proposal.candidate_binding or (req.proposal or {}).get("candidate_binding") or {}
        entry = self.candidates.get(b.get("candidate_id"))
        if entry is None:
            return {"code": "UNKNOWN_CANDIDATE", "got": b.get("candidate_id")}
        rows_hash = payload_hash(np.asarray(proposal.candidate, dtype=float).tolist())
        if any(b.get(k) != entry[k] for k in ("obs_seq", "executor_generation", "policy_hash")) or rows_hash != entry["rows_hash"]:
            return {"code": "CANDIDATE_MISMATCH", "candidate_id": entry["candidate_id"]}
        if entry["executor_generation"] != self.generation or entry["obs_seq"] != self.snapshot["obs_seq"]:
            return {"code": "STALE_CANDIDATE", "candidate_id": entry["candidate_id"],
                    "candidate_obs_seq": entry["obs_seq"], "current_obs_seq": self.snapshot["obs_seq"]}
        if entry["policy_hash"] != policy_hash(self.policy_identity):
            return {"code": "POLICY_MISMATCH", "candidate_id": entry["candidate_id"],
                    "candidate_policy": entry["policy_hash"], "current_policy": policy_hash(self.policy_identity)}
        remaining = self.spec.max_control_steps - self.control_step
        if proposal.horizon_steps > remaining:
            return {"code": "BAD_HORIZON", "got": proposal.horizon_steps, "remaining_control_steps": remaining}
        return None

    # ---- query ----
    def query(self, request_id):
        row = self.ledger.get(request_id)
        if not row:
            return None
        return self._receipt_from_row(row)

    def _receipt_from_row(self, row):
        return ExecutionReceipt(
            request_id=row["request_id"], status=row["status"], steps_requested=row["steps_requested"] or 0,
            steps_executed=row["steps_executed"] or 0,
            eef_before=json.loads(row["eef_before_json"]) if row["eef_before_json"] else None,
            eef_after=json.loads(row["eef_after_json"]) if row["eef_after_json"] else None,
            obs_seq_before=row["obs_seq"], obs_seq_after=row["obs_seq_after"],
            native_success=None if row["native_success"] is None else bool(row["native_success"]),
            rejection=json.loads(row["rejection_json"]) if row["rejection_json"] else None)

    # ---- submit ----
    def _precheck(self, req, proposal):
        """受理前的全部检查。返回 ("ok", None)、("duplicate", 旧回执) 或 ("reject", 拒绝回执)。"""
        if not req.payload_hash:
            req.compute_hash()
        if self.halted:
            return "reject", self._reject(req, {"code": "EPISODE_HALTED", "reason": self.halt_reason})
        # 去重：同 ID 同内容返回旧回执；同 ID 不同内容冲突
        existing = self.ledger.get(req.request_id)
        if existing:
            if existing["payload_hash"] == req.payload_hash:
                self.ledger.event(req.request_id, "DUPLICATE_RETURNED")
                return "duplicate", self._receipt_from_row(existing)
            self.ledger.event(req.request_id, "CONTENT_CONFLICT", got=req.payload_hash, existing=existing["payload_hash"])
            return "reject", ExecutionReceipt(request_id=req.request_id, status="REJECTED",
                                              rejection={"code": "CONTENT_CONFLICT", "existing_hash": existing["payload_hash"]})
        # 会话身份
        if req.episode_id != self.episode_id or req.executor_generation != self.generation:
            return "reject", self._reject(req, {"code": "BAD_SESSION",
                                                "got": {"episode_id": req.episode_id, "executor_generation": req.executor_generation},
                                                "expected": {"episode_id": self.episode_id, "executor_generation": self.generation}})
        # 执行器自己重算摘要，不信客户端带来的
        if payload_hash(req.payload()) != req.payload_hash:
            return "reject", self._reject(req, {"code": "BAD_HASH"})
        # 四道校验，和 runtime 用同一个函数
        o = self.snapshot["obs"]
        rejection = validate(proposal, self.snapshot["obs_seq"], o["robot0_eef_pos"], o["robot0_eef_quat"], self.spec,
                             session={"episode_id": self.episode_id, "executor_generation": self.generation})
        # 候选核对只在执行器做，登记表在这里
        if not rejection and proposal.tool == "accept_vla":
            rejection = self._check_candidate(req, proposal)
        if rejection:
            return "reject", self._reject(req, rejection)
        return "ok", None

    def check(self, req, proposal):
        """只做受理前检查，不执行。ROS2 网关在决定接不接 goal 时用它。"""
        with self.lock:
            return self._precheck(req, proposal)

    def submit(self, req, proposal, on_step=None, cancel_event=None):
        """req 是 ActionRequest，proposal 是 DecisionProposal。返回 ExecutionReceipt。
        on_step(已确认步数) 每步之后调用一次；cancel_event 被置位后，在下一步之前停下。"""
        with self.lock:
            kind, receipt = self._precheck(req, proposal)
            if kind != "ok":
                return receipt
            if proposal.tool == "give_up":
                self.ledger.accept(req, 0, self.eef())
                self.ledger.complete(req.request_id, 0, self.eef(), self.snapshot["obs_seq"], self.env.check_success())
                self.halted, self.halt_reason = True, "give_up"
                r = self.query(req.request_id)
                self.last_receipt = r
                return r
            if proposal.tool == "accept_vla":
                rows = np.asarray(proposal.candidate, dtype=float)[: proposal.horizon_steps]
                return self._run_steps(req, proposal.horizon_steps, lambda i: rows[i], on_step, cancel_event)
            t = proposal.target
            target_pos = np.asarray(t["position_m"], dtype=float)
            target_rot = Rotation.from_quat(t["quaternion_xyzw"])
            gripper = GRIPPER[t["gripper"]]

            def eef_command(_i):
                cur = self.env.observe()
                pos_err = target_pos - np.asarray(cur["robot0_eef_pos"])
                rot_err = (target_rot * Rotation.from_quat(cur["robot0_eef_quat"]).inv()).as_rotvec()
                return np.r_[np.clip(pos_err / self.spec.pos_scale_m, -1, 1),
                             np.clip(rot_err / self.spec.rot_scale_rad, -1, 1), gripper]
            return self._run_steps(req, proposal.horizon_steps, eef_command, on_step, cancel_event)

    def _run_steps(self, req, steps, command_at, on_step, cancel_event):
        """先落盘再执行；逐步确认，每步把已确认步数写进账本，崩溃后账本里是已确认的下界。"""
        before = self.eef()
        self.ledger.accept(req, steps, before)
        if self.fault_crash_after_accept:
            raise ExecutorCrash("simulated crash after ACCEPTED")
        self.ledger.running(req.request_id)
        executed = clipped = 0
        success = cancelled = False
        for i in range(steps):
            if cancel_event is not None and cancel_event.is_set():
                cancelled = True
                break
            raw = np.asarray(command_at(i), dtype=float)
            a = np.clip(raw, -1.0, 1.0)             # 候选里越界的分量裁到边界，计数写进事件
            clipped += int(np.any(a != raw))
            self.env.step(a)
            executed += 1
            self.control_step += 1
            self.ledger.progress(req.request_id, executed)
            if on_step is not None:
                on_step(executed)
            if self.pace_s:
                time.sleep(self.pace_s)
            if self.env.check_success():
                success = True
                break
        self._new_snapshot()
        success = success or self.env.check_success()
        status = "CANCELLED" if cancelled else "COMPLETED"
        self.ledger.complete(req.request_id, executed, self.eef(), self.snapshot["obs_seq"], success, status=status)
        if clipped:
            self.ledger.event(req.request_id, "CLIPPED_ROWS", count=clipped)
        self._score()
        r = self.query(req.request_id)
        r.sim_time_s = self.sim_time_s
        self.last_receipt = r
        if success:
            self.halted, self.halt_reason = True, "native_success"
        return r

    def _reject(self, req, rejection):
        self.ledger.reject(req, rejection, eef_before=self.eef())
        return ExecutionReceipt(request_id=req.request_id, status="REJECTED", rejection=rejection,
                                eef_before=self.eef(), eef_after=self.eef(), obs_seq_before=self.snapshot["obs_seq"])

    def close(self):
        self.ledger.close()
        self.env.close()
