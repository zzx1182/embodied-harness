"""runtime：打包上下文、调模型、校验、分配身份、提交、处理回执、预算和终止。"""
import json
import time
import uuid
from pathlib import Path

from .contracts import DecisionProposal, ActionRequest, WHITELIST, Spec
import numpy as np
from .validator import validate


class LeakError(RuntimeError):
    """白名单外的键进了上下文。"""


class Transport:
    """进程内传输。drop_next_receipt 用来在测试里吞掉一次回执。"""

    def __init__(self, executor):
        self.executor = executor
        self.drop_next_receipt = False
        self.submitted = 0

    def submit(self, req, proposal):
        self.submitted += 1
        receipt = self.executor.submit(req, proposal)
        if self.drop_next_receipt:
            self.drop_next_receipt = False
            return None
        return receipt

    def query(self, request_id):
        return self.executor.query(request_id)


def scan_whitelist(obj, path="obs"):
    """递归扫描，任何一层出现名单外的观测键都抛错。"""
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k.endswith("_pos") or k.endswith("_quat") or k.endswith("_image") or k.startswith("robot0_"):
                if k not in WHITELIST:
                    raise LeakError(f"{path}.{k} 不在白名单里")
            scan_whitelist(v, f"{path}.{k}")
    elif isinstance(obj, (list, tuple)):
        for i, v in enumerate(obj):
            scan_whitelist(v, f"{path}[{i}]")


def build_context(bundle, history, spec: Spec):
    ctx = {"obs_seq": bundle.obs_seq, "instruction": bundle.instruction, "control_step": bundle.control_step,
           "sim_time_s": round(bundle.sim_time_s, 3), "proprio": bundle.proprio,
           "last_receipt": bundle.last_receipt_summary,
           "history": history[-spec.history_segments:]}
    scan_whitelist(ctx["proprio"])
    scan_whitelist(ctx["history"], "history")
    return ctx


def parse_decision(decision, raw=None):
    if isinstance(decision, dict) and "_error" in decision:
        return None, {"code": "MODEL_ERROR", "got": decision["_error"][:300]}
    if not isinstance(decision, dict) or "_parse_error" in decision:
        return None, {"code": "PARSE_ERROR", "got": str(decision)[:300]}
    tool = decision.get("tool")
    seq = decision.get("expected_obs_seq")
    if not isinstance(seq, int) or isinstance(seq, bool):
        seq = None
    if tool == "give_up":
        return DecisionProposal(tool="give_up", expected_obs_seq=seq, reason=str(decision.get("reason", "")),
                                reason_category=decision.get("reason_category"), raw=decision), None
    if tool == "accept_vla":
        return DecisionProposal(tool="accept_vla", expected_obs_seq=seq, horizon_steps=decision.get("horizon_steps"),
                                reason=str(decision.get("reason", "")), raw=decision), None
    if tool == "move_eef":
        h = decision.get("horizon_steps")
        return DecisionProposal(tool="move_eef", expected_obs_seq=seq, target=decision.get("target"),
                                horizon_steps=h, reason=str(decision.get("reason", "")), raw=decision), None
    return None, {"code": "BAD_TOOL", "got": tool}


class Runtime:
    """一个 episode 的决策循环。mode 决定每段谁出动作：
    direct      大模型给 move_eef
    vla         π₀.₅ 出候选，原样执行前缀
    hybrid      π₀.₅ 出候选，大模型审核后 accept_vla 或 move_eef
    orchestrate 大模型只下子指令（run_skill）和判断子目标，动作全由 π₀.₅ 产生（RoboProbe L2）
    scripted    写死的子指令顺序，每条给固定的动作块预算，不看观测
    """

    def __init__(self, executor, model, spec: Spec = None, out_dir=None, transport=None, policy=None, mode="direct", plan=None,
                 progress=True):
        self.executor = executor
        self.progress = progress              # 每个决策段结束在终端打一行进度
        self.seg_no = 0
        self.fatal = None                     # 接口配置错误（密钥、模型名、额度），重试没用，停机
        self.model = model
        self.policy = policy
        self.mode = mode
        self.plan = list(plan or [])          # scripted: [(instruction, chunk_budget), ...]
        self.plan_index = 0
        self.plan_chunks_done = 0
        self.policy_calls = 0
        self.policy_wait_s = 0.0
        self.segments = 0
        self.spec = spec or executor.spec
        self.transport = transport or Transport(executor)
        self.out_dir = Path(out_dir) if out_dir else None
        self.history = []
        self.declared = []
        self.model_calls = 0                  # 拿到了决策的调用，计入调用预算
        self.model_errors = 0                 # 接口失败（容量不足、网络），不计入调用预算，也不算模型的拒绝
        self.consecutive_model_errors = 0
        self.rejections = 0
        self.consecutive_rejections = 0
        self.model_wait_s = 0.0
        self.infra_wait_s = 0.0               # 接口失败的调用耗时加退避等待，不计入墙钟预算
        self.stop_reason = None
        self.started = None
        if policy is not None:
            executor.set_policy(getattr(policy, "identity", None))

    # ---- helpers ----
    def _segment_dir(self):
        self.seg_no += 1
        if not self.out_dir:
            return None
        d = self.out_dir / "segments" / f"{self.segments:03d}"
        self.segments += 1
        d.mkdir(parents=True, exist_ok=True)
        return d

    def _dump(self, d, name, obj):
        if d is not None:
            (d / name).write_text(json.dumps(obj, ensure_ascii=False, indent=2, default=str))

    def _save_images(self, seg, prefix=""):
        if seg is not None:
            from PIL import Image
            for k, arr in self.executor.get_images().items():
                Image.fromarray(arr).save(seg / f"{prefix}{k}.png")

    def new_request(self, bundle, proposal):
        req = ActionRequest(request_id=uuid.uuid4().hex[:16], episode_id=bundle.episode_id,
                            executor_generation=bundle.executor_generation, obs_seq=bundle.obs_seq,
                            proposal={k: v for k, v in proposal.raw.items()} if isinstance(proposal.raw, dict) else {"tool": proposal.tool})
        if proposal.candidate is not None:
            req.proposal["candidate"] = [[round(float(v), 6) for v in row] for row in proposal.candidate]
        if proposal.candidate_binding is not None:
            req.proposal["candidate_binding"] = dict(proposal.candidate_binding)
        req.compute_hash()
        return req

    def recover(self, req):
        """回执没到：拿 request_id 去账本查，查到终态为止；没有记录就同 ID 同内容重发。"""
        deadline = time.monotonic() + self.spec.query_budget_s
        while time.monotonic() < deadline:
            r = self.transport.query(req.request_id)
            if r is None:
                return "RESEND"
            if r.status in ("COMPLETED", "CANCELLED", "REJECTED", "UNKNOWN"):
                return r
            time.sleep(0.05)
        return "TIMEOUT"

    def _say(self, text):
        if self.progress:
            print(f"  段 {self.seg_no - 1:03d} | {text}", flush=True)

    def _progress(self, tool, outcome):
        self._say(f"{tool} | {outcome} | 控制步 {self.executor.control_step}/{self.spec.max_control_steps} | "
                  f"模型调用 {self.model_calls}/{self.spec.max_model_calls}")

    def budget_exhausted(self):
        ex = self.executor
        if self.fatal:
            return "infra:model_config"
        if ex.control_step >= self.spec.max_control_steps:
            return "budget:control_steps"
        if self.model_calls >= self.spec.max_model_calls:
            return "budget:model_calls"
        if self.started is not None and time.monotonic() - self.started - self.infra_wait_s >= self.spec.max_wall_s:
            return "budget:wall_clock"
        if self.consecutive_rejections >= self.spec.max_consecutive_rejections:
            return "budget:consecutive_rejections"
        if self.consecutive_model_errors >= self.spec.max_model_errors:
            return "infra:model_unavailable"
        return None

    def _ask_model(self, bundle, ctx, seg):
        """调一次模型。接口失败时退避后返回 None，不算模型的拒绝，也不计入调用预算。"""
        ex = self.executor
        decision, usage, elapsed = self.model.decide(ctx, ex.get_images(), out_dir=seg)
        self.model_wait_s += elapsed
        if isinstance(decision, dict) and "_fatal" in decision:
            self.model_errors += 1
            self.fatal = decision["_fatal"][:300]
            ex.ledger.model_call(bundle.episode_id, bundle.obs_seq, elapsed, usage, "error", seg / "model_raw.json" if seg else None)
            self._dump(seg, "decision.json", decision)
            self._dump(seg, "validation.json", {"ok": False, "infrastructure": self.fatal, "fatal": True})
            self._say(f"接口配置错误，停机：{self.fatal}")
            print("  检查 HARNESS_MODEL_BASE_URL、HARNESS_MODEL_NAME、HARNESS_MODEL_API_KEY（或 Codex 登录与额度）后重跑。", flush=True)
            return None
        if isinstance(decision, dict) and "_error" in decision:
            self.model_errors += 1
            self.consecutive_model_errors += 1
            ex.ledger.model_call(bundle.episode_id, bundle.obs_seq, elapsed, usage, "error", seg / "model_raw.json" if seg else None)
            self._dump(seg, "decision.json", decision)
            self._dump(seg, "validation.json", {"ok": False, "infrastructure": decision["_error"][:300]})
            self.infra_wait_s += elapsed
            if self.consecutive_model_errors < self.spec.max_model_errors:
                self._say(f"接口失败（连续第 {self.consecutive_model_errors} 次，{self.spec.model_error_backoff_s:.0f} 秒后重试）："
                          f"{decision['_error'][:120]}")
                time.sleep(self.spec.model_error_backoff_s)
                self.infra_wait_s += self.spec.model_error_backoff_s
            return None
        self.consecutive_model_errors = 0
        self.model_calls += 1
        status = "ok" if isinstance(decision, dict) and "_parse_error" not in decision else "parse_error"
        ex.ledger.model_call(bundle.episode_id, bundle.obs_seq, elapsed, usage, status, seg / "model_raw.json" if seg else None)
        self._dump(seg, "decision.json", decision)
        return decision

    def _reject_decision(self, bundle, decision, err, seg):
        """校验失败：原样留档，回给模型拒绝原因，同一观测上重提。"""
        ex = self.executor
        req = ActionRequest(request_id=uuid.uuid4().hex[:16], episode_id=bundle.episode_id,
                            executor_generation=bundle.executor_generation, obs_seq=bundle.obs_seq,
                            proposal=decision if isinstance(decision, dict) else {"raw": str(decision)})
        req.compute_hash()
        ex.ledger.reject(req, err, eef_before=ex.eef())
        self.rejections += 1
        self.consecutive_rejections += 1
        self._dump(seg, "validation.json", {"ok": False, "rejection": err})
        self.history.append({"request_id": req.request_id, "status": "REJECTED", "rejection": err})

    def _submit(self, req, proposal, seg, name="receipt.json"):
        receipt = self.transport.submit(req, proposal)
        if receipt is None:
            got = self.recover(req)
            if got == "RESEND":
                receipt = self.transport.submit(req, proposal)
            elif got == "TIMEOUT":
                return None
            else:
                receipt = got
        self._dump(seg, name, receipt.__dict__)
        return receipt

    def _infer(self, bundle, instruction, seg, prefix=""):
        ex = self.executor
        candidate, p_elapsed = self.policy.infer(ex.get_images(), bundle.proprio, instruction)
        self.policy_calls += 1
        self.policy_wait_s += p_elapsed
        ex.ledger.policy_call(bundle.episode_id, bundle.obs_seq, p_elapsed, len(candidate))
        binding = ex.register_candidate(candidate)
        if seg is not None:
            np.save(seg / f"{prefix}candidate.npy", candidate)
            self._dump(seg, f"{prefix}candidate_binding.json", binding)
        return candidate, binding

    def prefix_limit(self, candidate):
        """一段最多执行的前缀：不超过 max_prefix_steps、候选长度和剩余控制步。"""
        remaining = self.spec.max_control_steps - self.executor.control_step
        return max(0, min(self.spec.max_prefix_steps, len(candidate), remaining))

    def _run_chunk(self, instruction, seg, prefix=""):
        """让 π₀.₅ 按给定指令出一块候选并执行前缀，作为一条独立请求记账。"""
        ex = self.executor
        bundle = ex.get_observation()
        candidate, binding = self._infer(bundle, instruction, seg, prefix)
        n = self.prefix_limit(candidate)
        raw = {"tool": "accept_vla", "expected_obs_seq": bundle.obs_seq, "horizon_steps": n,
               "reason": f"skill: {instruction}", "instruction": instruction, "candidate_rows": len(candidate)}
        proposal = DecisionProposal(tool="accept_vla", expected_obs_seq=bundle.obs_seq, horizon_steps=n,
                                    reason=raw["reason"], raw=raw, candidate=candidate.tolist(), candidate_binding=binding)
        req = self.new_request(bundle, proposal)
        return self._submit(req, proposal, seg, f"{prefix}receipt.json")

    # ---- one decision segment: direct / vla / hybrid ----
    def step(self):
        """走一个决策段。返回 stop_reason 或 None。"""
        if self.mode == "orchestrate":
            return self.step_orchestrate()
        if self.mode == "scripted":
            return self.step_scripted()
        ex = self.executor
        if ex.halted:
            return ex.halt_reason
        b = self.budget_exhausted()
        if b:
            return b
        if self.mode == "vla" and self.spec.max_policy_calls is not None and self.policy_calls >= self.spec.max_policy_calls:
            return "budget:policy_calls"
        bundle = ex.get_observation()
        ctx = build_context(bundle, self.history, self.spec)
        seg = self._segment_dir()
        self._save_images(seg)
        candidate = binding = None
        if self.mode in ("vla", "hybrid"):
            candidate, binding = self._infer(bundle, bundle.instruction, seg)
            prefix_limit = self.prefix_limit(candidate)
            ctx["candidate_action_prefix"] = [[round(float(v), 4) for v in row] for row in candidate[:prefix_limit]]
            ctx["candidate_horizon"] = int(len(candidate))
            ctx["max_prefix_steps"] = prefix_limit
        self._dump(seg, "context.json", ctx)
        if self.mode == "vla":
            decision = {"tool": "accept_vla", "expected_obs_seq": bundle.obs_seq, "horizon_steps": prefix_limit,
                        "reason": "VLA-only: execute the native candidate prefix"}
            self._dump(seg, "decision.json", decision)
        else:
            decision = self._ask_model(bundle, ctx, seg)
            if decision is None:
                return None
        proposal, err = parse_decision(decision)
        if err is None and proposal.tool == "accept_vla":
            if candidate is None:
                err = {"code": "NO_CANDIDATE", "got": "accept_vla without a policy candidate"}
            else:
                proposal.candidate = candidate.tolist()
                proposal.candidate_binding = binding
                proposal.raw = {**proposal.raw, "candidate_rows": len(candidate)}
        if err is None:
            o = ex.raw_obs_for_test()
            err = validate(proposal, bundle.obs_seq, o["robot0_eef_pos"], o["robot0_eef_quat"], self.spec,
                           session={"episode_id": bundle.episode_id, "executor_generation": bundle.executor_generation})
        if err is not None:
            self._reject_decision(bundle, decision, err, seg)
            self._progress(decision.get("tool", "?") if isinstance(decision, dict) else "?", f"REJECTED {err.get('code')}")
            return None
        self._dump(seg, "validation.json", {"ok": True})
        req = self.new_request(bundle, proposal)
        self._dump(seg, "request.json", {**req.payload(), "request_id": req.request_id, "payload_hash": req.payload_hash})
        receipt = self._submit(req, proposal, seg)
        if receipt is None or receipt.status == "UNKNOWN":
            self._progress(proposal.tool, "UNKNOWN")
            return "UNKNOWN"
        if receipt.status == "REJECTED":
            self.rejections += 1
            self.consecutive_rejections += 1
            self.history.append(receipt.summary())
            self._progress(proposal.tool, f"REJECTED {(receipt.rejection or {}).get('code')}")
            return None
        self.consecutive_rejections = 0
        self.history.append(receipt.summary())
        self._progress(proposal.tool, f"{receipt.status} {receipt.steps_executed} 步")
        if proposal.tool == "give_up":
            return "give_up"
        if receipt.native_success:
            return "native_success"
        return None

    # ---- orchestrate: model issues sub-instructions, π₀.₅ produces every action ----
    def step_orchestrate(self):
        ex = self.executor
        if ex.halted:
            return ex.halt_reason
        b = self.budget_exhausted()
        if b:
            return b
        bundle = ex.get_observation()
        ctx = build_context(bundle, self.history, self.spec)
        ctx["mode"] = "orchestrate"
        ctx["max_skill_chunks"] = self.spec.max_skill_chunks
        ctx["declared_subgoals"] = list(self.declared)
        seg = self._segment_dir()
        self._save_images(seg)
        self._dump(seg, "context.json", ctx)
        decision = self._ask_model(bundle, ctx, seg)
        if decision is None:
            return None
        if not isinstance(decision, dict) or "_parse_error" in decision:
            self._reject_decision(bundle, decision, {"code": "PARSE_ERROR", "got": str(decision)[:300]}, seg)
            return None
        tool = decision.get("tool")
        if decision.get("expected_obs_seq") != bundle.obs_seq:
            self._reject_decision(bundle, decision, {"code": "OBSERVATION_EXPIRED", "expected": decision.get("expected_obs_seq"),
                                                     "current": bundle.obs_seq}, seg)
            return None
        if tool == "run_skill":
            instruction = str(decision.get("instruction") or "").strip()
            chunks = decision.get("chunks")
            if not instruction or not isinstance(chunks, int) or isinstance(chunks, bool) or not 1 <= chunks <= self.spec.max_skill_chunks:
                self._reject_decision(bundle, decision, {"code": "BAD_SKILL_CALL", "instruction": instruction, "chunks": chunks,
                                                         "limit": self.spec.max_skill_chunks}, seg)
                return None
            self._dump(seg, "validation.json", {"ok": True})
            receipts = []
            for i in range(chunks):
                if ex.halted or ex.control_step >= self.spec.max_control_steps:
                    break
                r = self._run_chunk(instruction, seg, f"chunk{i:02d}_")
                if r is None or r.status == "UNKNOWN":
                    return "UNKNOWN"
                receipts.append(r)
                if r.native_success:
                    break
            self.consecutive_rejections = 0
            done = [r for r in receipts if r.status == "COMPLETED"]
            self.history.append({"tool": "run_skill", "instruction": instruction, "chunks_requested": chunks,
                                 "chunks_executed": len(done), "steps_executed": sum(r.steps_executed for r in done),
                                 "eef_before": receipts[0].eef_before if receipts else None,
                                 "eef_after": receipts[-1].eef_after if receipts else None})
            self._progress(f"run_skill「{instruction}」", f"{len(done)} 块 {sum(r.steps_executed for r in done)} 步")
            if receipts and receipts[-1].native_success:
                return "native_success"
            return None
        if tool in ("declare_subgoal_done", "finish"):
            subgoal = str(decision.get("subgoal") or ("ALL" if tool == "finish" else "")).strip()
            image = str(seg / "agentview_image.png") if seg else None
            ex.ledger.declare(bundle.episode_id, bundle.obs_seq, bundle.control_step,
                              "finish" if tool == "finish" else "subgoal", subgoal, ex.goal_progress(), image)
            self._dump(seg, "validation.json", {"ok": True})
            self.consecutive_rejections = 0
            self._progress(tool, f"RECORDED {subgoal}")
            if tool == "finish":
                return "model_finished"
            self.declared.append(subgoal)
            self.history.append({"tool": "declare_subgoal_done", "subgoal": subgoal, "status": "RECORDED"})
            return None
        if tool == "give_up":
            self._dump(seg, "validation.json", {"ok": True})
            return "give_up"
        self._reject_decision(bundle, decision, {"code": "BAD_TOOL", "got": tool}, seg)
        return None

    # ---- scripted: fixed sub-instruction order, fixed chunk budget each, no observation ----
    def step_scripted(self):
        ex = self.executor
        if ex.halted:
            return ex.halt_reason
        if ex.control_step >= self.spec.max_control_steps:
            return "budget:control_steps"
        if self.plan_index >= len(self.plan):
            return "plan_exhausted"
        instruction, budget = self.plan[self.plan_index]
        seg = self._segment_dir()
        self._save_images(seg)
        self._dump(seg, "context.json", {"mode": "scripted", "instruction": instruction, "plan_index": self.plan_index,
                                         "chunk": self.plan_chunks_done, "chunk_budget": budget})
        r = self._run_chunk(instruction, seg)
        if r is None or r.status == "UNKNOWN":
            return "UNKNOWN"
        self.plan_chunks_done += 1
        if self.plan_chunks_done >= budget:
            self.plan_index += 1
            self.plan_chunks_done = 0
        if r.native_success:
            return "native_success"
        return None

    def run_episode(self):
        self.started = time.monotonic()
        while True:
            stop = self.step()
            if stop:
                self.stop_reason = stop
                break
        ex = self.executor
        ex.ledger.set_meta(stop_reason=self.stop_reason, model_calls=self.model_calls, model_errors=self.model_errors,
                           rejections=self.rejections, mode=self.mode, plan=self.plan,
                           policy_calls=self.policy_calls, policy_wait_s=round(self.policy_wait_s, 3),
                           policy_identity=getattr(self.policy, "identity", None),
                           wall_s=round(time.monotonic() - self.started, 3), model_wait_s=round(self.model_wait_s, 3),
                           infra_wait_s=round(self.infra_wait_s, 3),
                           model_identity=getattr(self.model, "identity", {}))
        return self.stop_reason
