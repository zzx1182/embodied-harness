"""「ROS2 Action 网关与执行回执」「具身 Agent 评测、扰动与任务编排」两课新增能力的用例：取消、接口失败退避、编排模式、写死顺序、误验收审计、汇总归因。

    python -m unittest -v tests/test_extensions.py
"""
import json
import sys
import tempfile
import threading
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from harness.contracts import Spec, ActionRequest
from harness.envs import FakeEnv
from harness.executor import Executor
from harness.models import ScriptedModel
from harness.runtime import Runtime, build_context, parse_decision
from harness.report import build_report
from harness.aggregate import classify


class FakePolicy:
    """每次返回 10 步往 +x 走的候选。"""
    identity = {"kind": "fake"}

    def __init__(self):
        self.instructions = []

    def infer(self, images, proprio, instruction):
        self.instructions.append(instruction)
        c = np.zeros((10, 7))
        c[:, 0] = 0.5
        return c, 0.001


class ErrorThenOkModel(ScriptedModel):
    """前两次返回接口失败，之后按脚本给建议。"""

    def __init__(self, script, errors=2):
        super().__init__(script)
        self.errors = errors

    def decide(self, context, images, out_dir=None):
        if self.errors:
            self.errors -= 1
            return {"_error": "codex exit 1: Selected model is at capacity."}, None, 0.01
        return super().decide(context, images, out_dir)


def move(ctx, dx=0.02, steps=3):
    pos = list(ctx["proprio"]["robot0_eef_pos"]); pos[0] += dx
    return {"tool": "move_eef", "expected_obs_seq": ctx["obs_seq"],
            "target": {"frame_id": "robot0_base", "position_m": pos,
                       "quaternion_xyzw": list(ctx["proprio"]["robot0_eef_quat"]), "gripper": "open"},
            "horizon_steps": steps, "reason": "test"}


class Case(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.ledger = Path(self.tmp.name) / "ledger.sqlite"

    def tearDown(self):
        self.tmp.cleanup()

    def test_11_cancel_mid_segment(self):
        ex = Executor(FakeEnv(), self.ledger, Spec())
        ctx = build_context(ex.get_observation(), [], ex.spec)
        p, _ = parse_decision(move(ctx, steps=5))
        req = ActionRequest("req-c", ex.episode_id, ex.generation, ctx["obs_seq"], p.raw); req.compute_hash()
        cancel = threading.Event()
        seen = []

        def on_step(n):
            seen.append(n)
            if n == 2:
                cancel.set()
        r = ex.submit(req, p, on_step=on_step, cancel_event=cancel)
        self.assertEqual(r.status, "CANCELLED")
        self.assertEqual(r.steps_executed, 2)
        self.assertEqual(ex.control_step, 2)
        self.assertEqual(seen, [1, 2])
        self.assertLessEqual(len(seen), r.steps_executed)

    def test_12_model_error_is_not_a_rejection(self):
        spec = Spec(max_model_calls=5, model_error_backoff_s=0.0, max_consecutive_rejections=2)
        ex = Executor(FakeEnv(), self.ledger, spec)
        rt = Runtime(ex, ErrorThenOkModel([lambda c: move(c)], errors=3), spec)
        rt.started = None
        for _ in range(4):
            rt.step()
        self.assertEqual(rt.model_errors, 3)
        self.assertEqual(rt.rejections, 0)
        self.assertEqual(rt.model_calls, 1)
        self.assertEqual(ex.control_step, 3)

    def test_13_orchestrate_and_audit(self):
        spec = Spec(max_control_steps=200, max_model_calls=10, max_skill_chunks=5)
        env = FakeEnv(success_after_steps=10_000)
        ex = Executor(env, self.ledger, spec)
        policy = FakePolicy()
        script = [
            lambda c: {"tool": "run_skill", "expected_obs_seq": c["obs_seq"], "instruction": "pick up the fake goal", "chunks": 2, "subgoal": "", "reason": ""},
            lambda c: {"tool": "declare_subgoal_done", "expected_obs_seq": c["obs_seq"], "instruction": "", "chunks": 0, "subgoal": "fake goal", "reason": ""},
            lambda c: {"tool": "finish", "expected_obs_seq": c["obs_seq"], "instruction": "", "chunks": 0, "subgoal": "", "reason": ""},
        ]
        rt = Runtime(ex, ScriptedModel(script), spec, policy=policy, mode="orchestrate")
        stop = rt.run_episode()
        self.assertEqual(stop, "model_finished")
        self.assertEqual(ex.control_step, 20)
        self.assertEqual(policy.instructions, ["pick up the fake goal"] * 2)
        rep = build_report(self.ledger)
        self.assertEqual(rep["false_acceptances"], 2)    # 子目标和 finish 都在真值不成立时宣布
        self.assertEqual(rep["items_done"], 0)

    def test_14_scripted_plan(self):
        spec = Spec(max_control_steps=100)
        ex = Executor(FakeEnv(), self.ledger, spec)
        policy = FakePolicy()
        rt = Runtime(ex, None, spec, policy=policy, mode="scripted", plan=[("first", 2), ("second", 1)])
        self.assertEqual(rt.run_episode(), "plan_exhausted")
        self.assertEqual(policy.instructions, ["first", "first", "second"])
        self.assertEqual(ex.control_step, 30)

    def test_15_classify(self):
        stuck = {"native_success": False, "stop_reason": "budget:model_calls",
                 "segments": [{"status": "COMPLETED", "tool": "move_eef", "target_pos": [0, 0, 0.1],
                               "eef_before": {"pos": [0, 0, 0.1]}, "eef_after": {"pos": [0, 0, 0.1]}}] * 6}
        self.assertEqual(classify(stuck), "决策")
        infra = {"native_success": False, "stop_reason": "budget:consecutive_rejections",
                 "segments": [{"status": "REJECTED", "rejection": {"code": "MODEL_ERROR"}}] * 3}
        self.assertEqual(classify(infra), "基础设施")
        limit = {"native_success": False, "stop_reason": "budget:consecutive_rejections",
                 "segments": [{"status": "REJECTED", "rejection": {"code": "LIMIT"}}] * 3}
        self.assertEqual(classify(limit), "校验拒绝")
        self.assertIsNone(classify({"native_success": True}))


if __name__ == "__main__":
    unittest.main(verbosity=2)
