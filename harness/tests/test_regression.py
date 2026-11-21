"""「模型建议、执行校验与可靠回执」第七节的十条回归用例。跑在假环境和脚本模型上，不需要 MuJoCo，不需要模型接口。

    python -m unittest -v tests/test_regression.py
"""
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from harness.contracts import Spec, ActionRequest
from harness.envs import FakeEnv
from harness.executor import Executor, ExecutorCrash
from harness.models import ScriptedModel
from harness.runtime import Runtime, Transport, build_context, LeakError, parse_decision
from harness.report import build_report, write_report


def move(ctx, dx=0.02, steps=3, gripper="open", **extra):
    pos = list(ctx["proprio"]["robot0_eef_pos"])
    pos[0] += dx
    d = {"tool": "move_eef", "expected_obs_seq": ctx["obs_seq"],
         "target": {"frame_id": "robot0_base", "position_m": pos,
                    "quaternion_xyzw": list(ctx["proprio"]["robot0_eef_quat"]), "gripper": gripper},
         "horizon_steps": steps, "reason": "test"}
    d.update(extra)
    return d


def proposal_from(ctx, **kw):
    p, err = parse_decision(move(ctx, **kw))
    assert err is None, err
    return p


class Case(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.ledger = Path(self.tmp.name) / "ledger.sqlite"
        self.spec = Spec(max_control_steps=50, max_model_calls=20)

    def tearDown(self):
        self.tmp.cleanup()

    def make(self, env=None, **kw):
        ex = Executor(env or FakeEnv(), self.ledger, self.spec, **kw)
        return ex

    def request(self, ex, proposal):
        b = ex.get_observation()
        req = ActionRequest(request_id="req-1", episode_id=b.episode_id, executor_generation=b.executor_generation,
                            obs_seq=b.obs_seq, proposal=proposal.raw)
        req.compute_hash()
        return req

    # 1 同 ID 同内容重发
    def test_01_duplicate_same_payload(self):
        ex = self.make()
        ctx = build_context(ex.get_observation(), [], self.spec)
        p = proposal_from(ctx)
        req = self.request(ex, p)
        r1 = ex.submit(req, p)
        self.assertEqual(r1.status, "COMPLETED")
        steps = ex.control_step
        r2 = ex.submit(req, p)
        self.assertEqual(r2.status, "COMPLETED")
        self.assertEqual(r2.steps_executed, r1.steps_executed)
        self.assertEqual(ex.control_step, steps)
        self.assertEqual(len(ex.ledger.requests()), 1)

    # 2 同 ID 不同内容
    def test_02_duplicate_different_payload(self):
        ex = self.make()
        ctx = build_context(ex.get_observation(), [], self.spec)
        p = proposal_from(ctx)
        req = self.request(ex, p)
        ex.submit(req, p)
        steps = ex.control_step
        p2 = proposal_from(ctx, steps=4)
        req2 = ActionRequest(request_id="req-1", episode_id=req.episode_id, executor_generation=req.executor_generation,
                             obs_seq=req.obs_seq, proposal=p2.raw)
        req2.compute_hash()
        r = ex.submit(req2, p2)
        self.assertEqual(r.status, "REJECTED")
        self.assertEqual(r.rejection["code"], "CONTENT_CONFLICT")
        self.assertEqual(ex.control_step, steps)
        self.assertIn("CONTENT_CONFLICT", [e["event"] for e in ex.ledger.events("req-1")])

    # 3 旧观测
    def test_03_stale_observation(self):
        ex = self.make()
        ctx = build_context(ex.get_observation(), [], self.spec)
        p = proposal_from(ctx)
        req = self.request(ex, p)
        ex.advance_for_test(2)
        steps = ex.control_step
        r = ex.submit(req, p)
        self.assertEqual(r.status, "REJECTED")
        self.assertEqual(r.rejection["code"], "OBSERVATION_EXPIRED")
        self.assertEqual(ex.control_step, steps)

    # 4 回执丢失
    def test_04_receipt_lost(self):
        ex = self.make()
        model = ScriptedModel([lambda c: move(c)])
        tr = Transport(ex)
        tr.drop_next_receipt = True
        rt = Runtime(ex, model, self.spec, transport=tr)
        rt.step()
        self.assertEqual(tr.submitted, 1)
        self.assertEqual(ex.control_step, 3)
        self.assertEqual(rt.history[-1]["status"], "COMPLETED")

    # 5 UNKNOWN 后停止
    def test_05_unknown_after_crash(self):
        ex = self.make(fault_crash_after_accept=True)
        ctx = build_context(ex.get_observation(), [], self.spec)
        p = proposal_from(ctx)
        req = self.request(ex, p)
        with self.assertRaises(ExecutorCrash):
            ex.submit(req, p)
        ex.ledger.close()
        ex2 = Executor(FakeEnv(), self.ledger, self.spec)
        r = ex2.query("req-1")
        self.assertEqual(r.status, "UNKNOWN")
        self.assertTrue(ex2.halted)
        rt = Runtime(ex2, ScriptedModel([lambda c: move(c)]), self.spec)
        self.assertEqual(rt.run_episode(), "UNKNOWN")
        b = ex2.get_observation()
        req_new = ActionRequest(request_id="req-2", episode_id=b.episode_id, executor_generation=b.executor_generation,
                                obs_seq=b.obs_seq, proposal=p.raw)
        req_new.compute_hash()
        self.assertEqual(ex2.submit(req_new, p).status, "REJECTED")

    # 6 超限
    def test_06_limit(self):
        ex = self.make()
        ctx = build_context(ex.get_observation(), [], self.spec)
        p = proposal_from(ctx, dx=0.2)
        r = ex.submit(self.request(ex, p), p)
        self.assertEqual(r.rejection["code"], "LIMIT")
        self.assertEqual(r.rejection["field"], "position_m")
        self.assertIn("got", r.rejection)
        self.assertEqual(r.rejection["limit"], 0.05)
        self.assertEqual(ex.control_step, 0)

    # 7 NaN / Inf
    def test_07_non_finite(self):
        ex = self.make()
        ctx = build_context(ex.get_observation(), [], self.spec)
        for bad in (float("nan"), float("inf")):
            d = move(ctx)
            d["target"]["position_m"][1] = bad
            p, err = parse_decision(d)
            r = ex.submit(ActionRequest(request_id=f"nf-{bad}", episode_id=ex.episode_id, executor_generation=ex.generation,
                                        obs_seq=ctx["obs_seq"], proposal=d), p)
            self.assertEqual(r.rejection["code"], "NON_FINITE")
        self.assertEqual(ex.control_step, 0)

    # 8 候选来源与观测不匹配
    def test_08_session_mismatch(self):
        ex = self.make()
        ctx = build_context(ex.get_observation(), [], self.spec)
        p = proposal_from(ctx)
        req = ActionRequest(request_id="req-x", episode_id="other-episode", executor_generation=ex.generation,
                            obs_seq=ctx["obs_seq"], proposal=p.raw)
        req.compute_hash()
        r = ex.submit(req, p)
        self.assertEqual(r.rejection["code"], "BAD_SESSION")
        self.assertNotEqual(r.rejection["code"], "OBSERVATION_EXPIRED")
        self.assertEqual(ex.control_step, 0)

    # 9 真值泄漏
    def test_09_leak(self):
        leaky = FakeEnv(extra_keys={"bowl_1_pos": [0.1, 0.2, 0.3]})
        ex = self.make(env=leaky)
        b = ex.get_observation()
        b.proprio["bowl_1_pos"] = [0.1, 0.2, 0.3]  # 模拟打包时把名单外字段带进来
        with self.assertRaises(LeakError):
            build_context(b, [], self.spec)
        clean = ex.get_observation()
        build_context(clean, [], self.spec)

    # 10 报告可重算
    def test_10_report_recompute(self):
        env = FakeEnv(success_after_steps=9)
        ex = self.make(env=env)
        model = ScriptedModel([lambda c: move(c, dx=0.2), lambda c: move(c)])
        rt = Runtime(ex, model, self.spec, out_dir=Path(self.tmp.name) / "run")
        stop = rt.run_episode()
        self.assertEqual(stop, "native_success")
        r1 = write_report(self.ledger, Path(self.tmp.name) / "run")
        os.remove(Path(self.tmp.name) / "run" / "report.json")
        r2 = build_report(self.ledger)
        for k in ("control_steps", "model_calls", "rejections", "completed_segments", "native_success"):
            self.assertEqual(r1[k], r2[k])
        self.assertEqual([s["steps_executed"] for s in r1["segments"]], [s["steps_executed"] for s in r2["segments"]])
        self.assertEqual(r2["control_steps"], ex.control_step)
        self.assertEqual(r2["rejections"], 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
