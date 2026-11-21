"""候选核对的用例：π₀.₅ 的候选必须和出它的观测、执行器代际、策略身份绑定，执行前逐项核对。
三条必拒的反例（旧帧候选、前缀越界、纠偏后接旧尾部）都应留下 REJECTED 记录，控制步不增加。
跑在假环境和假策略上，不需要 MuJoCo，也不需要策略服务。

    python -m unittest -v tests/test_candidates.py
"""
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from harness.contracts import Spec, ActionRequest, DecisionProposal
from harness.envs import FakeEnv
from harness.executor import Executor
from harness.runtime import build_context, parse_decision

POLICY = {"kind": "fake", "checkpoint": "pi05_libero", "preprocess": "rotate180+pad224"}


def candidate(rows=10):
    c = np.zeros((rows, 7))
    c[:, 0] = 0.5
    return c


class Case(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.ex = Executor(FakeEnv(), Path(self.tmp.name) / "ledger.sqlite", Spec(max_control_steps=50))
        self.ex.set_policy(POLICY)
        self.n = 0

    def tearDown(self):
        self.tmp.cleanup()

    def accept(self, rows, binding, horizon):
        """按当前快照构造一条 accept_vla 请求并提交，返回回执。"""
        b = self.ex.get_observation()
        p = DecisionProposal(tool="accept_vla", expected_obs_seq=b.obs_seq, horizon_steps=horizon,
                             candidate=np.asarray(rows).tolist(), candidate_binding=binding, raw={"tool": "accept_vla"})
        self.n += 1
        req = ActionRequest(f"req-{self.n}", b.episode_id, b.executor_generation, b.obs_seq,
                            {"tool": "accept_vla", "horizon_steps": horizon, "candidate": p.candidate,
                             "candidate_binding": binding})
        req.compute_hash()
        return req.request_id, self.ex.submit(req, p)

    def assert_rejected(self, rows, binding, horizon, code):
        before = self.ex.control_step
        rid, r = self.accept(rows, binding, horizon)
        self.assertEqual(r.status, "REJECTED")
        self.assertEqual(r.rejection["code"], code)
        self.assertEqual(self.ex.ledger.get(rid)["status"], "REJECTED")
        self.assertEqual(self.ex.control_step, before)

    def test_accepts_a_fresh_candidate(self):
        c = candidate()
        _, r = self.accept(c, self.ex.register_candidate(c), 4)
        self.assertEqual(r.status, "COMPLETED")
        self.assertEqual(self.ex.control_step, 4)

    def test_stale_frame_candidate(self):
        c = candidate()
        binding = self.ex.register_candidate(c)
        self.ex.advance_for_test()                     # 快照从 n 变成 n+1，候选描述的是过去的状态
        self.assert_rejected(c, binding, 4, "STALE_CANDIDATE")

    def test_prefix_out_of_range(self):
        c = candidate()
        self.assert_rejected(c, self.ex.register_candidate(c), 11, "BAD_HORIZON")      # k 超过 10
        self.ex.advance_for_test(45)                                                  # 剩余预算只有 5 步
        c = candidate()
        self.assert_rejected(c, self.ex.register_candidate(c), 8, "BAD_HORIZON")

    def test_old_tail_after_correction(self):
        c = candidate()
        binding = self.ex.register_candidate(c)
        b = self.ex.get_observation()
        ctx = build_context(b, [], self.ex.spec)
        pos = list(ctx["proprio"]["robot0_eef_pos"]); pos[0] += 0.02
        fix, err = parse_decision({"tool": "move_eef", "expected_obs_seq": ctx["obs_seq"], "horizon_steps": 3, "reason": "纠偏",
                                   "target": {"frame_id": "robot0_base", "position_m": pos,
                                              "quaternion_xyzw": list(ctx["proprio"]["robot0_eef_quat"]), "gripper": "open"}})
        self.assertIsNone(err)
        req = ActionRequest("fix", b.episode_id, b.executor_generation, b.obs_seq, fix.raw); req.compute_hash()
        self.assertEqual(self.ex.submit(req, fix).status, "COMPLETED")
        self.assert_rejected(c[3:], binding, 7, "CANDIDATE_MISMATCH")   # 拿旧候选剩下的 7 步接着跑
        self.assert_rejected(c, binding, 7, "STALE_CANDIDATE")          # 整块重交也不行，机械臂已经不在原处

    def test_policy_changed(self):
        c = candidate()
        binding = self.ex.register_candidate(c)
        self.ex.set_policy({**POLICY, "checkpoint": "another"})
        self.assert_rejected(c, binding, 4, "POLICY_MISMATCH")

    def test_unregistered_candidate(self):
        self.assert_rejected(candidate(), {"candidate_id": "made-up", "obs_seq": 1}, 4, "UNKNOWN_CANDIDATE")


if __name__ == "__main__":
    unittest.main(verbosity=2)
