"""接口出错时的处理和预算口径的用例：配置错误立即停机、限流才重试、逐段进度、VLA-only 只受控制步预算约束。
跑在假环境上，不需要 MuJoCo，也不需要真实模型接口（401 和 429 由本机起的一个小 HTTP 服务返回）。

    python -m unittest -v tests/test_infra.py
"""
import contextlib
import io
import json
import os
import shutil
import sys
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from harness.contracts import Spec
from harness.envs import FakeEnv
from harness.executor import Executor
from harness.models import CodexCLIModel, OpenAICompatibleModel, ScriptedModel
from harness.runtime import Runtime


class FakePolicy:
    identity = {"kind": "fake"}

    def infer(self, images, proprio, instruction):
        c = np.zeros((10, 7))
        c[:, 0] = 0.1
        return c, 0.001


class ReplyModel(ScriptedModel):
    """每次都返回同一个接口结果，用来模拟配置错误或接口繁忙。"""

    def __init__(self, reply):
        super().__init__([])
        self.reply, self.calls = reply, 0

    def decide(self, context, images, out_dir=None):
        self.calls += 1
        return dict(self.reply), None, 0.01


def status_server(code):
    """本机起一个只会返回固定状态码的 HTTP 服务，返回 (base_url, server)。"""
    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            self.rfile.read(int(self.headers.get("Content-Length", 0)))
            body = json.dumps({"error": {"message": f"status {code}"}}).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):
            pass
    server = HTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return f"http://127.0.0.1:{server.server_address[1]}/v1", server


class Case(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.ledger = Path(self.tmp.name) / "ledger.sqlite"

    def tearDown(self):
        self.tmp.cleanup()

    def run_quiet(self, rt):
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            stop = rt.run_episode()
        return stop, buf.getvalue()

    def test_config_error_stops_at_once(self):
        spec = Spec(max_model_calls=5, model_error_backoff_s=30.0)   # 真要退避就会卡住测试
        ex = Executor(FakeEnv(), self.ledger, spec)
        model = ReplyModel({"_fatal": "HTTP 401: invalid api key"})
        stop, out = self.run_quiet(Runtime(ex, model, spec))
        self.assertEqual(stop, "infra:model_config")
        self.assertEqual(model.calls, 1)
        self.assertEqual(ex.control_step, 0)
        self.assertIn("接口配置错误，停机：HTTP 401", out)

    def test_busy_interface_backs_off_and_says_so(self):
        spec = Spec(max_model_calls=5, model_error_backoff_s=0.0, max_model_errors=3)
        ex = Executor(FakeEnv(), self.ledger, spec)
        model = ReplyModel({"_error": "HTTP 429: rate limited"})
        stop, out = self.run_quiet(Runtime(ex, model, spec))
        self.assertEqual(stop, "infra:model_unavailable")
        self.assertEqual(model.calls, 3)
        self.assertIn("接口失败（连续第 1 次", out)

    def test_progress_line_per_segment(self):
        spec = Spec(max_model_calls=2, max_control_steps=50)
        ex = Executor(FakeEnv(), self.ledger, spec)

        def move(c):
            pos = list(c["proprio"]["robot0_eef_pos"]); pos[0] += 0.02
            return {"tool": "move_eef", "expected_obs_seq": c["obs_seq"], "horizon_steps": 3, "reason": "t",
                    "target": {"frame_id": "robot0_base", "position_m": pos,
                               "quaternion_xyzw": list(c["proprio"]["robot0_eef_quat"]), "gripper": "open"}}
        stop, out = self.run_quiet(Runtime(ex, ScriptedModel([move, move]), spec))
        self.assertEqual(stop, "budget:model_calls")
        self.assertIn("段 000 | move_eef | COMPLETED 3 步 | 控制步 3/50 | 模型调用 1/2", out)
        self.assertIn("段 001 | move_eef | COMPLETED 3 步 | 控制步 6/50 | 模型调用 2/2", out)

    def test_vla_only_is_bounded_by_control_steps(self):
        spec = Spec(max_model_calls=2, max_control_steps=50)       # 大模型调用上限不该截断 VLA-only
        ex = Executor(FakeEnv(success_after_steps=10_000), self.ledger, spec)
        rt = Runtime(ex, None, spec, policy=FakePolicy(), mode="vla")
        stop, _ = self.run_quiet(rt)
        self.assertEqual(stop, "budget:control_steps")
        self.assertEqual(ex.control_step, 50)
        self.assertEqual(rt.policy_calls, 5)

    def test_policy_call_cap_when_asked(self):
        spec = Spec(max_control_steps=50, max_policy_calls=2)
        ex = Executor(FakeEnv(success_after_steps=10_000), self.ledger, spec)
        rt = Runtime(ex, None, spec, policy=FakePolicy(), mode="vla")
        stop, _ = self.run_quiet(rt)
        self.assertEqual(stop, "budget:policy_calls")
        self.assertEqual(ex.control_step, 20)

    def test_openai_client_tells_config_errors_from_busy(self):
        for code, key in ((401, "_fatal"), (404, "_fatal"), (429, "_error"), (503, "_error")):
            url, server = status_server(code)
            try:
                model = OpenAICompatibleModel(base_url=url, model="m", api_key="k", timeout_s=5)
                decision, usage, _ = model.decide({"obs_seq": 1}, {"agentview_image": np.zeros((8, 8, 3), np.uint8)})
            finally:
                server.shutdown()
                server.server_close()
            self.assertIn(key, decision, code)
            self.assertIn(f"HTTP {code}", decision[key])

    def test_codex_scratch_is_per_process(self):
        os.environ.pop("HARNESS_CODEX_SCRATCH", None)
        a, b = CodexCLIModel(), CodexCLIModel()
        self.assertNotEqual(a.scratch, b.scratch)
        self.assertTrue(Path(a.scratch).is_dir())
        for m in (a, b):
            shutil.rmtree(m.scratch, ignore_errors=True)


if __name__ == "__main__":
    unittest.main(verbosity=2)
