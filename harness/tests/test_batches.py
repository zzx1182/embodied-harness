"""批次和汇总的用例：换了模型不许续跑，重复的 case 不许悄悄覆盖或重复计数，模式不全的 case 不进配对表。
用假的 run_one 和手写的 report.json，不需要 MuJoCo、策略服务或模型接口。

    python -m unittest -v tests/test_batches.py
"""
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import run_matrix
from harness.aggregate import load_runs, paired_cases, summarize_paired
from harness.models import OpenAICompatibleModel, model_identity


def fake_report(task, success=False):
    return {"env": {"task_id": task, "seed": 0, "init_state_index": 0, "instruction": f"task {task}"},
            "native_success": success, "stop_reason": "success" if success else "budget:control_steps",
            "control_steps": 50, "sim_time_s": 2.5, "policy_calls": 0, "model_calls": 3, "model_errors": 0,
            "rejections": 0, "tokens": {}, "wall_s": 1.0, "segments": []}


def write_batch(root, mode, tasks, suite="libero_object", model=None, budget=300):
    root.mkdir(parents=True)
    (root / "declaration.json").write_text(json.dumps({"mode": mode, "suite": suite, "tasks": tasks, "seeds": [0],
                                                       "init_state_index": [0], "budgets": {"max_control_steps": budget}, "denominator": len(tasks)}))
    for task in tasks:
        case = root / f"{suite}_t{task:02d}_s0"
        case.mkdir()
        (case / "report.json").write_text(json.dumps({**fake_report(task), "model": model}))


class Case(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def run_matrix(self, *args):
        argv = ["run_matrix.py", "--mode", "direct", "--tasks", "0-1", "--out", str(self.root / "batch"), *args]
        with mock.patch.object(sys, "argv", argv), mock.patch("sys.stdout"):
            run_matrix.main()

    def test_resume_refuses_another_model(self):
        calls = []

        def interrupted(mode, suite, task, seed, init, spec, case_dir, *rest, **kw):
            calls.append(task)
            if task == 1:
                raise KeyboardInterrupt           # 跑到第二个 case 时中断
            case_dir.mkdir(parents=True)
            (case_dir / "report.json").write_text(json.dumps(fake_report(task)))
            return fake_report(task)

        with mock.patch.dict(os.environ, {"HARNESS_MODEL_NAME": "model-a"}), mock.patch.object(run_matrix, "run_one", interrupted):
            with self.assertRaises(KeyboardInterrupt):
                self.run_matrix("--model", "openai")
            with self.assertRaises(SystemExit) as refused:
                self.run_matrix("--model", "codex")
        self.assertIn("model", str(refused.exception))
        with mock.patch.dict(os.environ, {"HARNESS_MODEL_NAME": "model-b"}), mock.patch.object(run_matrix, "run_one", interrupted):
            with self.assertRaises(SystemExit):
                self.run_matrix("--model", "openai")    # 同一个后端换了模型名也不行
        self.assertEqual(calls, [0, 1])

    def test_model_identity_matches_the_model_object(self):
        with mock.patch.dict(os.environ, {"HARNESS_MODEL_NAME": "m", "HARNESS_MODEL_BASE_URL": "http://x/v1/"}):
            self.assertEqual(model_identity("openai"), OpenAICompatibleModel().identity)

    def test_duplicate_case_across_batches_is_refused(self):
        write_batch(self.root / "first", "direct", [0, 1])
        write_batch(self.root / "rerun", "direct", [1])
        with self.assertRaises(SystemExit) as refused:
            load_runs([self.root])
        self.assertIn("rerun", str(refused.exception))

    def test_same_directory_twice_counts_once(self):
        write_batch(self.root / "only", "direct", [0, 1])
        runs, declarations = load_runs([self.root, self.root / "only"])
        self.assertEqual((len(runs), len(declarations)), (2, 1))

    def test_batch_with_two_models_is_refused(self):
        write_batch(self.root / "mixed", "direct", [0, 1], model={"model": "a"})
        report = self.root / "mixed" / "libero_object_t01_s0" / "report.json"
        report.write_text(json.dumps({**json.loads(report.read_text()), "model": {"model": "b"}}))
        with self.assertRaises(SystemExit):
            load_runs([self.root])

    def test_paired_table_only_keeps_cases_with_every_mode(self):
        # task 0 三种模式都有；task 1 只有 Direct 和 VLA-only；task 2 只有 Hybrid 和 VLA-only
        write_batch(self.root / "direct", "direct", [0, 1])
        write_batch(self.root / "hybrid", "hybrid", [0, 2])
        write_batch(self.root / "vla", "vla", [0, 1, 2])
        runs, _ = load_runs([self.root])
        keep, dropped = paired_cases(runs)
        self.assertEqual(([k[1] for k in keep], dropped), ([0], 2))
        self.assertEqual({row["mode"]: row["started"] for row in summarize_paired(runs)}, {"direct": 1, "hybrid": 1, "vla": 1})


if __name__ == "__main__":
    unittest.main(verbosity=2)
