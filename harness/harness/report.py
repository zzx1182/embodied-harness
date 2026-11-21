"""从账本重算 episode 报告。报告删掉也能从 ledger.sqlite 重新生成，账本之外没有第二份事实。"""
import json
import re
from pathlib import Path

from .ledger import Ledger


def _object_words(predicate):
    """'in alphabet_soup_1 basket_1_contain_region' -> ['alphabet', 'soup']，取谓词第一个参数的物体名。"""
    parts = predicate.split()
    if len(parts) < 2:
        return []
    return [w for w in re.sub(r"_\d+$", "", parts[1]).split("_") if w]


def audit_declarations(declarations):
    """模型宣布子目标完成时，用当时的仿真真值回头对。宣布了但真值不成立，记一次误验收。"""
    out = []
    for d in declarations:
        truth = json.loads(d["truth_json"] or "[]")
        text = (d["subgoal"] or "").lower().replace("_", " ")
        if d["kind"] == "finish":
            matched = truth
        else:
            matched = [t for t in truth if _object_words(t["predicate"]) and all(w in text for w in _object_words(t["predicate"]))]
        verdict = None if not matched else all(t["value"] for t in matched)
        out.append({"kind": d["kind"], "subgoal": d["subgoal"], "control_step": d["control_step"], "obs_seq": d["obs_seq"],
                    "matched_predicates": [t["predicate"] for t in matched], "true_at_declaration": verdict,
                    "image_path": d["image_path"]})
    return out


def build_report(ledger_path):
    led = Ledger(ledger_path)
    reqs = led.requests()
    calls = led.model_calls()
    pcalls = led.policy_calls()
    executed = [r for r in reqs if r["status"] in ("COMPLETED", "CANCELLED")]
    completed = [r for r in reqs if r["status"] == "COMPLETED"]
    rejected = [r for r in reqs if r["status"] == "REJECTED" and '"MODEL_ERROR"' not in (r["rejection_json"] or "")]
    unknown = [r for r in reqs if r["status"] == "UNKNOWN"]
    ok_calls = [c for c in calls if c["status"] != "error"]
    err_calls = [c for c in calls if c["status"] == "error"]
    spec = led.meta("spec", {}) or {}
    hz = spec.get("control_hz", 20.0)
    control_steps = sum(r["steps_executed"] or 0 for r in executed)
    tokens = {k: sum((c[k] or 0) for c in calls) for k in ("input_tokens", "output_tokens", "cached_input_tokens")}
    segments = []
    for r in reqs:
        payload = json.loads(r["payload_json"] or "{}").get("proposal", {})
        target = (payload.get("target") or {}).get("position_m") if isinstance(payload, dict) else None
        segments.append({"request_id": r["request_id"], "status": r["status"], "tool": payload.get("tool") if isinstance(payload, dict) else None,
                         "instruction": payload.get("instruction") if isinstance(payload, dict) else None,
                         "steps_requested": r["steps_requested"], "steps_executed": r["steps_executed"],
                         "native_success": r["native_success"], "target_pos": target,
                         "rejection": json.loads(r["rejection_json"]) if r["rejection_json"] else None,
                         "eef_before": json.loads(r["eef_before_json"]) if r["eef_before_json"] else None,
                         "eef_after": json.loads(r["eef_after_json"]) if r["eef_after_json"] else None})
    scores = led.scores()
    final_progress = json.loads(scores[-1]["progress_json"]) if scores else None
    declarations = audit_declarations(led.declarations())
    report = {
        "run_id": led.meta("run_id"), "episode_id": led.meta("episode_id"), "env": led.meta("env"),
        "model": led.meta("model_identity"), "stop_reason": led.meta("stop_reason"),
        "native_success": any(bool(r["native_success"]) for r in executed),
        "control_steps": control_steps, "sim_time_s": round(control_steps / hz, 3),
        "model_calls": len(ok_calls), "model_errors": len(err_calls),
        "rejections": len(rejected), "unknown": len(unknown),
        "mode": led.meta("mode", "direct"), "plan": led.meta("plan"), "policy": led.meta("policy_identity"),
        "policy_calls": len(pcalls), "policy_wait_s": round(sum(c["elapsed_s"] or 0 for c in pcalls), 3),
        "completed_segments": len(completed), "cancelled_segments": len(executed) - len(completed),
        "wall_s": led.meta("wall_s"), "model_wait_s": led.meta("model_wait_s"), "infra_wait_s": led.meta("infra_wait_s"),
        "tokens": tokens, "budgets": led.meta("budgets"), "controller_spec_hash": led.meta("controller_spec_hash"),
        "final_progress": final_progress,
        "items_done": None if final_progress is None else sum(bool(t["value"]) for t in final_progress),
        "items_total": None if final_progress is None else len(final_progress),
        "declarations": declarations,
        "false_acceptances": sum(1 for d in declarations if d["true_at_declaration"] is False),
        "segments": segments,
    }
    led.close()
    return report


def report_markdown(r):
    lines = [f"# Episode {r['episode_id']} 报告", "",
             "| 项 | 值 |", "|---|---|",
             f"| 环境 | {json.dumps(r['env'], ensure_ascii=False)} |",
             f"| 模型 | {json.dumps(r['model'], ensure_ascii=False)} |",
             f"| 停止原因 | {r['stop_reason']} |", f"| 原生成功 | {r['native_success']} |",
             f"| 控制步 | {r['control_steps']} |", f"| 仿真时间 s | {r['sim_time_s']} |",
             f"| 模式 | {r['mode']} |", f"| 策略调用数 / 等待 s | {r['policy_calls']} / {r['policy_wait_s']} |",
             f"| 模型调用数 / 接口失败 | {r['model_calls']} / {r['model_errors']} |", f"| 拒绝数 | {r['rejections']} |", f"| UNKNOWN | {r['unknown']} |",
             f"| 目标谓词完成 | {r['items_done']} / {r['items_total']} |", f"| 子目标宣布 / 误验收 | {len(r['declarations'])} / {r['false_acceptances']} |",
             f"| 墙钟 s | {r['wall_s']} |", f"| 模型等待 s | {r['model_wait_s']} |", f"| 等接口恢复 s（不计入墙钟预算） | {r['infra_wait_s']} |",
             f"| token 输入/输出/缓存 | {r['tokens']['input_tokens']} / {r['tokens']['output_tokens']} / {r['tokens']['cached_input_tokens']} |",
             "", "| 段 | 状态 | 请求步 | 执行步 | x 前 | x 后 | 拒绝 |", "|---|---|---|---|---|---|---|"]
    for i, s in enumerate(r["segments"], 1):
        xb = s["eef_before"]["pos"][0] if s["eef_before"] else ""
        xa = s["eef_after"]["pos"][0] if s["eef_after"] else ""
        rej = s["rejection"]["code"] if s["rejection"] else ""
        lines.append(f"| {i} | {s['status']} | {s['steps_requested'] or ''} | {s['steps_executed'] or ''} | {xb} | {xa} | {rej} |")
    return "\n".join(lines) + "\n"


def write_report(ledger_path, out_dir):
    out = Path(out_dir)
    r = build_report(ledger_path)
    (out / "report.json").write_text(json.dumps(r, ensure_ascii=False, indent=2))
    (out / "report.md").write_text(report_markdown(r))
    return r
