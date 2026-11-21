"""按任务、seed 批量跑同一模式，逐 case 记录，分母写死在命令行里。

  python run_matrix.py --mode vla --suite libero_object --tasks 0-9 --seeds 0 --out outputs/matrix_vla_object

已经有 report.json 的 case 会跳过，中断后重跑同一条命令即可续上。续跑时模式、任务、seed、预算、
模型身份（--model 和对应的环境变量）、策略 manifest 有任何一项和第一次不同，都会拒绝，免得一批结果里混着两套配置。
结果在 <out>/matrix.json 和 <out>/matrix.md。
"""
import argparse
import hashlib
import json
import time
from pathlib import Path

from harness.contracts import Spec
from harness.models import model_identity
from harness.runner import run_one, parse_plan

MODEL_MODES = ("direct", "hybrid", "orchestrate")


def parse_range(text):
    out = []
    for part in text.split(","):
        part = part.strip()
        if "-" in part:
            a, b = part.split("-")
            out.extend(range(int(a), int(b) + 1))
        elif part:
            out.append(int(part))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["direct", "vla", "hybrid", "orchestrate", "scripted"], required=True)
    ap.add_argument("--model", choices=["openai", "codex"], default="openai")
    ap.add_argument("--policy-host", default="127.0.0.1")
    ap.add_argument("--policy-port", type=int, default=43861)
    ap.add_argument("--suite", default="libero_object")
    ap.add_argument("--tasks", default="0-9")
    ap.add_argument("--seeds", default="0")
    ap.add_argument("--init", default="0", help="初态编号，可以是 0 或 0-2")
    ap.add_argument("--max-control-steps", type=int, default=300)
    ap.add_argument("--max-calls", type=int, default=40, help="大模型调用上限（VLA-only 不受它约束）")
    ap.add_argument("--max-policy-calls", type=int, default=None, help="策略调用上限，默认不设")
    ap.add_argument("--max-wall-s", type=float, default=1800)
    ap.add_argument("--plan", default=None, help="scripted 模式的子指令顺序：\"指令1::块数1||指令2::块数2\"")
    ap.add_argument("--policy-manifest", default=None, help="policy_manifest.py 写出的 manifest，给了就把它的哈希冻结进声明，换了权重不能续跑")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    spec = Spec(max_control_steps=args.max_control_steps, max_model_calls=args.max_calls, max_wall_s=args.max_wall_s,
                max_policy_calls=args.max_policy_calls)
    inits = parse_range(str(args.init))
    cases = [(t, s, i) for t in parse_range(args.tasks) for s in parse_range(args.seeds) for i in inits]
    declaration = {"mode": args.mode, "suite": args.suite, "tasks": parse_range(args.tasks), "seeds": parse_range(args.seeds),
                   "init_state_index": inits, "plan": args.plan, "budgets": {"max_control_steps": spec.max_control_steps,
                   "max_calls": spec.max_model_calls, "max_policy_calls": spec.max_policy_calls, "max_wall_s": spec.max_wall_s},
                   "spec_hash": spec.hash(), "denominator": len(cases),
                   "model": model_identity(args.model) if args.mode in MODEL_MODES else None,
                   "policy_manifest_sha256": hashlib.sha256(Path(args.policy_manifest).read_bytes()).hexdigest() if args.policy_manifest else None,
                   "started": time.strftime("%Y-%m-%dT%H:%M:%S%z")}
    decl_path = out / "declaration.json"
    if decl_path.exists():
        # 续跑：声明在第一次运行前就冻结了，开始时间保留原值，只追加续跑时间；配置变了就不许续跑。
        # 旧版本写的声明没有 model、policy_manifest_sha256 两项，这两项只在旧声明里有时才比。
        old = json.loads(decl_path.read_text())
        frozen = ("mode", "suite", "tasks", "seeds", "init_state_index", "plan", "budgets", "spec_hash", "denominator",
                  "model", "policy_manifest_sha256")
        changed = [k for k in frozen if (k in old or k not in ("model", "policy_manifest_sha256")) and old.get(k) != declaration[k]]
        if changed:
            raise SystemExit(f"{decl_path} 已存在，这几项和第一次不同：{', '.join(changed)}。换一个 --out 重新开始，别在旧声明下续跑。")
        declaration["started"] = old["started"]
        declaration["resumed"] = old.get("resumed", []) + [time.strftime("%Y-%m-%dT%H:%M:%S%z")]
    decl_path.write_text(json.dumps(declaration, ensure_ascii=False, indent=2))
    rows = []
    for task, seed, init in cases:
        case_dir = out / (f"{args.suite}_t{task:02d}_s{seed}" + (f"_i{init}" if len(inits) > 1 or init else ""))
        if (case_dir / "report.json").exists():
            r = json.loads((case_dir / "report.json").read_text())
            print(f"skip {case_dir.name}: 已有报告 success={r['native_success']}")
        else:
            print(f"=== {case_dir.name} ===", flush=True)
            r = run_one(args.mode, args.suite, task, seed, init, spec, case_dir, args.model,
                        args.policy_host, args.policy_port, gif=False, quiet=True, plan=parse_plan(args.plan))
            print(f"  {r['stop_reason']} success={r['native_success']} steps={r['control_steps']} policy={r['policy_calls']} model={r['model_calls']} rej={r['rejections']} wall={r['wall_s']}s", flush=True)
        rows.append({"task": task, "seed": seed, "init": init, "instruction": (r.get("env") or {}).get("instruction"),
                     "stop_reason": r["stop_reason"], "success": r["native_success"], "control_steps": r["control_steps"],
                     "sim_time_s": r["sim_time_s"], "policy_calls": r["policy_calls"], "model_calls": r["model_calls"],
                     "rejections": r["rejections"], "model_errors": r.get("model_errors"), "items_done": r.get("items_done"),
                     "items_total": r.get("items_total"), "false_acceptances": r.get("false_acceptances"),
                     "wall_s": r["wall_s"], "tokens": r["tokens"], "dir": case_dir.name})
        write_matrix(out, declaration, rows)
    print(f"完成 {len(rows)}/{len(cases)}，成功 {sum(bool(x['success']) for x in rows)}，见 {out}/matrix.md")


def write_matrix(out, declaration, rows):
    n = len(rows)
    succ = sum(bool(r["success"]) for r in rows)
    summary = {"declaration": declaration, "finished": n, "successes": succ,
               "success_rate": round(succ / n, 4) if n else None, "rows": rows,
               "stop_reasons": {k: sum(r["stop_reason"] == k for r in rows) for k in sorted({r["stop_reason"] for r in rows})}}
    (out / "matrix.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    lines = [f"# {declaration['mode']} · {declaration['suite']} · 分母 {declaration['denominator']}", "",
             f"完成 {n}，成功 {succ}，成功率 {summary['success_rate']}。停止原因：{summary['stop_reasons']}", "",
             "| task | seed | init | 成功 | 停止原因 | 控制步 | 策略调用 | 模型调用 | 拒绝 | 完成谓词 | 墙钟 s |", "|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        done = "" if r.get("items_done") is None else f"{r['items_done']}/{r['items_total']}"
        lines.append(f"| {r['task']} | {r['seed']} | {r.get('init', 0)} | {'✓' if r['success'] else '✗'} | {r['stop_reason']} | {r['control_steps']} | {r['policy_calls']} | {r['model_calls']} | {r['rejections']} | {done} | {r['wall_s']} |")
    (out / "matrix.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
