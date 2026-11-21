"""「具身 Agent 评测、扰动与任务编排」：把多批 run_matrix 的结果汇成一份评测报告。

每一格都来自某个 case 目录里的 report.json，report.json 又能从 ledger.sqlite 重算，所以这里不存第二份事实。

失败主因按下面的顺序判，命中第一条就停，一个失败 episode 只有一个主因：
  基础设施  停止原因是 crash、infra、UNKNOWN，或者连续拒绝全是接口失败（旧账本里的 MODEL_ERROR）
  校验拒绝  停止原因是连续拒绝，且拒绝码是 LIMIT、NON_FINITE、BAD_* 这类校验失败
  决策      最后 8 段执行里末端净位移不到 1.5 cm，在原地徘徊
  执行未达  move_eef 段目标和实测末端的偏差，最近 8 段的中位数超过 2 cm
  超时      其余预算耗尽的情况
感知类失败要用物体真值判断目标点对不对，账本的评分表只记目标谓词真假、不记物体坐标，这里不自动判，出现时人工看图补标。
"""
import json
import statistics
from pathlib import Path

BASE_SUITES = ("libero_spatial", "libero_object", "libero_goal", "libero_10")
CAUSES = ("感知", "决策", "校验拒绝", "执行未达", "超时", "基础设施")


def perturbation_of(suite):
    if suite.endswith("_swap"):
        return "Position", suite[: -len("_swap")]
    if suite.endswith("_task"):
        return "Task", suite[: -len("_task")]
    return "无", suite


def _dist(a, b):
    return sum((x - y) ** 2 for x, y in zip(a, b)) ** 0.5


def classify(report):
    """返回失败主因，成功返回 None。"""
    if report.get("native_success"):
        return None
    stop = report.get("stop_reason") or ""
    if stop.startswith("crash") or stop.startswith("infra") or stop == "UNKNOWN":
        return "基础设施"
    segs = report.get("segments") or []
    if stop == "budget:consecutive_rejections":
        codes = [s["rejection"]["code"] for s in segs if s["status"] == "REJECTED" and s.get("rejection")]
        tail = codes[-3:]
        if tail and all(c == "MODEL_ERROR" for c in tail):
            return "基础设施"
        return "校验拒绝"
    executed = [s for s in segs if s["status"] in ("COMPLETED", "CANCELLED") and s.get("eef_before") and s.get("eef_after")]
    window = executed[-8:]
    if len(window) >= 5 and _dist(window[-1]["eef_after"]["pos"], window[0]["eef_before"]["pos"]) < 0.015:
        return "决策"
    moves = [s for s in executed if s.get("tool") == "move_eef" and s.get("target_pos")]
    if len(moves) >= 3:
        errs = [_dist(s["target_pos"], s["eef_after"]["pos"]) for s in moves[-8:]]
        if statistics.median(errs) > 0.02:
            return "执行未达"
    return "超时"


def load_runs(roots):
    """读出所有批次。同一个目录给了两遍只算一次；两个批次里有同一个 case 的同一种模式、或者一个批次里混了两种模型，
    都直接报错停下，由人决定留哪一批，不在汇总里悄悄覆盖或重复计数。"""
    runs, declarations, seen = [], [], set()
    for root in roots:
        for decl_path in sorted(Path(root).rglob("declaration.json")):
            if decl_path.resolve() in seen:
                continue
            seen.add(decl_path.resolve())
            decl = json.loads(decl_path.read_text())
            declarations.append({**decl, "dir": str(decl_path.parent)})
            perturb, base = perturbation_of(decl["suite"])
            for case_dir in sorted(p for p in decl_path.parent.iterdir() if p.is_dir()):
                rp = case_dir / "report.json"
                if not rp.exists():
                    continue
                r = json.loads(rp.read_text())
                env = r.get("env") or {}
                runs.append({
                    "suite": decl["suite"], "base_suite": base, "perturbation": perturb, "mode": decl["mode"],
                    "budget_steps": (decl.get("budgets") or {}).get("max_control_steps"),
                    "task": env.get("task_id"), "seed": env.get("seed"), "init": env.get("init_state_index"),
                    "instruction": env.get("instruction"), "success": bool(r.get("native_success")),
                    "stop_reason": r.get("stop_reason"), "control_steps": r.get("control_steps") or 0,
                    "model_calls": r.get("model_calls") or 0, "model_errors": r.get("model_errors") or 0,
                    "policy_calls": r.get("policy_calls") or 0, "rejections": r.get("rejections") or 0,
                    "tokens": r.get("tokens") or {}, "wall_s": r.get("wall_s"), "model_wait_s": r.get("model_wait_s"),
                    "policy_wait_s": r.get("policy_wait_s"), "sim_time_s": r.get("sim_time_s"),
                    "items_done": r.get("items_done"), "items_total": r.get("items_total"),
                    "declarations": len(r.get("declarations") or []), "false_acceptances": r.get("false_acceptances") or 0,
                    "failure_cause": classify(r), "dir": str(case_dir), "model": r.get("model"),
                    "budgets": decl.get("budgets") or {},
                })
    _check_unique(runs)
    return runs, declarations


def _check_unique(runs):
    first, batch_models = {}, {}
    for r in runs:
        key = (r["suite"], r["task"], r["seed"], r["init"], r["mode"], r["budget_steps"])
        if key in first:
            raise SystemExit(f"同一个 case、同一种模式出现了两次：\n  {first[key]}\n  {r['dir']}\n"
                             "补跑或重跑的结果和原批次重复了。只把要用的那一批目录传给 eval_report.py，另一批挪走或单独出报告。")
        first[key] = r["dir"]
        if r["model"]:
            batch_models.setdefault(str(Path(r["dir"]).parent), set()).add(json.dumps(r["model"], sort_keys=True))
    mixed = [d for d, models in batch_models.items() if len(models) > 1]
    if mixed:
        raise SystemExit(f"这些批次里的 case 用了不同的模型，不能当成一批比较：{', '.join(mixed)}")


MODE_NAMES = {"direct": "Direct", "vla": "VLA-only", "hybrid": "Hybrid", "orchestrate": "编排", "scripted": "写死顺序"}
MODE_ORDER = ["direct", "vla", "hybrid", "orchestrate", "scripted"]


def mode_name(m):
    return MODE_NAMES.get(m, m)


def _group_row(rs):
    """一组 episode 的计数：成功/开始执行、控制步和调用的合计与中位、接口失败、token、部分完成、失败主因。"""
    steps = [r["control_steps"] for r in rs]
    calls = [r["model_calls"] for r in rs]
    causes = {}
    for r in rs:
        if r["failure_cause"]:
            causes[r["failure_cause"]] = causes.get(r["failure_cause"], 0) + 1
    tok = {k: sum((r["tokens"] or {}).get(k) or 0 for r in rs) for k in ("input_tokens", "output_tokens", "cached_input_tokens")}
    items = [r for r in rs if r["items_total"]]
    return {"started": len(rs), "successes": sum(r["success"] for r in rs),
            "suites": sorted({r["suite"] for r in rs}), "cases": len({(r["suite"], r["task"], r["seed"], r["init"]) for r in rs}),
            "control_steps_total": sum(steps), "control_steps_median": statistics.median(steps),
            "model_calls_total": sum(calls), "model_calls_median": statistics.median(calls),
            "model_errors_total": sum(r["model_errors"] for r in rs), "tokens": tok,
            "items_done_total": sum(r["items_done"] or 0 for r in items) if items else None,
            "items_total": sum(r["items_total"] or 0 for r in items) if items else None,
            "false_acceptances": sum(r["false_acceptances"] for r in rs), "failure_causes": causes}


def _group(runs, key):
    groups = {}
    for r in runs:
        groups.setdefault(key(r), []).append(r)
    order = lambda kv: (str(kv[0][0]), MODE_ORDER.index(kv[0][1]) if kv[0][1] in MODE_ORDER else 9, str(kv[0][2]))
    return [(k, _group_row(rs)) for k, rs in sorted(groups.items(), key=order)]


def summarize_total(runs):
    """跨套件合计：按 扰动 × 模式 × 控制步预算 分组，四个套件合并成一行。"""
    return [{"perturbation": p, "mode": m, "budget_steps": b, **row}
            for (p, m, b), row in _group(runs, lambda r: (r["perturbation"], r["mode"], r["budget_steps"]))]


def paired_cases(runs):
    """同一扰动、同一控制步上限下，先找出跑过两种以上模式的 case，再只留几种模式全都跑过的那些，
    这样表里每种模式面对的是同一批 case、分母相同。模式不全的 case（比如只补跑了其中两种）不进表，返回它们的个数。"""
    groups = {}
    for key, by_mode in paired(runs).items():
        if len(by_mode) >= 2:
            groups.setdefault((perturbation_of(key[0])[0], key[4]), []).append((key, set(by_mode)))
    keep, dropped = set(), 0
    for cases in groups.values():
        full = set().union(*(modes for _, modes in cases))
        for key, modes in cases:
            if modes == full:
                keep.add(key)
            else:
                dropped += 1
    return keep, dropped


def summarize_paired(runs):
    """同 case 配对子集：只留几种模式全都跑过的 case（套件、任务、seed、初态、控制步上限），再跨套件合计。"""
    keep, _ = paired_cases(runs)
    sub = [r for r in runs if (r["suite"], r["task"], r["seed"], r["init"], r["budget_steps"]) in keep]
    return [{"perturbation": p, "mode": m, "budget_steps": b, **row}
            for (p, m, b), row in _group(sub, lambda r: (r["perturbation"], r["mode"], r["budget_steps"]))]


def summarize(runs):
    """按 套件 × 扰动 × 模式 × 控制步预算 分组。预算不同的两组不合并。"""
    groups = {}
    for r in runs:
        groups.setdefault((r["base_suite"], r["perturbation"], r["mode"], r["budget_steps"]), []).append(r)
    rows = []
    for (base, pert, mode, budget), rs in sorted(groups.items(), key=lambda kv: tuple(str(x) for x in kv[0])):
        steps = [r["control_steps"] for r in rs]
        calls = [r["model_calls"] for r in rs]
        causes = {}
        for r in rs:
            if r["failure_cause"]:
                causes[r["failure_cause"]] = causes.get(r["failure_cause"], 0) + 1
        tok = {k: sum((r["tokens"] or {}).get(k) or 0 for r in rs) for k in ("input_tokens", "output_tokens", "cached_input_tokens")}
        items = [r for r in rs if r["items_total"]]
        rows.append({"base_suite": base, "perturbation": pert, "mode": mode, "budget_steps": budget, "started": len(rs),
                     "successes": sum(r["success"] for r in rs),
                     "control_steps_total": sum(steps), "control_steps_median": statistics.median(steps),
                     "model_calls_total": sum(calls), "model_calls_median": statistics.median(calls),
                     "model_errors_total": sum(r["model_errors"] for r in rs), "tokens": tok,
                     "items_done": [f"{r['items_done']}/{r['items_total']}" for r in items],
                     "false_acceptances": sum(r["false_acceptances"] for r in rs), "failure_causes": causes})
    return rows


def paired(runs):
    """同一个 case（套件、任务、seed、初态）且控制步预算相同，才放进同一行比较。"""
    by_case = {}
    for r in runs:
        by_case.setdefault((r["suite"], r["task"], r["seed"], r["init"], r["budget_steps"]), {})[r["mode"]] = r
    return by_case


def _mixed_budget_note(runs):
    """表按控制步上限分组；同一模式、同一控制步上限下如果混了模型调用或墙钟上限不同的批次，在表头写明，不悄悄合并。"""
    budgets = {}
    for r in runs:
        budgets.setdefault((r["perturbation"], mode_name(r["mode"]), r["budget_steps"]), set()).add(json.dumps(r["budgets"], sort_keys=True))
    mixed = [f"{p} {m} {b} 步" for (p, m, b), bs in sorted(budgets.items(), key=str) if len(bs) > 1]
    return f"注意：{'、'.join(mixed)} 这几行合并了其他预算不同的批次，比较前先看评测声明。" if mixed else ""


def render(runs, declarations, title="具身 Agent 评测报告"):
    modes = sorted({r["mode"] for r in runs}, key=lambda m: MODE_ORDER.index(m) if m in MODE_ORDER else 9)
    models = sorted({json.dumps(r["model"], ensure_ascii=False, sort_keys=True) for r in runs if r["model"]})
    lines = [f"# {title}", "", "## 评测声明", "",
             "- 接口层级：Direct 给末端目标（move_eef），VLA-only 执行 π₀.₅ 原生动作块前缀，Hybrid 审核候选后二选一，"
             "编排模式只下子指令、动作全由 π₀.₅ 产生，写死顺序不看观测。",
             "- 感知条件：模型只看两路 256×256 图像和五项本体量，物体真值不进上下文，只写进评分表。",
             f"- 模型：{'; '.join(models) if models else '无'}",
             "", "| 批次目录 | 模式 | 套件 | 任务 | seed | 初态 | 预算 | 分母 |", "|---|---|---|---|---|---|---|---|"]
    for d in declarations:
        lines.append(f"| {Path(d['dir']).name} | {mode_name(d['mode'])} | {d['suite']} | {d['tasks']} | {d['seeds']} | {d.get('init_state_index')} | "
                     f"{json.dumps(d['budgets'], ensure_ascii=False)} | {d['denominator']} |")
    lines += ["", "## 汇总", "",
              "| 套件 | 扰动 | 模式 | 控制步上限 | 成功/开始执行 | 控制步 合计/中位 | 模型调用 合计/中位 | 接口失败 | token 输入/输出/缓存 | 失败主因 |",
              "|---|---|---|---|---|---|---|---|---|---|"]
    for s in summarize(runs):
        t = s["tokens"]
        causes = "，".join(f"{k} {v}" for k, v in sorted(s["failure_causes"].items())) or "无"
        lines.append(f"| {s['base_suite']} | {s['perturbation']} | {mode_name(s['mode'])} | {s['budget_steps']} | {s['successes']}/{s['started']} | "
                     f"{s['control_steps_total']}/{s['control_steps_median']} | {s['model_calls_total']}/{s['model_calls_median']} | "
                     f"{s['model_errors_total']} | {t['input_tokens']}/{t['output_tokens']}/{t['cached_input_tokens']} | {causes} |")
    _, dropped = paired_cases(runs)
    paired_note = ("只统计同一个 case（套件、任务、seed、初态、控制步上限）上几种模式全都跑过的 episode，"
                   "同一扰动、同一控制步上限下每种模式面对的是同一批 case，可以直接比较。各模式的模型调用和墙钟上限见上面的评测声明。"
                   + (f"另有 {dropped} 个 case 只跑了其中几种模式，没放进这张表。" if dropped else ""))
    for heading, note, rows in (
            ("跨套件合计", "同一扰动、同一模式、同一控制步上限的 episode 跨套件合并成一行。" + _mixed_budget_note(runs), summarize_total(runs)),
            ("同 case 配对子集", paired_note, summarize_paired(runs))):
        lines += ["", f"## {heading}", "", note, "",
                  "| 扰动 | 模式 | 控制步上限 | 套件数 / case 数 | 成功/开始执行 | 部分完成 | 控制步 合计/中位 | 模型调用 合计/中位 | 接口失败 | token 输入/输出/缓存 | 失败主因 |",
                  "|---|---|---|---|---|---|---|---|---|---|---|"]
        for s in rows:
            t = s["tokens"]
            causes = "，".join(f"{k} {v}" for k, v in sorted(s["failure_causes"].items())) or "无"
            part = f"{s['items_done_total']}/{s['items_total']}" if s["items_total"] else "不适用"
            lines.append(f"| {s['perturbation']} | {mode_name(s['mode'])} | {s['budget_steps']} | {len(s['suites'])} / {s['cases']} | "
                         f"{s['successes']}/{s['started']} | {part} | {s['control_steps_total']}/{s['control_steps_median']} | "
                         f"{s['model_calls_total']}/{s['model_calls_median']} | {s['model_errors_total']} | "
                         f"{t['input_tokens']}/{t['output_tokens']}/{t['cached_input_tokens']} | {causes} |")
    lines += ["", "## 逐 case 配对", "", "| 套件 | 任务 | seed | 初态 | 控制步上限 | " + " | ".join(mode_name(m) for m in modes) + " |",
              "|---|---|---|---|---|" + "---|" * len(modes)]
    for (suite, task, seed, init, budget), by_mode in sorted(paired(runs).items(), key=lambda kv: tuple(str(x) for x in kv[0])):
        cells = []
        for m in modes:
            r = by_mode.get(m)
            if r is None:
                cells.append("没跑")
            else:
                mark = "✓" if r["success"] else f"✗ {r['failure_cause']}"
                extra = f"，{r['items_done']}/{r['items_total']}" if r["items_total"] and r["items_total"] > 1 else ""
                cells.append(f"{mark}（{r['control_steps']} 步{extra}）")
        lines.append(f"| {suite} | {task} | {seed} | {init} | {budget} | " + " | ".join(cells) + " |")
    multi = [r for r in runs if (r["items_total"] or 0) > 1]
    if multi:
        lines += ["", "## 多目标任务的部分完成与宣布", "",
                  "部分完成数来自评分表里目标谓词的最终真值。宣布次数是模型调用 declare_subgoal_done 和 finish 的次数，"
                  "误验收是宣布时对应物体的谓词为假的次数。", "",
                  "| 套件 | 任务 | 初态 | 模式 | 成功 | 完成目标 | 宣布 | 误验收 | 控制步 | 模型调用 |", "|---|---|---|---|---|---|---|---|---|---|"]
        for r in sorted(multi, key=lambda r: (r["suite"], str(r["task"]), str(r["init"]), modes.index(r["mode"]))):
            lines.append(f"| {r['suite']} | {r['task']} | {r['init']} | {mode_name(r['mode'])} | {'✓' if r['success'] else '✗'} | "
                         f"{r['items_done']}/{r['items_total']} | {r['declarations']} | {r['false_acceptances']} | "
                         f"{r['control_steps']} | {r['model_calls']} |")
    n_infra = sum(1 for r in runs if r["failure_cause"] == "基础设施")
    lines += ["", "## 限制与边界", "",
              f"- 一共 {len(runs)} 条 episode，每个 case 只跑了一次，大模型输出不可逐位复现，π₀.₅ 的动作块是采样出来的。",
              f"- 基础设施失败 {n_infra} 条，计入开始执行数，不算任务失败。",
              "- 感知类失败没有自动判定，需要看图人工补标。",
              "- 参照数字不在本表里，和论文作者报告值对比时任务数、seed 数、预算都不同。"]
    return "\n".join(lines) + "\n"


def write(roots, out_dir, title="具身 Agent 评测报告"):
    runs, declarations = load_runs(roots)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "runs.json").write_text(json.dumps(runs, ensure_ascii=False, indent=2))
    (out / "summary.json").write_text(json.dumps(summarize(runs), ensure_ascii=False, indent=2))
    (out / "summary_total.json").write_text(json.dumps(summarize_total(runs), ensure_ascii=False, indent=2))
    (out / "summary_paired.json").write_text(json.dumps(summarize_paired(runs), ensure_ascii=False, indent=2))
    (out / "report.md").write_text(render(runs, declarations, title))
    return runs
