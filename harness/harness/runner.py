"""跑一条 episode 的公共入口，run_episode.py 和 run_matrix.py 都用它。"""
import json
import time
from pathlib import Path

from .contracts import Spec
from .envs import LiberoEnv
from .executor import Executor
from .models import make_model
from .runtime import Runtime
from .report import write_report


def parse_plan(text):
    """'指令1::块数1||指令2::块数2' -> [(指令1, 块数1), (指令2, 块数2)]"""
    if not text:
        return None
    plan = []
    for part in text.split("||"):
        instruction, _, n = part.rpartition("::")
        plan.append((instruction.strip(), int(n)))
    return plan


def run_one(mode="direct", suite="libero_object", task=0, seed=0, init=0, spec=None, out=None,
            model_kind="openai", policy_host="127.0.0.1", policy_port=43861, gif=False, quiet=False, plan=None):
    """返回 report dict。mode 是 direct / vla / hybrid / orchestrate / scripted。
    scripted 需要 plan：[(子指令, 动作块数), ...]。"""
    spec = spec or Spec()
    out = Path(out or f"outputs/{mode}_{suite}_{task}_{time.strftime('%Y%m%d-%H%M%S')}")
    out.mkdir(parents=True, exist_ok=True)
    env = LiberoEnv(suite, task, seed, init, size=spec.image_size)
    executor = Executor(env, out / "ledger.sqlite", spec)
    model = make_model(model_kind) if mode in ("direct", "hybrid", "orchestrate") else None
    policy = None
    if mode in ("vla", "hybrid", "orchestrate", "scripted"):
        from .policy import PolicyClient
        policy = PolicyClient(policy_host, policy_port, image_size=224)
    rt = Runtime(executor, model, spec, out_dir=out, policy=policy, mode=mode, plan=plan)
    if not quiet:
        print(f"[{mode}] run {executor.run_id} episode {executor.episode_id} | {suite}/{task} seed={seed} init={init}")
        print(f"指令: {env.instruction}")
        if model is not None:
            print(f"模型: {model.identity}")
        if policy is not None:
            print(f"策略: {policy.identity}")
    try:
        stop = rt.run_episode()
    except Exception as e:  # 任何未处理的异常也要留下报告，账本里已有的事实不丢
        executor.ledger.set_meta(stop_reason=f"crash:{type(e).__name__}", model_calls=rt.model_calls, rejections=rt.rejections,
                                 mode=mode, policy_calls=rt.policy_calls)
        write_report(out / "ledger.sqlite", out)
        raise
    finally:
        if policy is not None:
            policy.close()
    report = write_report(out / "ledger.sqlite", out)
    if gif:
        from PIL import Image
        frames = [Image.open(p) for p in sorted((out / "segments").glob("*/agentview_image.png"))]
        if frames:
            frames[0].save(out / "rollout.gif", save_all=True, append_images=frames[1:], duration=400, loop=0)
    if not quiet:
        print(f"停止原因: {stop} | 控制步 {executor.control_step} | 策略调用 {rt.policy_calls} | 模型调用 {rt.model_calls} | 拒绝 {rt.rejections} | 原生成功 {env.check_success()}")
        print(json.dumps({k: report[k] for k in ("stop_reason", "native_success", "control_steps", "sim_time_s", "policy_calls", "model_calls", "model_errors", "rejections", "items_done", "items_total", "false_acceptances", "wall_s", "model_wait_s", "tokens")}, ensure_ascii=False))
        print(f"报告: {out}/report.md")
    executor.close()
    return report
