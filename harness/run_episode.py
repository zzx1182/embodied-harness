"""跑一条 episode。三种模式共用同一个执行器和账本。

Direct（「模型建议、执行校验与可靠回执」）：
  export HARNESS_MODEL_BASE_URL=https://你的接口/v1  HARNESS_MODEL_NAME=模型名  HARNESS_MODEL_API_KEY=密钥
  python run_episode.py --mode direct --model openai --suite libero_object --task 0 --max-calls 30 --out outputs/direct_object0

VLA-only（「π₀.₅、Direct 与 Hybrid 控制」，需要先起 π₀.₅ 策略服务）：
  python run_episode.py --mode vla --policy-port 43861 --suite libero_object --task 0 --out outputs/vla_object0

Hybrid（「π₀.₅、Direct 与 Hybrid 控制」，两者都要）：
  python run_episode.py --mode hybrid --model openai --policy-port 43861 --suite libero_object --task 0 --out outputs/hybrid_object0

编排与写死顺序（「具身 Agent 评测、扰动与任务编排」，LIBERO-10 字母汤加番茄酱）：
  python run_episode.py --mode orchestrate --model openai --policy-port 43861 --suite libero_10 --task 0 --max-control-steps 400 --out outputs/orch_l10_0
  python run_episode.py --mode scripted --policy-port 43861 --suite libero_10 --task 0 --max-control-steps 400 \\
      --plan "pick up the alphabet soup and place it in the basket::20||pick up the tomato sauce and place it in the basket::20" --out outputs/scripted_l10_0
"""
import argparse

from harness.contracts import Spec
from harness.runner import run_one, parse_plan


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["direct", "vla", "hybrid", "orchestrate", "scripted"], default="direct")
    ap.add_argument("--model", choices=["openai", "codex"], default="openai")
    ap.add_argument("--policy-host", default="127.0.0.1")
    ap.add_argument("--policy-port", type=int, default=43861)
    ap.add_argument("--suite", default="libero_object")
    ap.add_argument("--task", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--init", type=int, default=0)
    ap.add_argument("--max-control-steps", type=int, default=300)
    ap.add_argument("--max-calls", type=int, default=40, help="大模型调用上限（VLA-only 不调大模型，不受它约束）")
    ap.add_argument("--max-policy-calls", type=int, default=None, help="策略调用上限，默认不设，VLA-only 只受控制步预算约束")
    ap.add_argument("--max-wall-s", type=float, default=1800)
    ap.add_argument("--plan", default=None, help="scripted 模式的子指令顺序：\"指令1::块数1||指令2::块数2\"")
    ap.add_argument("--out", default=None)
    ap.add_argument("--gif", action="store_true", help="把每段的前视画面拼成 rollout.gif")
    args = ap.parse_args()
    spec = Spec(max_control_steps=args.max_control_steps, max_model_calls=args.max_calls, max_wall_s=args.max_wall_s,
                max_policy_calls=args.max_policy_calls)
    run_one(args.mode, args.suite, args.task, args.seed, args.init, spec, args.out, args.model,
            args.policy_host, args.policy_port, args.gif, plan=parse_plan(args.plan))


if __name__ == "__main__":
    main()
