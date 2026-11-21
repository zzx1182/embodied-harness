"""「LIBERO 仿真环境与第一条可执行动作」 · 第五节：验证超量程和含 NaN 的请求会被拦截。

两个坏动作都走 check_action，被拒绝的请求照样记进账本，steps_executed 是 0，
eef_after 和 eef_before 一样，因为动作压根没发出去。

加 --demo-clipping 会把超量程的 2.0 直接发给环境 5 步，让你看到 robosuite 悄悄裁到 1.0 之后机械臂照样动了，
而记录里写的却是 2.0。这一段只是演示，不记账本。
"""
import argparse
import math
import time
from pathlib import Path

from common import make_env, check_action, eef, Ledger


BAD_ACTIONS = {
    "超量程": [2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    "含 NaN": [float("nan"), 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", default="libero_object")
    ap.add_argument("--task", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--init", type=int, default=0)
    ap.add_argument("--steps", type=int, default=5)
    ap.add_argument("--out", default="outputs/first_loop")
    ap.add_argument("--demo-clipping", action="store_true")
    args = ap.parse_args()

    ledger = Ledger(Path(args.out) / "ledger.jsonl")
    env, obs, _ = make_env(args.suite, args.task, args.seed, args.init)

    for name, action in BAD_ACTIONS.items():
        request_id = ledger.next_request_id()
        before = eef(obs)
        row = {
            "request_id": request_id,
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "request": [None if (isinstance(v, float) and math.isnan(v)) else v for v in action],
            "steps_requested": args.steps,
            "steps_executed": 0,
            "eef_before": before,
            "eef_after": before,
            "status": "PENDING",
        }
        try:
            check_action(action)
        except AssertionError as e:
            row["status"] = f"REJECTED: {e}"
            ledger.append(row)
            print(f"[{request_id}] {name} 被拦下: {e}")
            print(f"           末端位置不变: {before['pos']}")
            continue
        raise SystemExit(f"{name} 没有被拦下，检查 check_action")

    if args.demo_clipping:
        print("--- 演示：把 2.0 直接发给环境，不经过检查 ---")
        before = eef(obs)
        for _ in range(args.steps):
            obs, *_ = env.step(BAD_ACTIONS["超量程"])
        after = eef(obs)
        print(f"发了 {args.steps} 步 2.0，x 方向实测位移 {(after['pos'][0] - before['pos'][0]) * 1000:+.2f} mm")
        print("控制器把 2.0 裁成了 1.0 再执行，而且不会告诉你。这就是要在发送前自己拦的原因。")

    print(f"账本现在有 {len(ledger.rows())} 行: {ledger.path}")
    env.close()


if __name__ == "__main__":
    main()
