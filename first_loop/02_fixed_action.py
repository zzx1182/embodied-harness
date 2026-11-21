"""「LIBERO 仿真环境与第一条可执行动作」 · 第三、四节：连发一个固定的有界动作，比较请求量和实测位移。

每跑一次记一行账本。默认 x 维填 0.2，其余六维填 0，连发 15 步。
再用 --steps 5 跑一次，两行放在一起就是第四节的对照表。
"""
import argparse
import time
from pathlib import Path

from common import make_env, check_action, eef, Ledger, POS_SCALE_M


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", default="libero_object")
    ap.add_argument("--task", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--init", type=int, default=0)
    ap.add_argument("--dx", type=float, default=0.2, help="x 维的归一化命令，[-1, 1]")
    ap.add_argument("--steps", type=int, default=15)
    ap.add_argument("--out", default="outputs/first_loop")
    args = ap.parse_args()

    ledger = Ledger(Path(args.out) / "ledger.jsonl")
    env, obs, _ = make_env(args.suite, args.task, args.seed, args.init)

    action = [args.dx, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    request_id = ledger.next_request_id()
    before = eef(obs)
    row = {
        "request_id": request_id,
        "ts": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "request": action,
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
        print(f"[{request_id}] 被拒绝，动作没有发出: {e}")
        return

    executed = 0
    for _ in range(args.steps):
        obs, reward, done, info = env.step(action)
        executed += 1
    after = eef(obs)
    row.update({"steps_executed": executed, "eef_after": after, "status": "COMPLETED"})
    ledger.append(row)

    requested_mm = args.dx * POS_SCALE_M * args.steps * 1000
    measured_mm = (after["pos"][0] - before["pos"][0]) * 1000
    print(f"[{request_id}] 动作 {action} 连发 {executed} 步")
    print(f"请求量（名义，x 方向）: {args.dx} × {POS_SCALE_M} m × {args.steps} = {requested_mm:+.2f} mm")
    print(f"实测位移（x 方向）    : {measured_mm:+.2f} mm")
    print(f"差值                  : {requested_mm - measured_mm:+.2f} mm")
    print(f"y、z 方向实测: {(after['pos'][1] - before['pos'][1]) * 1000:+.2f} mm, {(after['pos'][2] - before['pos'][2]) * 1000:+.2f} mm")
    print(f"已追加到 {ledger.path}")
    env.close()


if __name__ == "__main__":
    main()
