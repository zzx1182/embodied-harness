"""「LIBERO 仿真环境与第一条可执行动作」 · 第六节：把账本打印成对照表。

每一行都能对回 ledger.jsonl 里的一条记录，request_id 就是那一行的编号。
"""
import argparse
from pathlib import Path

from common import Ledger, POS_SCALE_M


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="outputs/first_loop")
    args = ap.parse_args()

    rows = Ledger(Path(args.out) / "ledger.jsonl").rows()
    if not rows:
        raise SystemExit("账本是空的，先跑 02_fixed_action.py 和 03_bad_actions.py")

    print(f"{'request_id':10s} {'状态':12s} {'请求 x':>8s} {'请求步数':>8s} {'执行步数':>8s} {'名义请求 mm':>11s} {'实测 dx mm':>10s} {'差值 mm':>8s}")
    completed = rejected = 0
    for r in rows:
        x = r["request"][0]
        status = r["status"].split(":")[0]
        completed += status == "COMPLETED"
        rejected += status == "REJECTED"
        if status == "COMPLETED":
            nominal = x * POS_SCALE_M * r["steps_requested"] * 1000
            dx = (r["eef_after"]["pos"][0] - r["eef_before"]["pos"][0]) * 1000
            print(f"{r['request_id']:10s} {status:12s} {x:8.2f} {r['steps_requested']:8d} {r['steps_executed']:8d} {nominal:11.2f} {dx:10.2f} {nominal - dx:8.2f}")
        else:
            print(f"{r['request_id']:10s} {status:12s} {str(x):>8s} {r['steps_requested']:8d} {r['steps_executed']:8d} {'':>11s} {'':>10s} {'':>8s}   {r['status'].split(':', 1)[1].strip()}")
    print(f"共 {len(rows)} 行，COMPLETED {completed}，REJECTED {rejected}")


if __name__ == "__main__":
    main()
