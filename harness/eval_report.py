"""「具身 Agent 评测、扰动与任务编排」：把若干批 run_matrix 的输出汇成一份报告。

  python eval_report.py outputs/matrix_vla outputs/matrix_vla_pro outputs/matrix_pair outputs/orchestration --out outputs/eval_report

输出 report.md（评测声明、汇总、逐 case 配对、限制与边界）、summary.json 和 runs.json。
"""
import argparse

from harness.aggregate import write


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("roots", nargs="+", help="run_matrix 的输出目录，会递归找 declaration.json")
    ap.add_argument("--out", required=True)
    ap.add_argument("--title", default="具身 Agent 评测报告")
    args = ap.parse_args()
    runs = write(args.roots, args.out, args.title)
    print(f"{len(runs)} 条 episode，报告写在 {args.out}/report.md")


if __name__ == "__main__":
    main()
