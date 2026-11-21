"""「LIBERO 仿真环境与第一条可执行动作」 · 第二节：重置一个 LIBERO 任务并读取观测。

跑完会在输出目录留下：
  agentview.png        桌面固定机位画面
  wrist.png            手腕相机画面
  observation.json     七项白名单观测（图像只记形状）
  dropped_keys.json    被过滤掉的键名
"""
import argparse
import json
from pathlib import Path

from common import make_env, split_obs, eef, save_image, jsonable


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", default="libero_object")
    ap.add_argument("--task", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--init", type=int, default=0, help="固定初态编号")
    ap.add_argument("--out", default="outputs/first_loop")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    env, obs, instruction = make_env(args.suite, args.task, args.seed, args.init)
    kept, dropped = split_obs(obs)

    print(f"任务: {args.suite} / {args.task}")
    print(f"指令: {instruction}")
    print(f"初态: seed={args.seed} init_state_index={args.init}")
    print(f"观测字典一共 {len(obs)} 个键，白名单保留 {len(kept)} 项，过滤掉 {len(dropped)} 项")
    print("--- 白名单七项 ---")
    for k, v in kept.items():
        print(f"{k:26s} shape={tuple(v.shape)}")
    print("--- 过滤掉的键（前 12 个）---")
    for k in dropped[:12]:
        print(f"  {k}")
    if len(dropped) > 12:
        print(f"  ... 还有 {len(dropped) - 12} 个")
    print(f"末端位置 (m): {eef(obs)['pos']}")

    save_image(kept["agentview_image"], out / "agentview.png")
    save_image(kept["robot0_eye_in_hand_image"], out / "wrist.png")
    record = {
        "suite": args.suite,
        "task": args.task,
        "instruction": instruction,
        "seed": args.seed,
        "init_state_index": args.init,
        "control_step": 0,
        "whitelist": {
            k: (list(v.shape) if v.ndim == 3 else jsonable(v)) for k, v in kept.items()
        },
    }
    (out / "observation.json").write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
    (out / "dropped_keys.json").write_text(json.dumps(dropped, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"已写出 {out}/agentview.png, wrist.png, observation.json, dropped_keys.json")
    env.close()


if __name__ == "__main__":
    main()
