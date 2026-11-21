"""「π₀.₅、Direct 与 Hybrid 控制」第二节：在同一份冻结观测上检查策略服务。

  python probe_frozen.py --suite libero_object --task 0 --port 43861 --fault fault-no-rotate --out outputs/frozen_probe

reset 一次、拿一份观测后不再步进，然后：
  1. 用正常预处理（libero-v1）连发两次，两次都应返回 (10, 7)、全部有限，前 6 维（位姿增量）都在 [-1, 1] 里；
     第 7 维夹爪常见 -1.004 这类略超出一点的值，执行器会裁到边界并在账本里记 CLIPPED_ROWS。
     两次之间有差别是 π₀.₅ 采样的随机性，量级记下来。
  2. 换成故意做错的预处理（--fault）再发一次。服务端照样返回 (10, 7)，什么都不报，
     只有记录里的预处理版本不同。fault-no-rotate 的候选和正常那两次明显对不上；
     fault-state-order 对 pi05_libero 是空实验，这份权重不读 state，差异只是又一次采样。
结果写进 --out 目录的 frozen_probe.json，候选存成 .npy。
"""
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from harness.contracts import PROPRIO_KEYS
from harness.policy import PREPROCESS_VERSIONS, PolicyClient


def check(candidate):
    pose, grip = candidate[:, :6], candidate[:, 6]
    return {"shape": list(candidate.shape), "finite": bool(np.isfinite(candidate).all()),
            "pose_in_range": bool(np.all(np.abs(pose) <= 1.0)),
            "gripper_overshoot": round(max(0.0, float(np.abs(grip).max()) - 1.0), 4),
            "min": round(float(candidate.min()), 4), "max": round(float(candidate.max()), 4)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", default="libero_object")
    ap.add_argument("--task", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--init", type=int, default=0)
    ap.add_argument("--port", type=int, default=43861)
    ap.add_argument("--fault", default="fault-no-rotate", choices=[k for k in PREPROCESS_VERSIONS if k.startswith("fault")])
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    from harness.envs import LiberoEnv
    env = LiberoEnv(args.suite, args.task, args.seed, args.init)
    obs = env.observe()
    images = env.images()
    proprio = {k: np.asarray(obs[k]) for k in PROPRIO_KEYS}
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    records = []
    runs = [("libero-v1", "normal-1"), ("libero-v1", "normal-2"), (args.fault, "fault")]
    clients = {}
    for version, name in runs:
        client = clients.get(version) or clients.setdefault(
            version, PolicyClient(port=args.port, client_src=os.environ.get("HARNESS_OPENPI_CLIENT_SRC"), preprocess=version))
        candidate, elapsed = client.infer(images, proprio, env.instruction)
        np.save(out / f"{name}.npy", candidate)
        records.append({"name": name, "preprocess_version": version, "elapsed_ms": round(elapsed * 1000, 1), **check(candidate)})
    for c in clients.values():
        c.close()
    env.close()

    a, b, f = (np.load(out / f"{n}.npy") for n in ("normal-1", "normal-2", "fault"))
    summary = {
        "env": env.meta, "instruction": env.instruction,
        "records": records,
        "normal_vs_normal_max_abs_diff": round(float(np.abs(a - b).max()), 4),
        "fault_vs_normal_mean_abs_diff": round(float(np.abs(f - (a + b) / 2).mean()), 4),
        "normal_vs_normal_mean_abs_diff": round(float(np.abs(a - b).mean()), 4),
    }
    (out / "frozen_probe.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    for r in records:
        print(f"{r['name']:<9} 预处理 {r['preprocess_version']:<18} 形状 {tuple(r['shape'])} 有限 {r['finite']} "
              f"位姿 6 维在 [-1, 1] 内 {r['pose_in_range']} 夹爪超出 {r['gripper_overshoot']} 耗时 {r['elapsed_ms']} ms")
    print(f"两次正常请求之间的平均差 {summary['normal_vs_normal_mean_abs_diff']}，故障请求和正常的平均差 {summary['fault_vs_normal_mean_abs_diff']}")
    print(f"初态哈希 {env.meta['init_state_sha256'][:16]}…，记录写在 {out / 'frozen_probe.json'}")


if __name__ == "__main__":
    main()
