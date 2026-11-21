"""「π₀.₅、Direct 与 Hybrid 控制」第二节：写一份策略 manifest，跟着每一次运行走。

  python policy_manifest.py --ckpt /你的目录/checkpoints/pi05_libero --openpi /你的目录/openpi \
      --files /你的目录/openpi/checkpoint_files.json --port 43861 --out outputs/policy_manifest.json

--files 是 fetch_pi05_libero.py 写出的逐文件大小与 CRC32C；不给就只列文件名和大小，并注明没有校验。
--port 给了就顺带向正在运行的策略服务要一次元数据，确认服务加载的配置名。
任何一项变了都算换了策略，新旧 manifest 用 diff 就能看出差在哪。
"""
import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from harness.contracts import Spec
from harness.policy import PREPROCESS_VERSIONS


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def git_head(repo):
    try:
        return subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def lerobot_commit(openpi):
    """openpi 的 uv.lock 里锁着 lerobot 的 git 源，形如 git+https://github.com/huggingface/lerobot?rev=...#<commit>。"""
    lock = Path(openpi) / "uv.lock"
    if not lock.exists():
        return None
    m = re.search(r'name = "lerobot"\nversion = "[^"]*"\nsource = \{ git = "[^"]*#([0-9a-f]{7,40})"', lock.read_text())
    return m.group(1) if m else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="pi05_libero checkpoint 目录")
    ap.add_argument("--openpi", required=True, help="openpi 仓库目录")
    ap.add_argument("--files", help="fetch_pi05_libero.py 写出的 checkpoint_files.json")
    ap.add_argument("--config", default="pi05_libero")
    ap.add_argument("--preprocess", default="libero-v1", choices=list(PREPROCESS_VERSIONS))
    ap.add_argument("--port", type=int, help="策略服务端口，给了就读一次服务元数据")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    ckpt = Path(args.ckpt).resolve()
    if args.files:
        files = json.loads(Path(args.files).read_text())
        checked = f"{sum(1 for f in files if f.get('ok'))}/{len(files)} 个文件大小和 CRC32C 与 GCS 一致"
    else:
        files = [{"file": str(p.relative_to(ckpt)), "bytes": p.stat().st_size}
                 for p in sorted(ckpt.rglob("*")) if p.is_file()]
        checked = "没有校验 CRC32C，先跑 fetch_pi05_libero.py 再用 --files 传进来"
    norm = ckpt / "assets" / "physical-intelligence" / "libero" / "norm_stats.json"
    spec = Spec()
    manifest = {
        "config": args.config,
        "checkpoint": {"dir": str(ckpt), "files": files, "total_bytes": sum(f["bytes"] for f in files), "verified": checked},
        "norm_stats": {"path": str(norm), "sha256": sha256(norm) if norm.exists() else None,
                       "note": None if norm.exists() else "没找到 norm_stats.json，服务会照样起来但候选量程全错"},
        "preprocess": {"version": args.preprocess, "description": PREPROCESS_VERSIONS[args.preprocess]},
        "code": {"openpi_commit": git_head(args.openpi), "lerobot_commit": lerobot_commit(args.openpi)},
        "inputs": {"observation/image": [224, 224, 3], "observation/wrist_image": [224, 224, 3],
                   "observation/state": [8], "prompt": "LIBERO task.language 原文"},
        "controller": {"type": spec.controller, "control_hz": spec.control_hz, "position_scale_m": spec.pos_scale_m,
                       "rotation_scale_rad": spec.rot_scale_rad, "input_range": [-1, 1], "action_dims": 7},
    }
    if args.port:
        from harness.policy import PolicyClient
        client = PolicyClient(port=args.port, client_src=os.environ.get("HARNESS_OPENPI_CLIENT_SRC"))
        manifest["server_metadata"] = client.identity.get("server_metadata")
        client.close()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(manifest, ensure_ascii=False, indent=2))
    print(f"manifest 写在 {out}")
    print(f"  checkpoint: {len(files)} 个文件，{manifest['checkpoint']['total_bytes']:,} 字节，{checked}")
    print(f"  norm_stats: {'sha256 ' + manifest['norm_stats']['sha256'][:16] if manifest['norm_stats']['sha256'] else '没找到'}")
    print(f"  预处理: {args.preprocess}  openpi: {(manifest['code']['openpi_commit'] or '未知')[:7]}  "
          f"lerobot: {(manifest['code']['lerobot_commit'] or '未知')[:7]}")
    if not manifest["code"]["openpi_commit"]:
        print("  openpi 目录不是 git clone 出来的，读不到 commit；在 manifest 的 code.openpi_commit 里手动填上")
    if args.port:
        print(f"  服务元数据: {manifest.get('server_metadata')}")


if __name__ == "__main__":
    main()
