#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


TEMPLATES = [
    Path("configs/olmoe_static_adapter_cluster.json"),
    Path("configs/olmoe_single_memory_cluster.json"),
    Path("configs/olmoe_cms_cluster.json"),
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=Path("configs/generated_olmoe_ablation"))
    parser.add_argument("--seeds", nargs="+", type=int, default=[17, 23, 42])
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    written = []
    for template in TEMPLATES:
        with template.open() as handle:
            cfg = json.load(handle)
        stem = template.stem.replace("_cluster", "")
        for seed_id, seed in enumerate(args.seeds):
            run_cfg = json.loads(json.dumps(cfg))
            run_cfg["run"]["seed"] = seed
            run_cfg["data"]["seed"] = 1000 + 100000 * seed_id
            run_cfg["data"]["eval_seed"] = 900000 + 100000 * seed_id
            run_cfg["run"]["output_dir"] = f"results/{stem}_seed{seed}"
            out = args.out_dir / f"{stem}_seed{seed}.json"
            out.write_text(json.dumps(run_cfg, indent=2) + "\n")
            written.append(out)

    manifest = args.out_dir / "manifest.txt"
    manifest.write_text("\n".join(str(path) for path in written) + "\n")
    print(f"wrote {len(written)} configs")
    print(f"manifest {manifest}")


if __name__ == "__main__":
    main()
