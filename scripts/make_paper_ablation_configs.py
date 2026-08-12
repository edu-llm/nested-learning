#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def deep_update(base: dict[str, Any], patch: dict[str, Any]) -> dict[str, Any]:
    out = json.loads(json.dumps(base))
    for key, value in patch.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_update(out[key], value)
        else:
            out[key] = value
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=Path("configs/generated_paper_ablation"))
    parser.add_argument("--seeds", nargs="+", type=int, default=[17, 23, 42])
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    cms = json.load(open("configs/olmoe_cms_cluster.json"))
    single = json.load(open("configs/olmoe_single_memory_cluster.json"))
    static = json.load(open("configs/olmoe_static_adapter_cluster.json"))
    base_eval = json.load(open("configs/olmoe_base_eval.json"))

    variants: list[tuple[str, dict[str, Any]]] = [
        ("base_eval", base_eval),
        ("static_adapter", static),
        ("single_memory", single),
        ("cms", cms),
        (
            "cms_no_dgd",
            deep_update(cms, {"adapter": {"memory_update_rule": "hebbian"}}),
        ),
        (
            "cms_no_momentum",
            deep_update(cms, {"adapter": {"update_momentum": 0.0}}),
        ),
        (
            "cms_no_retention",
            deep_update(cms, {"adapter": {"retention_factor": 1.0}}),
        ),
        (
            "cms_no_self_reference",
            deep_update(cms, {"adapter": {"self_reference_scale": 0.0}}),
        ),
        (
            "cms_two_levels",
            deep_update(
                cms,
                {
                    "adapter": {
                        "chunk_sizes": [128, 2048],
                        "learning_rates": [0.30, 0.035],
                    }
                },
            ),
        ),
        (
            "cms_four_levels",
            deep_update(
                cms,
                {
                    "adapter": {
                        "chunk_sizes": [64, 256, 1024, 4096],
                        "learning_rates": [0.35, 0.18, 0.08, 0.025],
                    }
                },
            ),
        ),
        (
            "cms_bptt_updates",
            deep_update(cms, {"adapter": {"detach_memory_updates": False}}),
        ),
        (
            "cms_m3lite_optimizer",
            deep_update(
                cms,
                {
                    "train": {
                        "optimizer": "m3lite",
                        "slow_beta": 0.99,
                        "slow_interval": 16,
                        "slow_weight": 0.25,
                    }
                },
            ),
        ),
    ]

    written = []
    for variant_name, template in variants:
        for seed_id, seed in enumerate(args.seeds):
            cfg = json.loads(json.dumps(template))
            cfg["run"]["seed"] = seed
            cfg["data"]["seed"] = 1000 + 100000 * seed_id
            cfg["data"]["eval_seed"] = 900000 + 100000 * seed_id
            cfg["run"]["output_dir"] = f"results/paper_{variant_name}_seed{seed}"
            out = args.out_dir / f"{variant_name}_seed{seed}.json"
            out.write_text(json.dumps(cfg, indent=2) + "\n")
            written.append(out)

    manifest = args.out_dir / "manifest.txt"
    manifest.write_text("\n".join(str(path) for path in written) + "\n")
    print(f"wrote {len(written)} configs")
    print(f"manifest {manifest}")


if __name__ == "__main__":
    main()
