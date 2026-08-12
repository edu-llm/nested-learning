#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


CORE8_VARIANTS = [
    "base_eval",
    "static_adapter",
    "single_memory",
    "cms",
    "cms_no_dgd",
    "cms_no_momentum",
    "cms_no_retention",
    "cms_no_self_reference",
]

PAPER12_VARIANTS = [
    *CORE8_VARIANTS,
    "cms_two_levels",
    "cms_four_levels",
    "cms_bptt_updates",
    "cms_m3lite_optimizer",
]


def read_json(path: Path) -> dict[str, Any]:
    with path.open() as handle:
        return json.load(handle)


def deep_update(base: dict[str, Any], patch: dict[str, Any]) -> dict[str, Any]:
    out = json.loads(json.dumps(base))
    for key, value in patch.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_update(out[key], value)
        else:
            out[key] = value
    return out


def templates() -> dict[str, dict[str, Any]]:
    cms = read_json(Path("configs/olmoe_cms_cluster.json"))
    single = read_json(Path("configs/olmoe_single_memory_cluster.json"))
    static = read_json(Path("configs/olmoe_static_adapter_cluster.json"))
    base_eval = read_json(Path("configs/olmoe_base_eval.json"))
    return {
        "base_eval": base_eval,
        "static_adapter": static,
        "single_memory": single,
        "cms": cms,
        "cms_no_dgd": deep_update(cms, {"adapter": {"memory_update_rule": "hebbian"}}),
        "cms_no_momentum": deep_update(cms, {"adapter": {"update_momentum": 0.0}}),
        "cms_no_retention": deep_update(cms, {"adapter": {"retention_factor": 1.0}}),
        "cms_no_self_reference": deep_update(cms, {"adapter": {"self_reference_scale": 0.0}}),
        "cms_two_levels": deep_update(
            cms,
            {
                "adapter": {
                    "chunk_sizes": [128, 2048],
                    "learning_rates": [0.30, 0.035],
                }
            },
        ),
        "cms_four_levels": deep_update(
            cms,
            {
                "adapter": {
                    "chunk_sizes": [64, 256, 1024, 2048],
                    "learning_rates": [0.35, 0.18, 0.08, 0.035],
                }
            },
        ),
        "cms_bptt_updates": deep_update(cms, {"adapter": {"detach_memory_updates": False}}),
        "cms_m3lite_optimizer": deep_update(
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
    }


def apply_12h_bounds(
    cfg: dict[str, Any],
    *,
    variant: str,
    seed: int,
    seed_id: int,
    max_steps: int,
    eval_every: int,
    eval_size: int,
    max_length: int,
    gradient_accumulation_steps: int,
    rank: int,
) -> dict[str, Any]:
    cfg = json.loads(json.dumps(cfg))
    cfg["run"]["seed"] = seed
    cfg["run"]["log_every"] = 10
    cfg["run"]["output_dir"] = f"results/block12_{variant}_seed{seed}"
    cfg["data"]["seed"] = 1000 + 100000 * seed_id
    cfg["data"]["eval_seed"] = 900000 + 100000 * seed_id
    cfg["data"]["eval_size"] = eval_size
    cfg["data"]["max_length"] = max_length
    cfg["data"]["train_size"] = max(4096, max_steps * 32)
    cfg["adapter"]["rank"] = rank
    cfg["train"]["ddp_find_unused_parameters"] = True
    cfg["train"]["gradient_checkpointing_use_reentrant"] = False

    if cfg["train"]["max_steps"] == 0:
        cfg["run"]["eval_every"] = 1
        cfg["data"]["train_size"] = 1
        cfg["train"]["gradient_accumulation_steps"] = 1
        cfg["train"]["warmup_steps"] = 0
        cfg["train"]["num_workers"] = 0
    else:
        cfg["run"]["eval_every"] = eval_every
        cfg["train"]["max_steps"] = max_steps
        cfg["train"]["gradient_accumulation_steps"] = gradient_accumulation_steps
        cfg["train"]["warmup_steps"] = max(10, max_steps // 20)
        cfg["train"]["num_workers"] = 4

    return cfg


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=Path("configs/generated_block_12h"))
    parser.add_argument("--profile", choices=["core8", "paper12"], default="core8")
    parser.add_argument("--seeds", nargs="+", type=int, default=[17])
    parser.add_argument("--max-steps", type=int, default=800)
    parser.add_argument("--eval-every", type=int, default=200)
    parser.add_argument("--eval-size", type=int, default=128)
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--rank", type=int, default=256)
    args = parser.parse_args()

    selected = CORE8_VARIANTS if args.profile == "core8" else PAPER12_VARIANTS
    all_templates = templates()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    rows: list[tuple[int, int, str, str, Path]] = []
    written: list[Path] = []
    for seed_id, seed in enumerate(args.seeds):
        for variant in selected:
            cfg = apply_12h_bounds(
                all_templates[variant],
                variant=variant,
                seed=seed,
                seed_id=seed_id,
                max_steps=args.max_steps,
                eval_every=args.eval_every,
                eval_size=args.eval_size,
                max_length=args.max_length,
                gradient_accumulation_steps=args.gradient_accumulation_steps,
                rank=args.rank,
            )
            out = args.out_dir / f"{variant}_seed{seed}.json"
            out.write_text(json.dumps(cfg, indent=2) + "\n")
            written.append(out)
            index = len(written) - 1
            node = index % 8 + 1
            wave = index // 8 + 1
            run_name = f"nl12_{variant}_s{seed}"
            rows.append((wave, node, run_name, str(out), out))

    manifest = args.out_dir / "manifest.txt"
    manifest.write_text("\n".join(str(path) for path in written) + "\n")

    dispatch_plan = args.out_dir / "dispatch_plan.tsv"
    with dispatch_plan.open("w") as handle:
        handle.write("wave\tnode\trun_name\tconfig\n")
        for wave, node, run_name, config, _ in rows:
            handle.write(f"{wave}\t{node}\t{run_name}\t{config}\n")

    print(f"wrote {len(written)} configs")
    print(f"manifest {manifest}")
    print(f"dispatch plan {dispatch_plan}")


if __name__ == "__main__":
    main()
