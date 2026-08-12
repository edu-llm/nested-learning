#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


def read_last_metrics(path: Path) -> dict | None:
    metrics_path = path / "metrics.jsonl"
    if not metrics_path.exists():
        return None
    last = None
    with metrics_path.open() as handle:
        for line in handle:
            line = line.strip()
            if line:
                last = json.loads(line)
    return last


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("result_dirs", nargs="*", type=Path)
    parser.add_argument("--manifest", type=Path, default=None)
    args = parser.parse_args()

    result_dirs = list(args.result_dirs)
    if args.manifest is not None:
        with args.manifest.open() as handle:
            for line in handle:
                config_path = line.strip()
                if not config_path:
                    continue
                with open(config_path) as config_handle:
                    cfg = json.load(config_handle)
                result_dirs.append(Path(cfg["run"]["output_dir"]))
    if not result_dirs:
        parser.error("provide result dirs or --manifest")

    rows = []
    for result_dir in result_dirs:
        metrics = read_last_metrics(result_dir)
        if metrics is None:
            rows.append((result_dir.name, None))
        else:
            rows.append((result_dir.name, metrics))

    print(
        f"{'run':<34} {'step':>8} {'facts':>8} {'needle':>8} "
        f"{'lang':>8} {'guard_ppl':>10}"
    )
    print("-" * 86)
    for name, metrics in rows:
        if metrics is None:
            print(f"{name:<34} {'missing':>8}")
            continue
        print(
            f"{name:<34} "
            f"{int(metrics.get('step', 0)):>8} "
            f"{100 * float(metrics.get('facts_eval_accuracy', 0.0)):>7.1f}% "
            f"{100 * float(metrics.get('needle_eval_accuracy', 0.0)):>7.1f}% "
            f"{100 * float(metrics.get('language_eval_accuracy', 0.0)):>7.1f}% "
            f"{float(metrics.get('guardrail_ppl', 0.0)):>10.2f}"
        )


if __name__ == "__main__":
    main()
