#!/usr/bin/env python3
"""
Dependency-free proof of concept for nested multi-timescale memory.

The task is intentionally synthetic:
- A stream teaches key -> class "facts".
- Half the facts are volatile and change often.
- Half the facts are stable and mostly repeat.
- A single memory has to choose one plasticity setting.
- A CMS-style model keeps fast/medium/slow memories and routes queries by a
  learned-in-real-life but explicit-in-this-toy volatility feature.

This is not evidence that a Qwen3-scale architecture works. It is a smoke test
for the core mechanism: multiple update timescales can dominate any one
timescale when the stream mixes volatile and stable knowledge.
"""

from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, pstdev
from typing import Callable


Vector = list[float]


def dot(a: Vector, b: Vector) -> float:
    return sum(x * y for x, y in zip(a, b))


def add_scaled(row: Vector, key: Vector, scale: float) -> None:
    for i, value in enumerate(key):
        row[i] += scale * value


class LinearDeltaMemory:
    """Tiny online associative memory trained by one-step L2/delta updates."""

    def __init__(self, num_classes: int, dim: int, learning_rate: float) -> None:
        self.num_classes = num_classes
        self.dim = dim
        self.learning_rate = learning_rate
        self.weights = [[0.0] * dim for _ in range(num_classes)]

    def update(self, key: Vector, target_class: int) -> None:
        logits = self.logits(key)
        for class_id, row in enumerate(self.weights):
            target = 1.0 if class_id == target_class else 0.0
            add_scaled(row, key, self.learning_rate * (target - logits[class_id]))

    def logits(self, key: Vector) -> Vector:
        return [dot(row, key) for row in self.weights]

    def predict(self, key: Vector) -> int:
        logits = self.logits(key)
        return max(range(self.num_classes), key=lambda i: logits[i])


class StaticInitialMemory(LinearDeltaMemory):
    """Static adapter control: learns the initial corpus, then freezes."""

    def __init__(self, num_classes: int, dim: int, learning_rate: float, train_events: int) -> None:
        super().__init__(num_classes, dim, learning_rate)
        self.train_events = train_events
        self._seen_events = 0

    def update(self, key: Vector, target_class: int) -> None:
        if self._seen_events < self.train_events:
            super().update(key, target_class)
        self._seen_events += 1


class GatedCmsMemory:
    """Three memories with different plasticities and a simple routing gate."""

    def __init__(self, num_classes: int, dim: int) -> None:
        self.num_classes = num_classes
        self.fast = LinearDeltaMemory(num_classes, dim, learning_rate=0.30)
        self.medium = LinearDeltaMemory(num_classes, dim, learning_rate=0.12)
        self.slow = LinearDeltaMemory(num_classes, dim, learning_rate=0.035)

    def update(self, key: Vector, target_class: int) -> None:
        self.fast.update(key, target_class)
        self.medium.update(key, target_class)
        self.slow.update(key, target_class)

    def logits(self, key: Vector) -> Vector:
        # The first feature marks volatile facts. In a real model this would be
        # a learned gate from the residual stream, not a hand-coded bit.
        volatile = key[0] > 0.0
        if volatile:
            weights = (0.15, 0.80, 0.05)
        else:
            weights = (0.05, 0.15, 0.80)

        fast = self.fast.logits(key)
        medium = self.medium.logits(key)
        slow = self.slow.logits(key)
        return [
            weights[0] * fast[i] + weights[1] * medium[i] + weights[2] * slow[i]
            for i in range(self.num_classes)
        ]

    def predict(self, key: Vector) -> int:
        logits = self.logits(key)
        return max(range(self.num_classes), key=lambda i: logits[i])


@dataclass
class Stream:
    keys: list[Vector]
    final_values: list[int]
    volatile: list[bool]
    events: list[tuple[int, int]]
    initial_event_count: int


def make_stream(
    seed: int,
    num_facts: int,
    dim: int,
    num_classes: int,
    stream_steps: int,
    initial_repetitions: int,
) -> Stream:
    rng = random.Random(seed)
    keys: list[Vector] = []
    volatile: list[bool] = []

    for fact_id in range(num_facts):
        is_volatile = fact_id < num_facts // 2
        volatile.append(is_volatile)
        key = [(1.0 if rng.random() < 0.5 else -1.0) / math.sqrt(dim) for _ in range(dim)]
        key[0] = (1.0 if is_volatile else -1.0) / math.sqrt(dim)
        keys.append(key)

    values = [rng.randrange(num_classes) for _ in range(num_facts)]
    events: list[tuple[int, int]] = []

    for _ in range(initial_repetitions):
        order = list(range(num_facts))
        rng.shuffle(order)
        for fact_id in order:
            events.append((fact_id, values[fact_id]))

    initial_event_count = len(events)

    for _ in range(stream_steps):
        if rng.random() < 0.65:
            fact_id = rng.randrange(0, num_facts // 2)
            if rng.random() < 0.70:
                old_value = values[fact_id]
                new_value = rng.randrange(num_classes - 1)
                if new_value >= old_value:
                    new_value += 1
                values[fact_id] = new_value
            events.append((fact_id, values[fact_id]))
        else:
            fact_id = rng.randrange(num_facts // 2, num_facts)
            events.append((fact_id, values[fact_id]))

    return Stream(keys, values, volatile, events, initial_event_count)


def evaluate(model: object, stream: Stream) -> dict[str, float]:
    for fact_id, value in stream.events:
        model.update(stream.keys[fact_id], value)  # type: ignore[attr-defined]

    buckets = {"all": [], "volatile": [], "stable": []}
    for fact_id, key in enumerate(stream.keys):
        prediction = model.predict(key)  # type: ignore[attr-defined]
        correct = prediction == stream.final_values[fact_id]
        buckets["all"].append(correct)
        buckets["volatile" if stream.volatile[fact_id] else "stable"].append(correct)

    return {name: sum(values) / len(values) for name, values in buckets.items()}


def summarize(rows: list[dict[str, object]]) -> dict[str, dict[str, float]]:
    model_names = sorted({str(row["model"]) for row in rows})
    summary: dict[str, dict[str, float]] = {}
    for model_name in model_names:
        subset = [row for row in rows if row["model"] == model_name]
        summary[model_name] = {}
        for metric in ("all", "volatile", "stable"):
            values = [float(row[metric]) for row in subset]
            summary[model_name][f"{metric}_mean"] = mean(values)
            summary[model_name][f"{metric}_std"] = pstdev(values)
    return summary


def format_pct(value: float) -> str:
    return f"{100.0 * value:5.1f}%"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--num-facts", type=int, default=320)
    parser.add_argument("--dim", type=int, default=64)
    parser.add_argument("--num-classes", type=int, default=32)
    parser.add_argument("--stream-steps", type=int, default=2500)
    parser.add_argument("--initial-repetitions", type=int, default=10)
    parser.add_argument("--out", type=Path, default=Path("poc_results.json"))
    args = parser.parse_args()

    rows: list[dict[str, object]] = []

    for seed in range(args.seeds):
        stream = make_stream(
            seed=seed,
            num_facts=args.num_facts,
            dim=args.dim,
            num_classes=args.num_classes,
            stream_steps=args.stream_steps,
            initial_repetitions=args.initial_repetitions,
        )

        model_factories: list[tuple[str, Callable[[], object]]] = [
            (
                "static_initial",
                lambda s=stream: StaticInitialMemory(
                    args.num_classes,
                    args.dim,
                    learning_rate=0.12,
                    train_events=s.initial_event_count,
                ),
            ),
            ("single_fast", lambda: LinearDeltaMemory(args.num_classes, args.dim, learning_rate=0.30)),
            ("single_medium", lambda: LinearDeltaMemory(args.num_classes, args.dim, learning_rate=0.12)),
            ("single_slow", lambda: LinearDeltaMemory(args.num_classes, args.dim, learning_rate=0.035)),
            ("cms_gated", lambda: GatedCmsMemory(args.num_classes, args.dim)),
        ]

        for model_name, make_model in model_factories:
            metrics = evaluate(make_model(), stream)
            rows.append({"seed": seed, "model": model_name, **metrics})

    summary = summarize(rows)
    best_single = max(
        ("single_fast", "single_medium", "single_slow"),
        key=lambda name: summary[name]["all_mean"],
    )

    payload = {
        "config": vars(args) | {"out": str(args.out)},
        "rows": rows,
        "summary": summary,
        "best_single": best_single,
        "cms_gain_over_best_single_points": 100.0
        * (summary["cms_gated"]["all_mean"] - summary[best_single]["all_mean"]),
    }
    args.out.write_text(json.dumps(payload, indent=2) + "\n")

    print("Nested-memory local POC")
    print(f"seeds={args.seeds}, facts={args.num_facts}, dim={args.dim}, classes={args.num_classes}")
    print()
    print(f"{'model':<16} {'all':>10} {'volatile':>10} {'stable':>10}")
    print("-" * 50)
    for model_name in ("static_initial", "single_fast", "single_medium", "single_slow", "cms_gated"):
        metrics = summary[model_name]
        print(
            f"{model_name:<16} "
            f"{format_pct(metrics['all_mean']):>10} "
            f"{format_pct(metrics['volatile_mean']):>10} "
            f"{format_pct(metrics['stable_mean']):>10}"
        )
    print("-" * 50)
    print(
        "CMS gain over best single-timescale "
        f"({best_single}): {payload['cms_gain_over_best_single_points']:.1f} points"
    )
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
