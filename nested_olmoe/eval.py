from __future__ import annotations

from typing import Any

import torch

from .data import VALUES, FactExample
from .data import make_eval_examples, make_language_interference_example, make_needle_example


GENERAL_GUARDRAIL_TEXTS = [
    "The history of scientific discovery is full of careful measurements, failed hypotheses, and revised theories.",
    "A useful assistant should preserve instructions, reason through the evidence, and avoid inventing unsupported facts.",
    "Machine learning systems are evaluated not only by accuracy, but also by calibration, robustness, and efficiency.",
    "When a model reads a long document, it must distinguish relevant details from repeated distractors.",
]


@torch.no_grad()
def score_multiple_choice(
    model: torch.nn.Module,
    tokenizer: Any,
    examples: list[FactExample],
    *,
    device: torch.device,
    max_length: int,
) -> dict[str, float]:
    model.eval()
    correct = []
    volatile_correct = []
    stable_correct = []

    for example in examples:
        scores: list[float] = []
        for candidate in VALUES:
            prompt_ids = tokenizer(example.prompt, add_special_tokens=True).input_ids
            answer_ids = tokenizer(" " + candidate, add_special_tokens=False).input_ids
            input_ids = (prompt_ids + answer_ids)[-max_length:]
            prompt_len = min(len(prompt_ids), len(input_ids))
            labels = [-100] * prompt_len + input_ids[prompt_len:]

            batch = {
                "input_ids": torch.tensor([input_ids], dtype=torch.long, device=device),
                "attention_mask": torch.ones(1, len(input_ids), dtype=torch.long, device=device),
                "labels": torch.tensor([labels], dtype=torch.long, device=device),
            }
            outputs = model(**batch, use_cache=False)
            scores.append(float(outputs.loss.detach().cpu()))

        prediction = VALUES[min(range(len(scores)), key=lambda i: scores[i])]
        ok = prediction == example.answer.strip()
        correct.append(ok)
        if example.is_volatile:
            volatile_correct.append(ok)
        else:
            stable_correct.append(ok)

    def acc(items: list[bool]) -> float:
        return float(sum(items) / len(items)) if items else 0.0

    return {
        "eval_accuracy": acc(correct),
        "eval_volatile_accuracy": acc(volatile_correct),
        "eval_stable_accuracy": acc(stable_correct),
    }


def make_eval_suites(
    *,
    seed: int,
    size: int,
    num_facts: int,
    include_facts: bool = True,
    include_needle: bool = True,
    include_language: bool = True,
) -> dict[str, list[FactExample]]:
    suites: dict[str, list[FactExample]] = {}
    if include_facts:
        suites["facts"] = make_eval_examples(seed=seed, size=size, num_facts=num_facts)
    if include_needle:
        suites["needle"] = [make_needle_example(seed=seed + 100000 + idx) for idx in range(size)]
    if include_language:
        suites["language"] = [
            make_language_interference_example(seed=seed + 200000 + idx)
            for idx in range(size)
        ]
    return suites


def score_eval_suites(
    model: torch.nn.Module,
    tokenizer: Any,
    suites: dict[str, list[FactExample]],
    *,
    device: torch.device,
    max_length: int,
) -> dict[str, float]:
    metrics: dict[str, float] = {}
    for suite_name, examples in suites.items():
        suite_metrics = score_multiple_choice(
            model,
            tokenizer,
            examples,
            device=device,
            max_length=max_length,
        )
        for key, value in suite_metrics.items():
            metrics[f"{suite_name}_{key}"] = value
    return metrics


@torch.no_grad()
def score_perplexity_texts(
    model: torch.nn.Module,
    tokenizer: Any,
    *,
    device: torch.device,
    max_length: int,
) -> dict[str, float]:
    losses = []
    model.eval()
    for text in GENERAL_GUARDRAIL_TEXTS:
        input_ids = tokenizer(text, add_special_tokens=True).input_ids[-max_length:]
        batch = {
            "input_ids": torch.tensor([input_ids], dtype=torch.long, device=device),
            "attention_mask": torch.ones(1, len(input_ids), dtype=torch.long, device=device),
            "labels": torch.tensor([input_ids], dtype=torch.long, device=device),
        }
        outputs = model(**batch, use_cache=False)
        losses.append(float(outputs.loss.detach().cpu()))
    mean_loss = sum(losses) / len(losses)
    return {
        "guardrail_loss": mean_loss,
        "guardrail_ppl": float(torch.exp(torch.tensor(mean_loss)).item()),
    }
