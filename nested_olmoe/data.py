from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any

import torch
from torch.utils.data import Dataset


VALUES = [
    "amber",
    "blue",
    "cedar",
    "dawn",
    "ember",
    "frost",
    "gold",
    "harbor",
    "ivory",
    "jade",
    "kelp",
    "lilac",
    "moss",
    "navy",
    "opal",
    "pearl",
]


@dataclass
class FactExample:
    prompt: str
    answer: str
    is_volatile: bool
    suite: str = "facts"


def _new_value(rng: random.Random, old: str | None = None) -> str:
    choices = [value for value in VALUES if value != old]
    return rng.choice(choices)


def make_fact_example(
    *,
    seed: int,
    num_facts: int = 32,
    volatile_fraction: float = 0.5,
    max_updates_per_volatile: int = 3,
    stable_repetitions: int = 2,
) -> FactExample:
    rng = random.Random(seed)
    num_volatile = int(num_facts * volatile_fraction)
    fact_ids = [f"unit-{seed % 10000:04d}-{i:03d}" for i in range(num_facts)]
    values = {fact_id: _new_value(rng) for fact_id in fact_ids}
    volatile_ids = set(fact_ids[:num_volatile])
    lines: list[str] = [
        "You are reading a temporary registry. Answer using the latest stated code.",
    ]

    order = fact_ids[:]
    rng.shuffle(order)
    for fact_id in order:
        lines.append(f"Fact: {fact_id} has code {values[fact_id]}.")

    for _ in range(stable_repetitions):
        stable_ids = [fact_id for fact_id in fact_ids if fact_id not in volatile_ids]
        rng.shuffle(stable_ids)
        for fact_id in stable_ids[: max(1, len(stable_ids) // 3)]:
            lines.append(f"Reminder: {fact_id} still has code {values[fact_id]}.")

    volatile_order = list(volatile_ids)
    rng.shuffle(volatile_order)
    for fact_id in volatile_order:
        for _ in range(rng.randint(1, max_updates_per_volatile)):
            values[fact_id] = _new_value(rng, old=values[fact_id])
            lines.append(f"Update: {fact_id} now has code {values[fact_id]}.")
            if rng.random() < 0.35:
                distractor = rng.choice(fact_ids)
                lines.append(f"Note: {distractor} has code {values[distractor]}.")

    query_id = rng.choice(fact_ids)
    answer = values[query_id]
    is_volatile = query_id in volatile_ids
    prompt = "\n".join(lines) + f"\nQuestion: What is the current code for {query_id}?\nAnswer:"
    return FactExample(prompt=prompt, answer=" " + answer, is_volatile=is_volatile, suite="facts")


def make_needle_example(
    *,
    seed: int,
    context_tokens: int = 900,
    num_needles: int = 4,
) -> FactExample:
    rng = random.Random(seed)
    needles = []
    values: dict[str, str] = {}
    for i in range(num_needles):
        key = f"needle-{seed % 10000:04d}-{i:02d}"
        values[key] = _new_value(rng)
        needles.append(key)

    filler_words = [
        "archive",
        "window",
        "signal",
        "ledger",
        "orbit",
        "matrix",
        "garden",
        "lantern",
    ]
    filler = [
        " ".join(rng.choice(filler_words) for _ in range(18))
        for _ in range(max(8, context_tokens // 18))
    ]
    insert_positions = sorted(rng.sample(range(len(filler)), k=min(num_needles, len(filler))))
    for pos, key in zip(insert_positions, needles):
        filler[pos] = f"Hidden record: {key} has code {values[key]}. " + filler[pos]

    query_key = rng.choice(needles)
    prompt = (
        "Read the noisy document and answer with the exact current code.\n"
        + "\n".join(filler)
        + f"\nQuestion: What is the code for {query_key}?\nAnswer:"
    )
    return FactExample(prompt=prompt, answer=" " + values[query_key], is_volatile=False, suite="needle")


def make_language_interference_example(*, seed: int, vocab_size: int = 12) -> FactExample:
    rng = random.Random(seed)
    roots_a = [f"mav{i:02d}" for i in range(vocab_size)]
    roots_b = [f"tor{i:02d}" for i in range(vocab_size)]
    meanings_a = {word: _new_value(rng) for word in roots_a}
    meanings_b = {word: _new_value(rng) for word in roots_b}

    lines = ["Learn two toy languages. Answer using the latest lexicon entry."]
    for word, value in meanings_a.items():
        lines.append(f"Language A: {word} means {value}.")
    for word, value in meanings_b.items():
        lines.append(f"Language B: {word} means {value}.")

    for _ in range(max(3, vocab_size // 3)):
        word = rng.choice(roots_a)
        meanings_a[word] = _new_value(rng, old=meanings_a[word])
        lines.append(f"Correction in Language A: {word} now means {meanings_a[word]}.")
    for _ in range(max(3, vocab_size // 3)):
        word = rng.choice(roots_b)
        meanings_b[word] = _new_value(rng, old=meanings_b[word])
        lines.append(f"Correction in Language B: {word} now means {meanings_b[word]}.")

    query_a = rng.random() < 0.5
    query_word = rng.choice(roots_a if query_a else roots_b)
    answer = meanings_a[query_word] if query_a else meanings_b[query_word]
    language = "A" if query_a else "B"
    prompt = "\n".join(lines) + f"\nQuestion: In Language {language}, what does {query_word} mean?\nAnswer:"
    return FactExample(prompt=prompt, answer=" " + answer, is_volatile=True, suite="language")


class EpisodicFactsDataset(Dataset):
    def __init__(
        self,
        *,
        tokenizer: Any,
        size: int,
        seed: int,
        max_length: int,
        num_facts: int,
        train_suites: list[str] | None = None,
    ) -> None:
        self.tokenizer = tokenizer
        self.size = size
        self.seed = seed
        self.max_length = max_length
        self.num_facts = num_facts
        self.train_suites = train_suites or ["facts"]

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, idx: int) -> dict[str, Any]:
        rng = random.Random(self.seed + idx * 7919)
        suite = rng.choice(self.train_suites)
        example_seed = self.seed + idx
        if suite == "facts":
            example = make_fact_example(seed=example_seed, num_facts=self.num_facts)
        elif suite == "needle":
            example = make_needle_example(seed=example_seed)
        elif suite == "language":
            example = make_language_interference_example(seed=example_seed)
        else:
            raise ValueError(f"unsupported train suite: {suite}")
        prompt_ids = self.tokenizer(example.prompt, add_special_tokens=True).input_ids
        answer_ids = self.tokenizer(example.answer, add_special_tokens=False).input_ids
        eos = [self.tokenizer.eos_token_id] if self.tokenizer.eos_token_id is not None else []
        full_input_ids = prompt_ids + answer_ids + eos
        full_labels = [-100] * len(prompt_ids) + answer_ids + eos
        input_ids = full_input_ids[-self.max_length :]
        labels = full_labels[-self.max_length :]
        return {
            "input_ids": input_ids,
            "labels": labels,
            "is_volatile": example.is_volatile,
        }


def collate_batch(batch: list[dict[str, Any]], pad_token_id: int) -> dict[str, torch.Tensor]:
    max_len = max(len(item["input_ids"]) for item in batch)
    input_ids = []
    labels = []
    attention_mask = []
    is_volatile = []
    for item in batch:
        pad_len = max_len - len(item["input_ids"])
        input_ids.append(item["input_ids"] + [pad_token_id] * pad_len)
        labels.append(item["labels"] + [-100] * pad_len)
        attention_mask.append([1] * len(item["input_ids"]) + [0] * pad_len)
        is_volatile.append(bool(item["is_volatile"]))
    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
        "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
        "is_volatile": torch.tensor(is_volatile, dtype=torch.bool),
    }


def make_eval_examples(*, seed: int, size: int, num_facts: int) -> list[FactExample]:
    return [make_fact_example(seed=seed + idx, num_facts=num_facts) for idx in range(size)]
