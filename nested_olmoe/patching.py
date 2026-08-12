from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable

import torch
from torch import nn

from .adapters import CmsAdapterConfig, CmsMemoryAdapter, StaticSwiGLUAdapter


@dataclass
class InjectionReport:
    injected_modules: list[str]
    skipped_candidates: list[str]


class ResidualAdapterWrapper(nn.Module):
    def __init__(self, wrapped: nn.Module, adapter: nn.Module) -> None:
        super().__init__()
        self.wrapped = wrapped
        self.adapter = adapter

    def forward(self, *args, **kwargs):
        output = self.wrapped(*args, **kwargs)
        if isinstance(output, tuple):
            hidden = output[0]
            return (hidden + self.adapter(hidden), *output[1:])
        return output + self.adapter(output)


def _get_submodule_parent(model: nn.Module, module_name: str) -> tuple[nn.Module, str]:
    parts = module_name.split(".")
    parent = model
    for part in parts[:-1]:
        parent = getattr(parent, part)
    return parent, parts[-1]


def _extract_layer_id(module_name: str) -> int | None:
    match = re.search(r"(?:layers|h|blocks)\.(\d+)\.", module_name)
    return int(match.group(1)) if match else None


def _candidate_modules(model: nn.Module) -> list[str]:
    candidates = []
    for name, module in model.named_modules():
        lowered = name.lower()
        if lowered.endswith(".mlp") or "moe" in lowered or "feed_forward" in lowered:
            if not isinstance(module, ResidualAdapterWrapper):
                candidates.append(name)
    return candidates


def inject_nested_adapters(
    model: nn.Module,
    *,
    mode: str,
    hidden_size: int,
    rank: int,
    target_pattern: str = r"^model\.layers\.\d+\.mlp$",
    first_layer: int = 16,
    last_layer: int | None = None,
    chunk_sizes: Iterable[int] = (128, 512, 2048),
    learning_rates: Iterable[float] = (0.30, 0.12, 0.035),
    dropout: float = 0.0,
    output_scale: float = 0.05,
    memory_update_rule: str = "delta",
    retention_factor: float = 0.995,
    update_momentum: float = 0.0,
    self_reference_scale: float = 0.0,
    detach_memory_updates: bool = True,
    aux_loss_weight: float = 0.05,
) -> InjectionReport:
    if mode not in {"static", "single_memory", "cms"}:
        raise ValueError(f"unsupported adapter mode: {mode}")

    pattern = re.compile(target_pattern)
    injected: list[str] = []
    skipped: list[str] = []
    module_names = [name for name, _ in model.named_modules()]

    for module_name in module_names:
        if not pattern.match(module_name):
            continue
        layer_id = _extract_layer_id(module_name)
        if layer_id is not None and layer_id < first_layer:
            skipped.append(module_name)
            continue
        if last_layer is not None and layer_id is not None and layer_id > last_layer:
            skipped.append(module_name)
            continue

        parent, child_name = _get_submodule_parent(model, module_name)
        wrapped = getattr(parent, child_name)
        if mode == "static":
            adapter = StaticSwiGLUAdapter(hidden_size, rank, dropout, output_scale)
        else:
            chunks = tuple(chunk_sizes)
            lrs = tuple(learning_rates)
            if mode == "single_memory":
                chunks = (chunks[0],)
                lrs = (lrs[0],)
            adapter = CmsMemoryAdapter(
                CmsAdapterConfig(
                    hidden_size=hidden_size,
                    rank=rank,
                    chunk_sizes=chunks,
                    learning_rates=lrs,
                    dropout=dropout,
                    output_scale=output_scale,
                    memory_update_rule=memory_update_rule,
                    retention_factor=retention_factor,
                    update_momentum=update_momentum,
                    self_reference_scale=self_reference_scale,
                    detach_memory_updates=detach_memory_updates,
                    aux_loss_weight=aux_loss_weight,
                )
            )
        setattr(parent, child_name, ResidualAdapterWrapper(wrapped, adapter))
        injected.append(module_name)

    if not injected:
        candidates = "\n".join(f"  - {name}" for name in _candidate_modules(model)[:80])
        raise RuntimeError(
            "No modules matched target_pattern. "
            f"pattern={target_pattern!r}. Candidate MoE/MLP modules:\n{candidates}"
        )

    return InjectionReport(injected, skipped)


def collect_cms_aux_loss(model: nn.Module) -> torch.Tensor | None:
    losses: list[torch.Tensor] = []
    for module in model.modules():
        adapter = getattr(module, "adapter", None)
        if isinstance(adapter, CmsMemoryAdapter) and adapter.last_aux_loss is not None:
            losses.append(adapter.last_aux_loss)
    return torch.stack(losses).mean() if losses else None


def mark_trainable_parameters(model: nn.Module, *, train_base: bool = False) -> dict[str, int]:
    for _, param in model.named_parameters():
        param.requires_grad = bool(train_base)

    for module in model.modules():
        if isinstance(module, ResidualAdapterWrapper):
            for param in module.adapter.parameters():
                param.requires_grad = True

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return {"trainable": trainable, "total": total}


def adapter_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: tensor.detach().cpu()
        for name, tensor in model.state_dict().items()
        if ".adapter." in name
    }
