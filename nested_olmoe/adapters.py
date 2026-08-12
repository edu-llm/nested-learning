from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F


class RMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        variance = hidden_states.float().pow(2).mean(dim=-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance.to(hidden_states.dtype) + self.eps)
        return self.weight * hidden_states


@dataclass
class CmsAdapterConfig:
    hidden_size: int
    rank: int = 128
    chunk_sizes: tuple[int, ...] = (128, 512, 2048)
    learning_rates: tuple[float, ...] = (0.30, 0.12, 0.035)
    dropout: float = 0.0
    init_scale: float = 1e-3
    output_scale: float = 0.05
    memory_update_rule: str = "delta"
    retention_factor: float = 0.995
    update_momentum: float = 0.0
    self_reference_scale: float = 0.0
    detach_memory_updates: bool = True
    aux_loss_weight: float = 0.05

    def __post_init__(self) -> None:
        if len(self.chunk_sizes) != len(self.learning_rates):
            raise ValueError("chunk_sizes and learning_rates must have the same length")
        if self.memory_update_rule not in {"delta", "hebbian"}:
            raise ValueError("memory_update_rule must be 'delta' or 'hebbian'")
        if not 0.0 <= self.update_momentum < 1.0:
            raise ValueError("update_momentum must be in [0, 1)")


class StaticSwiGLUAdapter(nn.Module):
    """Parameter-matched static adapter control."""

    def __init__(self, hidden_size: int, rank: int, dropout: float, output_scale: float) -> None:
        super().__init__()
        self.norm = RMSNorm(hidden_size)
        self.gate = nn.Linear(hidden_size, rank, bias=False)
        self.up = nn.Linear(hidden_size, rank, bias=False)
        self.down = nn.Linear(rank, hidden_size, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.output_scale = nn.Parameter(torch.tensor(float(output_scale)))
        self.residual_gate = nn.Linear(hidden_size, 1, bias=True)

        nn.init.normal_(self.gate.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.up.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.down.weight)
        nn.init.constant_(self.residual_gate.bias, -4.0)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        x = self.norm(hidden_states)
        update = self.down(F.silu(self.gate(x)) * self.up(x))
        gate = torch.sigmoid(self.residual_gate(x))
        return gate * self.dropout(update) * self.output_scale


class CmsMemoryAdapter(nn.Module):
    """Chunked multi-timescale associative memory adapter.

    The adapter reads hidden states, retrieves from per-example memory states,
    and updates those states with a delta-rule objective once per chunk. The
    memory state is reset each forward call, which makes this a test-time
    in-context memory mechanism rather than persistent weight training.
    """

    def __init__(self, config: CmsAdapterConfig) -> None:
        super().__init__()
        self.config = config
        self.norm = RMSNorm(config.hidden_size)
        self.key = nn.Linear(config.hidden_size, config.rank, bias=False)
        self.value = nn.Linear(config.hidden_size, config.rank, bias=False)
        self.self_value = nn.Linear(config.rank, config.rank, bias=False)
        self.out = nn.Linear(config.rank, config.hidden_size, bias=False)
        self.gate = nn.Linear(config.hidden_size, 1, bias=True)
        self.dropout = nn.Dropout(config.dropout)
        self.output_scale = nn.Parameter(torch.tensor(float(config.output_scale)))
        self.level_logits = nn.Parameter(torch.zeros(len(config.chunk_sizes)))
        self.initial_memories = nn.ParameterList(
            [
                nn.Parameter(torch.empty(config.rank, config.rank))
                for _ in config.chunk_sizes
            ]
        )
        self._last_aux_loss: torch.Tensor | None = None

        nn.init.normal_(self.key.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.value.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.self_value.weight)
        nn.init.zeros_(self.out.weight)
        nn.init.constant_(self.gate.bias, -4.0)
        for memory in self.initial_memories:
            nn.init.normal_(memory, mean=0.0, std=config.init_scale)

    @property
    def last_aux_loss(self) -> torch.Tensor | None:
        return self._last_aux_loss

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if hidden_states.ndim != 3:
            raise ValueError(f"expected [batch, seq, hidden], got {tuple(hidden_states.shape)}")

        x = self.norm(hidden_states)
        keys = F.normalize(self.key(x).float(), p=2, dim=-1).to(x.dtype)
        values = self.value(x)
        level_weights = torch.softmax(self.level_logits, dim=0)
        level_outputs: list[torch.Tensor] = []
        aux_losses: list[torch.Tensor] = []

        for level_id, (chunk_size, learning_rate) in enumerate(
            zip(self.config.chunk_sizes, self.config.learning_rates)
        ):
            initial = self.initial_memories[level_id].to(dtype=hidden_states.dtype)
            memory = initial.unsqueeze(0).expand(hidden_states.shape[0], -1, -1).clone()
            velocity = torch.zeros_like(memory)
            chunks: list[torch.Tensor] = []

            for start in range(0, hidden_states.shape[1], chunk_size):
                end = min(start + chunk_size, hidden_states.shape[1])
                k_chunk = keys[:, start:end, :]
                base_v_chunk = values[:, start:end, :]
                retrieved = torch.bmm(k_chunk, memory)
                if self.config.self_reference_scale != 0.0:
                    self_values = self.self_value(retrieved)
                    v_chunk = base_v_chunk + float(self.config.self_reference_scale) * self_values
                else:
                    v_chunk = base_v_chunk
                chunks.append(retrieved)
                aux_losses.append(F.mse_loss(retrieved.float(), base_v_chunk.float()))

                update_keys = k_chunk.detach() if self.config.detach_memory_updates else k_chunk
                denom = max(1, end - start)
                if self.config.memory_update_rule == "delta":
                    update_error = (v_chunk - retrieved).detach() if self.config.detach_memory_updates else (v_chunk - retrieved)
                    delta = torch.einsum("blr,bld->brd", update_keys, update_error) / denom
                else:
                    update_values = v_chunk.detach() if self.config.detach_memory_updates else v_chunk
                    delta = torch.einsum("blr,bld->brd", update_keys, update_values) / denom

                velocity = float(self.config.update_momentum) * velocity + delta
                memory = float(self.config.retention_factor) * memory + float(learning_rate) * velocity
                if self.config.detach_memory_updates:
                    memory = memory.detach()
                    velocity = velocity.detach()

            level_outputs.append(torch.cat(chunks, dim=1) * level_weights[level_id])

        combined = self.out(torch.stack(level_outputs, dim=0).sum(dim=0))
        gate = torch.sigmoid(self.gate(x))
        self._last_aux_loss = torch.stack(aux_losses).mean() if aux_losses else None
        return gate * self.dropout(combined) * self.output_scale
