# NL.pdf Coverage Audit For The OLMoE Run

This audit maps the claims, mechanisms, and evaluation requirements in
`NL.pdf` to the current OLMoE experiment scaffold.

## Scope

The OLMoE scaffold is designed to test the central Nested Learning mechanism:

```text
Do nested, multi-timescale associative memories improve continual in-context
adaptation over static adapters, raw ICL, and single-timescale memory?
```

It is not a full reproduction of every experiment in the paper. In particular,
it does not reproduce 50B-token from-scratch language-model training,
BABILong at million-token contexts, ImageNet optimizer training, or every
published baseline model. Those are separate large-scale studies.

## Coverage Matrix

| NL.pdf requirement / finding | Covered in this run? | Implementation |
|---|---:|---|
| Use a strong base model | Yes | `allenai/OLMoE-1B-7B-0125` in all configs |
| Treat architecture modules as associative memories | Yes | `CmsMemoryAdapter` maps keys to values via online memory state |
| Multi-level nested update frequencies | Yes | CMS chunk levels: default `128`, `512`, `2048` |
| Compare against no nested memory / ICL | Yes | `configs/olmoe_base_eval.json` |
| Compare against continued finetuning | Optional | `configs/olmoe_full_finetune_control.json` trains the base model |
| Compare against extra static parameters | Yes | `configs/olmoe_static_adapter_cluster.json` |
| Compare against one memory level | Yes | `configs/olmoe_single_memory_cluster.json` |
| Test number of levels | Yes | generated `cms_two_levels`, `cms`, `cms_four_levels` configs |
| Test lowest-frequency / timescale choice | Partial | 2/3/4-level sweep changes frequency set; a finer frequency grid can be added |
| Use learned initial memory state / meta-initialization | Yes | `initial_memories` are trainable parameters |
| Knowledge transfer via direct conditioning | Yes | adapter output is residual-conditioned on OLMoE hidden states |
| Knowledge transfer through backprop | Partial | default detaches memory updates for stability; `cms_bptt_updates` enables gradient flow through updates |
| Independent/head-wise CMS aggregation | Yes | per-level outputs are combined with learned `level_logits` |
| Sequential/nested CMS variants | No | not implemented in this scaffold |
| DGD / delta-rule update | Yes | default `memory_update_rule: "delta"` |
| No-DGD ablation | Yes | generated `cms_no_dgd` uses `memory_update_rule: "hebbian"` |
| Retention / weight decay in memory update | Yes | `retention_factor: 0.995` |
| No-retention ablation | Yes | generated `cms_no_retention` uses `retention_factor: 1.0` |
| Momentum in memory update | Yes | `update_momentum: 0.5` |
| No-momentum ablation | Yes | generated `cms_no_momentum` |
| Self-referential value generation | Partial | `self_reference_scale` adds a learned self-value path; not full self-referential Titans |
| No self-reference ablation | Yes | generated `cms_no_self_reference` |
| Full Hope architecture | Partial | this is closer to Hope-Attention/CMS; full self-modifying Titans is not implemented |
| Optimizers as associative memories | Partial | AdamW default; `M3LiteAdamW` tests multi-timescale gradient memory without Newton-Schulz |
| M3 optimizer exactly as in paper | No | `m3lite` omits Newton-Schulz orthogonalization |
| Architecture-specific optimizer claim | Partial | generated `cms_m3lite_optimizer` compares AdamW vs M3-lite on the same adapter |
| Continual learning / interference eval | Yes | episodic fact stream with updates and stable/volatile split |
| New-language in-context eval | Yes | toy-language interference suite |
| Long-context retrieval / NIAH eval | Partial | synthetic needle retrieval within OLMoE's 4096-token context |
| General capability guardrail | Yes | small built-in text perplexity guardrail |
| Class-incremental CLINC/Banking/DBpedia | No | synthetic class/fact stream is a proxy, not those datasets |
| QASPER/LongHealth/RULER/BABILong | Partial/No | local needle proxy only; official benchmark integrations are not included |
| Formal language recognition | No | should be a separate synthetic benchmark integration |
| In-context recall/MAD | Partial | episodic facts and needle retrieval test recall/compression; MAD itself is not included |
| Catastrophic forgetting solved? | No claim | run measures forgetting/interference but does not claim a general solution |

## Generated Paper-Level Ablation Matrix

Generate the full paper-aligned matrix:

```bash
python3 scripts/make_paper_ablation_configs.py
```

This writes 36 configs:

- `base_eval`
- `static_adapter`
- `single_memory`
- `cms`
- `cms_no_dgd`
- `cms_no_momentum`
- `cms_no_retention`
- `cms_no_self_reference`
- `cms_two_levels`
- `cms_four_levels`
- `cms_bptt_updates`
- `cms_m3lite_optimizer`

Each variant is generated for seeds `17`, `23`, and `42`.

Launch the full matrix:

```bash
sbatch scripts/run_paper_ablation_array.slurm
```

## Minimum Claim Criteria

For the OLMoE run to support the Nested Learning mechanism:

```text
cms > base_eval
cms > static_adapter
cms > single_memory
cms > cms_no_dgd
cms > cms_no_momentum
cms > cms_no_retention
cms >= cms_two_levels
cms_four_levels is competitive or better than cms
guardrail_ppl does not materially degrade
```

The most important signal is not whether CMS beats raw OLMoE. The important
signal is whether CMS beats parameter-matched static adapters and
single-timescale memory under the same train/eval setup.

For a heavyweight "more training caused it" control:

```bash
CONFIG=configs/olmoe_full_finetune_control.json sbatch scripts/run_olmoe_cms_cluster.slurm
```

## Remaining High-Value Additions

1. Add official RULER or BABILong integration for real long-context retrieval.
   OLMoE's native 4096-token context limits this; use Qwen3 or Mistral NeMo for
   the long-context replication.

2. Add exact M3/Muon-style optimizer support with Newton-Schulz for matrix
   parameters. The current `m3lite` is deliberately safer but less faithful.

3. Add full self-referential Titans modules that update `k`, `v`, `q`, `eta`,
   `alpha`, and memory projections, rather than the lighter self-value path.

4. Add formal-language recognition and MAD-style synthetic suites.

5. Add a real general-capability evaluation harness, such as lm-eval, after the
   architecture signal is established.
