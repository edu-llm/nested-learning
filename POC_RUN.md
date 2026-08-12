# Local Proof-of-Concept Run

Command:

```bash
python3 continual_memory_poc.py
```

Result:

| Model | All | Volatile facts | Stable facts |
|---|---:|---:|---:|
| static_initial | 36.5% | 2.8% | 70.2% |
| single_fast | 43.8% | 46.5% | 41.0% |
| single_medium | 46.6% | 36.1% | 57.1% |
| single_slow | 44.8% | 16.2% | 73.2% |
| cms_gated | 54.2% | 39.0% | 69.5% |

The CMS-style gated memory beat the best single-timescale memory
(`single_medium`) by 7.6 percentage points overall.

Interpretation:

This is a synthetic smoke test, not evidence that a Qwen3-scale architecture
works. The result does support the narrow mechanism the larger experiment would
test: when the stream mixes volatile and stable knowledge, a multi-timescale
memory with routing can outperform any single update speed.

Files:

- `continual_memory_poc.py`: dependency-free local experiment.
- `poc_results.json`: full per-seed results and summary.
