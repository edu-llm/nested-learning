#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch
from accelerate import Accelerator, DistributedDataParallelKwargs
from torch.utils.data import DataLoader
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup, set_seed

from nested_olmoe import adapter_state_dict, collect_cms_aux_loss, inject_nested_adapters, mark_trainable_parameters
from nested_olmoe.data import EpisodicFactsDataset, collate_batch
from nested_olmoe.eval import make_eval_suites, score_eval_suites, score_perplexity_texts
from nested_olmoe.optim import M3LiteAdamW


def load_config(path: Path) -> dict:
    with path.open() as handle:
        return json.load(handle)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()
    cfg = load_config(args.config)
    output_dir = args.output_dir or Path(cfg["run"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    set_seed(int(cfg["run"]["seed"]))
    ddp_options = {
        "find_unused_parameters": bool(cfg["train"].get("ddp_find_unused_parameters", True))
    }
    if bool(cfg["train"].get("ddp_static_graph", True)):
        ddp_options["static_graph"] = True
    try:
        ddp_kwargs = DistributedDataParallelKwargs(**ddp_options)
    except TypeError:
        ddp_options.pop("static_graph", None)
        ddp_kwargs = DistributedDataParallelKwargs(**ddp_options)
    accelerator = Accelerator(
        gradient_accumulation_steps=1,
        mixed_precision=cfg["train"].get("mixed_precision", "bf16"),
        log_with=None,
        kwargs_handlers=[ddp_kwargs],
    )

    tokenizer = AutoTokenizer.from_pretrained(cfg["model"]["name"], trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model_config = AutoConfig.from_pretrained(cfg["model"]["name"], trust_remote_code=True)
    dtype = torch.bfloat16 if cfg["train"].get("mixed_precision", "bf16") == "bf16" else torch.float16
    model_kwargs = {
        "config": model_config,
        "torch_dtype": dtype,
        "trust_remote_code": True,
    }
    if cfg["model"].get("attn_implementation"):
        model_kwargs["attn_implementation"] = cfg["model"]["attn_implementation"]
    model = AutoModelForCausalLM.from_pretrained(cfg["model"]["name"], **model_kwargs)
    model.config.use_cache = False
    if cfg["train"].get("gradient_checkpointing", True):
        checkpoint_kwargs = {
            "use_reentrant": bool(cfg["train"].get("gradient_checkpointing_use_reentrant", False))
        }
        try:
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs=checkpoint_kwargs)
        except TypeError:
            if checkpoint_kwargs["use_reentrant"]:
                model.gradient_checkpointing_enable()
            elif accelerator.is_main_process:
                print(
                    json.dumps(
                        {
                            "warning": "gradient_checkpointing_disabled",
                            "reason": "non_reentrant_checkpointing_not_supported",
                        }
                    )
                )

    adapter_cfg = cfg["adapter"]
    if adapter_cfg["mode"] != "none":
        report = inject_nested_adapters(
            model,
            mode=adapter_cfg["mode"],
            hidden_size=int(model_config.hidden_size),
            rank=int(adapter_cfg["rank"]),
            target_pattern=adapter_cfg.get("target_pattern", r"^model\.layers\.\d+\.mlp$"),
            first_layer=int(adapter_cfg["first_layer"]),
            last_layer=adapter_cfg.get("last_layer", None),
            chunk_sizes=tuple(int(x) for x in adapter_cfg.get("chunk_sizes", [128, 512, 2048])),
            learning_rates=tuple(float(x) for x in adapter_cfg.get("learning_rates", [0.30, 0.12, 0.035])),
            dropout=float(adapter_cfg.get("dropout", 0.0)),
            output_scale=float(adapter_cfg.get("output_scale", 0.05)),
            memory_update_rule=adapter_cfg.get("memory_update_rule", "delta"),
            retention_factor=float(adapter_cfg.get("retention_factor", 0.995)),
            update_momentum=float(adapter_cfg.get("update_momentum", 0.0)),
            self_reference_scale=float(adapter_cfg.get("self_reference_scale", 0.0)),
            detach_memory_updates=bool(adapter_cfg.get("detach_memory_updates", True)),
            aux_loss_weight=float(adapter_cfg.get("aux_loss_weight", 0.05)),
        )
    else:
        report = None

    counts = mark_trainable_parameters(model, train_base=bool(cfg["train"].get("train_base", False)))
    if accelerator.is_main_process:
        metadata = {
            "config": cfg,
            "parameter_counts": counts,
            "injection_report": None if report is None else report.__dict__,
        }
        (output_dir / "run_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
        print(json.dumps(metadata, indent=2))

    eval_suites = make_eval_suites(
        seed=int(cfg["data"].get("eval_seed", 900000)),
        size=int(cfg["data"]["eval_size"]),
        num_facts=int(cfg["data"]["num_facts"]),
        include_facts=True,
        include_needle=bool(cfg["data"].get("include_needle_eval", True)),
        include_language=bool(cfg["data"].get("include_language_eval", True)),
    )
    max_steps = int(cfg["train"]["max_steps"])

    if max_steps == 0:
        model = accelerator.prepare(model)
        unwrapped = accelerator.unwrap_model(model)
        metrics = score_eval_suites(
            unwrapped,
            tokenizer,
            eval_suites,
            device=accelerator.device,
            max_length=int(cfg["data"]["max_length"]),
        )
        if bool(cfg["data"].get("include_guardrail_ppl", True)):
            metrics.update(
                score_perplexity_texts(
                    unwrapped,
                    tokenizer,
                    device=accelerator.device,
                    max_length=int(cfg["data"]["max_length"]),
                )
            )
        if accelerator.is_main_process:
            print(json.dumps({"step": 0, **metrics}))
            with (output_dir / "metrics.jsonl").open("a") as handle:
                handle.write(json.dumps({"step": 0, **metrics}) + "\n")
        return

    train_dataset = EpisodicFactsDataset(
        tokenizer=tokenizer,
        size=int(cfg["data"]["train_size"]),
        seed=int(cfg["data"]["seed"]),
        max_length=int(cfg["data"]["max_length"]),
        num_facts=int(cfg["data"]["num_facts"]),
        train_suites=list(cfg["data"].get("train_suites", ["facts"])),
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=int(cfg["train"]["per_device_batch_size"]),
        shuffle=True,
        collate_fn=lambda batch: collate_batch(batch, tokenizer.pad_token_id),
        num_workers=int(cfg["train"].get("num_workers", 2)),
        pin_memory=True,
    )

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    if not trainable_params:
        raise ValueError("No trainable parameters. Use max_steps=0 for eval-only configs.")
    optimizer_name = cfg["train"].get("optimizer", "adamw")
    if optimizer_name == "adamw":
        optimizer = torch.optim.AdamW(
            trainable_params,
            lr=float(cfg["train"]["learning_rate"]),
            betas=tuple(float(x) for x in cfg["train"].get("betas", [0.9, 0.95])),
            weight_decay=float(cfg["train"].get("weight_decay", 0.01)),
        )
    elif optimizer_name == "m3lite":
        optimizer = M3LiteAdamW(
            trainable_params,
            lr=float(cfg["train"]["learning_rate"]),
            betas=tuple(float(x) for x in cfg["train"].get("betas", [0.9, 0.95])),
            weight_decay=float(cfg["train"].get("weight_decay", 0.01)),
            slow_beta=float(cfg["train"].get("slow_beta", 0.99)),
            slow_interval=int(cfg["train"].get("slow_interval", 16)),
            slow_weight=float(cfg["train"].get("slow_weight", 0.25)),
        )
    else:
        raise ValueError(f"unsupported optimizer: {optimizer_name}")
    warmup_steps = int(cfg["train"].get("warmup_steps", max(1, max_steps // 20)))
    lr_scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, max_steps)

    model, optimizer, train_loader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_loader, lr_scheduler
    )

    accumulation_steps = max(1, int(cfg["train"]["gradient_accumulation_steps"]))
    completed_steps = 0
    micro_steps = 0
    model.train()
    optimizer.zero_grad(set_to_none=True)
    while completed_steps < max_steps:
        for batch in train_loader:
            batch.pop("is_volatile", None)
            outputs = model(**batch, use_cache=False)
            loss = outputs.loss
            cms_aux_loss = collect_cms_aux_loss(accelerator.unwrap_model(model))
            if cms_aux_loss is not None:
                loss = loss + float(adapter_cfg.get("aux_loss_weight", 0.05)) * cms_aux_loss
            unscaled_loss = loss.detach()
            accelerator.backward(loss / accumulation_steps)
            micro_steps += 1

            if micro_steps % accumulation_steps == 0:
                accelerator.clip_grad_norm_(model.parameters(), float(cfg["train"].get("max_grad_norm", 1.0)))
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad(set_to_none=True)

                completed_steps += 1
                if completed_steps % int(cfg["run"].get("log_every", 10)) == 0:
                    mean_loss = accelerator.gather(unscaled_loss).mean()
                    if accelerator.is_main_process:
                        print(
                            json.dumps(
                                {
                                    "step": completed_steps,
                                    "loss": float(mean_loss.cpu()),
                                    "lr": lr_scheduler.get_last_lr()[0],
                                }
                            )
                        )

                if completed_steps % int(cfg["run"].get("eval_every", 100)) == 0:
                    unwrapped = accelerator.unwrap_model(model)
                    device = accelerator.device
                    metrics = score_eval_suites(
                        unwrapped,
                        tokenizer,
                        eval_suites,
                        device=device,
                        max_length=int(cfg["data"]["max_length"]),
                    )
                    if bool(cfg["data"].get("include_guardrail_ppl", True)):
                        metrics.update(
                            score_perplexity_texts(
                                unwrapped,
                                tokenizer,
                                device=device,
                                max_length=int(cfg["data"]["max_length"]),
                            )
                        )
                    if accelerator.is_main_process:
                        print(json.dumps({"step": completed_steps, **metrics}))
                        with (output_dir / "metrics.jsonl").open("a") as handle:
                            handle.write(json.dumps({"step": completed_steps, **metrics}) + "\n")
                    model.train()

                if completed_steps >= max_steps:
                    break

    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        unwrapped = accelerator.unwrap_model(model)
        if bool(cfg["run"].get("save_full_model", False)):
            unwrapped.save_pretrained(output_dir / "final_model", safe_serialization=True)
            tokenizer.save_pretrained(output_dir / "final_model")
            print(f"saved {output_dir / 'final_model'}")
        else:
            torch.save(adapter_state_dict(unwrapped), output_dir / "adapter_state.pt")
            tokenizer.save_pretrained(output_dir / "tokenizer")
            print(f"saved {output_dir / 'adapter_state.pt'}")


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    main()
