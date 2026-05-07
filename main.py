import argparse
import gc
import json
import logging
import math
import os
import socket

import datasets
import numpy as np
import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from fed_utils import (
    FedHera,
    GeneralClient,
    client_selection,
    get_traffic_stats,
    modify_adapter,
    reset_traffic_stats,
    seed_torch,
)
from fed_utils.adaptive_peft import apply_lora_prefix_mask, load_weight_fedhera_if_exists
from fed_utils.model_aggregation import _load_fedhera_dense_global
from utils.prompter import Prompter

datasets.utils.logging.set_verbosity_error()
os.environ["WANDB_MODE"] = "disabled"


RESOURCE_RANKS = {
    "low": 4,
    "medium": 8,
    "high": 16,
}


def calculate_dynamic_budgets(layer_specs, num_clients, hetero_mode, seed=42):
    """
    Derive per-client bandwidth and compute budgets from the active LoRA layers.
    """
    del seed
    layer_specs = layer_specs or {}
    unit_params = 0
    for spec in layer_specs.values():
        d_in = spec.get("d_in")
        d_out = spec.get("d_out")
        if d_in is None or d_out is None:
            continue
        unit_params += d_in + d_out

    unit_comm_cost_bytes = unit_params * 2.0
    unit_comp_cost_ms = unit_params * 1.7e-04
    safety = 1.1

    targets = {
        "low": {"r_main": 4, "r_tot": 32},
        "medium": {"r_main": 8, "r_tot": 48},
        "high": {"r_main": 16, "r_tot": 64},
    }

    tier_bases = {}
    for tier, tgt in targets.items():
        b_down_mb = 0.0
        step_ms = 0.0
        if unit_params > 0:
            b_down_mb = (unit_comm_cost_bytes * tgt["r_tot"]) / (1024 * 1024)
            step_ms = unit_comp_cost_ms * tgt["r_main"]
            b_down_mb *= safety
            step_ms *= safety
        tier_bases[tier] = {
            "B_down_MB": float(b_down_mb),
            "VRAM_MB": 64000.0,
            "step_ms": float(step_ms),
        }

    distributions = {
        "setting_A": {"probs": [1 / 3, 1 / 3, 1 / 3], "tiers": ["low", "medium", "high"]},
        "setting_B": {"probs": [0.3, 0.5, 0.2], "tiers": ["low", "medium", "high"]},
    }

    client_budgets = {}
    dist = distributions.get(hetero_mode, distributions["setting_A"])
    probs = dist["probs"]
    tier_names = dist["tiers"]
    rng = np.random.default_rng(42)
    for client_id in range(num_clients):
        tier = str(rng.choice(tier_names, p=probs))
        base = tier_bases[tier]
        client_budgets[client_id] = {
            "tier": tier,
            "B_down_MB": float(base["B_down_MB"]),
            "VRAM_MB": float(base["VRAM_MB"]),
            "step_ms": float(base["step_ms"]),
        }
    return client_budgets


def extract_lora_layer_specs(model, target_modules):
    """
    Estimate LoRA layer shapes from the base model modules that receive adapters.
    """
    layer_specs = {}
    targets = tuple(target_modules or [])
    for name, module in model.named_modules():
        if not targets or not any(str(name).endswith(target) for target in targets):
            continue
        weight = getattr(module, "weight", None)
        if weight is None or not hasattr(weight, "shape") or len(weight.shape) < 2:
            continue

        if module.__class__.__name__ == "Conv1D":
            d_out, d_in = int(weight.shape[1]), int(weight.shape[0])
        else:
            d_out, d_in = int(weight.shape[0]), int(weight.shape[1])

        base_key = f"base_model.model.{name}.lora"
        layer_specs[base_key] = {"d_out": d_out, "d_in": d_in}
    return layer_specs


def parse_lora_target_modules(raw_value):
    try:
        parsed = json.loads(raw_value)
        if isinstance(parsed, list):
            return parsed
    except Exception:
        pass
    return [item.strip() for item in raw_value.split(",") if item.strip()]


def read_options():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--global_model",
        default="data_juicer",
        type=str,
        help="Path or HF id of the pretrained causal LM.",
    )
    parser.add_argument("--data_path", default="./data", type=str, help="Base path to federated data.")
    parser.add_argument("--cache_dir", default=None, type=str, help="Hugging Face dataset cache directory.")
    parser.add_argument("--output_dir", default=None, type=str, help="Base output directory.")
    parser.add_argument("--session_name", default="test", type=str, help="Experiment session name.")
    parser.add_argument("--seed", default=42, type=int, help="Random seed.")
    parser.add_argument(
        "--save_model",
        action="store_true",
        default=False,
        help="If set, save the latest dense FedHera aggregate as adapter_model.bin.",
    )
    parser.add_argument(
        "--deterministic",
        default=False,
        type=bool,
        help="Enable deterministic CUDA kernels at the cost of performance.",
    )
    parser.add_argument(
        "--device_map",
        default="cuda",
        type=str,
        help='Hugging Face device_map, e.g. "cuda", "auto", or "balanced".',
    )
    parser.add_argument(
        "--ablation",
        default=None,
        type=str,
        choices=[None, "uniform", "random"],
        help="FedHera allocation ablation. None means water-filling.",
    )

    parser.add_argument(
        "--aggregation",
        default="fedhera",
        type=str,
        choices=["fedhera"],
        help="Public export keeps only FedHera.",
    )
    parser.add_argument(
        "--hetero_mode",
        default="setting_B",
        type=str,
        choices=["setting_A", "setting_B"],
        help="Resource heterogeneity preset.",
    )
    parser.add_argument("--basis_update_every", default=1, type=int)
    parser.add_argument(
        "--baseline",
        default="fedavg",
        type=str,
        choices=["fedavg", "fedit"],
        help="Client-side optimizer style used inside FedHera.",
    )
    parser.add_argument(
        "--client_selection_frac",
        default=0.05,
        type=float,
        help="Fraction of clients participating in each round.",
    )
    parser.add_argument("--num_clients", default=1613, type=int, help="Total number of clients.")
    parser.add_argument(
        "--num_communication_rounds",
        default=50,
        type=int,
        help="Total number of communication rounds.",
    )
    parser.add_argument("--early_stop", default=True, type=bool, help="Enable early stopping.")
    parser.add_argument("--patience", default=10, type=int, help="Early stopping patience.")
    parser.add_argument(
        "--resume_epoch",
        default=None,
        type=int,
        help="Resume training from a previous communication round.",
    )
    parser.add_argument(
        "--dirichlet_alpha",
        default=None,
        type=float,
        help='If set, append "_noniid_<alpha>" to data_path.',
    )

    parser.add_argument("--local_batch_size", default=4, type=int, help="Local batch size.")
    parser.add_argument("--local_micro_batch_size", default=2, type=int, help="Local micro batch size.")
    parser.add_argument("--dataloader_num_workers", default=4, type=int, help="Data loader workers.")
    parser.add_argument("--local_num_epochs", default=5, type=int, help="Local epochs per client.")
    parser.add_argument("--local_learning_rate", default=1e-6, type=float, help="Local learning rate.")
    parser.add_argument("--cutoff_len", default=512, type=int, help="Tokenizer cutoff length.")
    parser.add_argument("--warmup", default=0, type=int, help="Warmup steps for local training.")
    parser.add_argument(
        "--lr_decay",
        default=True,
        type=bool,
        help="Halve local learning rate after round 15.",
    )
    parser.add_argument("--train_on_inputs", default=True, type=bool, help="Train on input tokens.")
    parser.add_argument("--group_by_length", default=False, type=bool, help="Enable length grouping.")
    parser.add_argument(
        "--prompt_template_name",
        default="alpaca",
        type=str,
        help="Prompt template name.",
    )

    parser.add_argument("--lora_r", default=8, type=int, help="Base LoRA rank.")
    parser.add_argument("--lora_alpha", default=16, type=int, help="LoRA alpha.")
    parser.add_argument("--lora_dropout", default=0.05, type=float, help="LoRA dropout.")
    parser.add_argument(
        "--lora_target_modules",
        default=None,
        type=parse_lora_target_modules,
        help="JSON list or comma-separated target modules. Omit for model defaults.",
    )
    parser.add_argument(
        "--use_atw",
        action="store_true",
        default=False,
        help="Enable Adaptive Tail Warm-up.",
    )
    parser.add_argument(
        "--atw_temperature",
        type=float,
        default=2.0,
        help="ATW temperature. Larger values mean slower warmup.",
    )
    parser.add_argument(
        "--calc_drift",
        action="store_true",
        default=False,
        help="Compute oracle drift against a high-rank adapter.",
    )
    parser.add_argument("--oracle_rank", default=512, type=int, help="Rank for the oracle drift baseline.")
    parser.add_argument(
        "--calc_cross_client_drift",
        action="store_true",
        default=False,
        help="Compute pairwise directional dispersion across clients.",
    )
    parser.add_argument(
        "--profile_system_costs",
        action="store_true",
        default=False,
        help="Profile client step latency, peak memory, and server SVD time.",
    )
    parser.add_argument(
        "--profile_warmup_steps",
        default=3,
        type=int,
        help="Warmup steps to ignore when profiling client latency.",
    )
    parser.add_argument(
        "--fedhera_coupled",
        action="store_true",
        default=False,
        help="Force coupled FedHera with r_tot == r_main.",
    )
    parser.add_argument(
        "--fedhera_server_agg",
        default="original",
        type=str,
        choices=["original", "unbiased"],
        help="Server aggregation rule used by FedHera.",
    )

    args = parser.parse_args()
    if isinstance(args.ablation, str) and args.ablation.lower() == "none":
        args.ablation = None
    return args


def model_and_tokenizer(global_model, device_map="cuda"):
    map_arg = device_map
    if isinstance(device_map, str) and device_map.lower() in ["cuda", "gpu", "single", "0"]:
        map_arg = {"": 0}

    use_cuda = torch.cuda.is_available()
    major, _ = torch.cuda.get_device_capability(0) if use_cuda else (0, 0)
    if use_cuda and major >= 8:
        model_dtype = torch.bfloat16
    elif use_cuda:
        model_dtype = torch.float16
    else:
        model_dtype = torch.float32

    model_load_kwargs = {
        "device_map": map_arg,
        "trust_remote_code": True,
        "torch_dtype": model_dtype,
    }
    if use_cuda:
        model_load_kwargs["attn_implementation"] = "flash_attention_2"

    try:
        model = AutoModelForCausalLM.from_pretrained(global_model, **model_load_kwargs)
    except Exception as exc:
        if model_load_kwargs.get("attn_implementation") == "flash_attention_2":
            logging.warning(
                "Failed to load %s with flash_attention_2 (%s). Falling back to default attention.",
                global_model,
                str(exc),
            )
            model_load_kwargs.pop("attn_implementation", None)
            model = AutoModelForCausalLM.from_pretrained(global_model, **model_load_kwargs)
        else:
            raise

    logging.info(
        "Loaded model %s with dtype=%s attn_implementation=%s",
        global_model,
        str(model_dtype).replace("torch.", ""),
        model_load_kwargs.get("attn_implementation", "default"),
    )

    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.config.use_cache = False
    model.enable_input_require_grads()

    tokenizer = AutoTokenizer.from_pretrained(
        global_model,
        trust_remote_code=True,
        use_fast=True,
        padding_side="left",
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 0

    return model, tokenizer


def resolve_lora_target_modules(model, user_target_modules=None):
    model_type = getattr(model.config, "model_type", "").lower()

    if user_target_modules:
        return user_target_modules
    if model_type in ["llama", "mistral", "gemma"]:
        return [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ]
    if model_type == "gpt2":
        return ["c_attn", "c_proj", "c_fc"]
    return ["q_proj", "v_proj"]


def compute_pairwise_directional_dispersion(client_dense_updates, eps=1e-12):
    """
    Compute cross-client drift in reconstructed dense update space.
    """
    client_ids = list(client_dense_updates.keys())
    if len(client_ids) < 2:
        return None

    all_layer_keys = sorted(
        {
            layer_key
            for updates in client_dense_updates.values()
            for layer_key in updates.keys()
        }
    )
    if not all_layer_keys:
        return None

    total_weight = 0.0
    weighted_dispersion_sum = 0.0
    total_pairs = 0
    zero_norm_pairs = 0
    layer_stats = []

    for layer_key in all_layer_keys:
        layer_updates = {}
        layer_norms = {}
        layer_numel = None

        for client_id in client_ids:
            tensor = client_dense_updates[client_id].get(layer_key)
            if tensor is not None:
                tensor_f = tensor.float()
                layer_numel = tensor_f.numel()
            else:
                tensor_f = None
            layer_updates[client_id] = tensor_f

        if layer_numel is None:
            continue

        for client_id in client_ids:
            tensor_f = layer_updates[client_id]
            if tensor_f is None:
                norm_val = 0.0
            else:
                norm_val = math.sqrt(float(torch.sum(tensor_f * tensor_f).item()))
            layer_norms[client_id] = norm_val

        pair_dispersion_sum = 0.0
        pair_count = 0
        layer_zero_norm_pairs = 0
        for idx_i in range(len(client_ids)):
            for idx_j in range(idx_i + 1, len(client_ids)):
                cid_i = client_ids[idx_i]
                cid_j = client_ids[idx_j]
                norm_i = layer_norms[cid_i]
                norm_j = layer_norms[cid_j]

                if norm_i <= eps and norm_j <= eps:
                    cosine = 1.0
                    layer_zero_norm_pairs += 1
                elif norm_i <= eps or norm_j <= eps:
                    cosine = 0.0
                    layer_zero_norm_pairs += 1
                else:
                    dot = float(torch.sum(layer_updates[cid_i] * layer_updates[cid_j]).item())
                    cosine = dot / max(norm_i * norm_j, eps)
                    cosine = max(-1.0, min(1.0, cosine))

                pair_dispersion_sum += 1.0 - cosine
                pair_count += 1

        if pair_count == 0:
            continue

        layer_dispersion = pair_dispersion_sum / float(pair_count)
        layer_weight = sum(layer_norms.values()) / float(len(client_ids))
        weighted_dispersion_sum += layer_weight * layer_dispersion
        total_weight += layer_weight
        total_pairs += pair_count
        zero_norm_pairs += layer_zero_norm_pairs
        layer_stats.append(
            {
                "layer": layer_key,
                "dispersion": layer_dispersion,
                "weight": layer_weight,
                "mean_update_norm": layer_weight,
                "param_count": layer_numel,
            }
        )

    if not layer_stats:
        return None

    layer_stats.sort(key=lambda item: item["weight"], reverse=True)
    if total_weight <= eps:
        dispersion = sum(item["dispersion"] for item in layer_stats) / float(len(layer_stats))
    else:
        dispersion = weighted_dispersion_sum / total_weight

    global_client_norms = {}
    for client_id in client_ids:
        sq_norm = 0.0
        for tensor in client_dense_updates[client_id].values():
            sq_norm += float(torch.sum(tensor.float() * tensor.float()).item())
        global_client_norms[client_id] = math.sqrt(max(sq_norm, 0.0))

    return {
        "dispersion": dispersion,
        "num_clients": len(client_ids),
        "num_pairs": total_pairs,
        "zero_norm_pairs": zero_norm_pairs,
        "num_layers": len(layer_stats),
        "mean_client_update_norm": sum(global_client_norms.values()) / float(len(global_client_norms)),
        "max_client_update_norm": max(global_client_norms.values()) if global_client_norms else 0.0,
        "min_client_update_norm": min(global_client_norms.values()) if global_client_norms else 0.0,
        "layer_stats_top": layer_stats[:3],
    }


def _summarize_system_costs(args, output_dir, profile_metrics):
    if not profile_metrics:
        return None

    def _mean(values):
        return float(np.mean(values)) if values else None

    def _std(values):
        return float(np.std(values)) if values else None

    client_latency = profile_metrics.get("client_step_latency_ms_values", [])
    peak_mem = profile_metrics.get("peak_gpu_mem_mb_values", [])
    server_svd = profile_metrics.get("server_svd_time_ms_per_round", [])

    summary = {
        "setting_name": str(args.session_name),
        "aggregation": str(args.aggregation),
        "global_model": str(args.global_model),
        "data_path": str(args.data_path),
        "avg_client_step_latency_ms": _mean(client_latency),
        "std_client_step_latency_ms": _std(client_latency),
        "avg_peak_gpu_mem_mb": _mean(peak_mem),
        "max_peak_gpu_mem_mb": float(max(peak_mem)) if peak_mem else None,
        "avg_server_svd_time_ms_per_round": _mean(server_svd),
        "std_server_svd_time_ms_per_round": _std(server_svd),
        "num_profiled_clients": len(client_latency),
        "num_profiled_rounds": len(profile_metrics.get("round_client_step_latency_ms", [])),
        "num_profiled_server_rounds": len(server_svd),
        "round_client_step_latency_ms": profile_metrics.get("round_client_step_latency_ms", []),
        "round_peak_gpu_mem_mb": profile_metrics.get("round_peak_gpu_mem_mb", []),
        "server_svd_time_ms_per_round": server_svd,
        "profile_warmup_steps": int(args.profile_warmup_steps),
    }

    summary_path = os.path.join(output_dir, "system_costs_summary.json")
    os.makedirs(output_dir, exist_ok=True)
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    logging.info("[SystemCostSummary] %s", summary)
    logging.info("[SystemCostSummary] Saved to %s", summary_path)
    return summary


def FL_training(model, tokenizer, prompter, data_path, output_dir, args, config):
    logging.info("The process of federated instruction-tuning has started.")
    reset_traffic_stats()
    previously_selected_clients_set = set()
    output_dir = os.path.join(output_dir, str(args.num_clients))

    system_cost_profile = None
    if args.profile_system_costs:
        system_cost_profile = {
            "client_step_latency_ms_values": [],
            "peak_gpu_mem_mb_values": [],
            "round_client_step_latency_ms": [],
            "round_peak_gpu_mem_mb": [],
            "server_svd_time_ms_per_round": [],
        }

    local_dataset_len_dict = {}
    best_rouge_l = 0.0
    best_round = None
    patience = args.patience
    current_count = 0
    start_epoch = int(args.resume_epoch or 0)

    dense_global_params = None
    if args.fedhera_server_agg == "unbiased":
        if start_epoch > 0:
            dense_global_params = _load_fedhera_dense_global(output_dir, start_epoch - 1)
            if dense_global_params is None:
                logging.warning(
                    "[FedHera] Missing dense cache for resume epoch %d. Reinitializing from zeros.",
                    start_epoch - 1,
                )
        if dense_global_params is None:
            dense_global_params = {}
            for key, spec in FL_training.layer_specs.items():
                d_out = int(spec.get("d_out", 0))
                d_in = int(spec.get("d_in", 0))
                dense_global_params[key] = torch.zeros((d_out, d_in), dtype=torch.float32)

    optim = "sgd" if args.baseline == "fedavg" else "adamw_torch"
    fedhera_default_lora_state = {
        key: value.detach().cpu().clone()
        for key, value in model.state_dict().items()
        if key.endswith("_A.local.weight") or key.endswith("_B.local.weight")
    }

    for epoch in tqdm(range(start_epoch, args.num_communication_rounds)):
        local_train_results = 0.0
        local_eval_results = 0.0
        local_eval_rouge_1 = 0.0
        local_eval_rouge_l = 0.0
        total_data_num = 0
        cross_client_dense_updates = {} if args.calc_cross_client_drift else None
        round_step_latency_values = [] if args.profile_system_costs else None
        round_peak_gpu_mem_values = [] if args.profile_system_costs else None

        logging.info("\nIn Epoch %s", epoch)
        logging.info("Conducting client selection")

        selected_clients_set = client_selection(
            args.num_clients,
            args.client_selection_frac,
            seed=args.seed,
            other_info=epoch,
        )
        if epoch == 15 and args.lr_decay:
            args.local_learning_rate = args.local_learning_rate / 2

        for client_index, client_id in enumerate(selected_clients_set):
            train_path = data_path + "/local_training_" + str(client_id) + ".json"
            train_data = load_dataset("json", data_files=train_path, cache_dir=args.cache_dir)
            local_dataset_len_dict[client_id] = len(train_data["train"])
            del train_data
            total_data_num += local_dataset_len_dict[client_id]

            pkg = None
            meta = None
            for previous_epoch in range(epoch - 1, -1, -1):
                pkg_tmp, meta_tmp = load_weight_fedhera_if_exists(output_dir, client_id, previous_epoch)
                if pkg_tmp is not None and meta_tmp is not None:
                    pkg, meta = pkg_tmp, meta_tmp
                    break

            hera_hooks = None
            if pkg is not None and meta is not None:
                per_layer_r_tot = {}
                for base_key, info in meta.items():
                    if info.get("skip", False):
                        continue
                    rank_tot = int(info.get("r_tot", 0))
                    if rank_tot <= 0:
                        continue
                    module_key = base_key.rsplit(".", 1)[0]
                    per_layer_r_tot[module_key] = rank_tot

                if per_layer_r_tot:
                    modify_adapter(
                        model,
                        "local",
                        modify_module_rank=per_layer_r_tot,
                        lora_alpha=args.lora_alpha,
                        lora_dropout=args.lora_dropout,
                        init_lora_weights=False,
                    )
                _ = model.load_state_dict(pkg, strict=False)
                per_layer_r_main = {
                    key: int(value.get("r_main", 0))
                    for key, value in meta.items()
                    if not value.get("skip", False)
                }
                hera_hooks = apply_lora_prefix_mask(model, per_layer_r_main)
            else:
                base_rank = int(args.lora_r)
                targets = (
                    config.target_modules
                    if hasattr(config, "target_modules")
                    else FL_training.lora_target_modules
                )
                if isinstance(targets, str):
                    targets = [targets]
                base_rank_map = {module_name: base_rank for module_name in targets}
                if base_rank_map:
                    modify_adapter(
                        model,
                        "local",
                        modify_module_rank=base_rank_map,
                        lora_alpha=args.lora_alpha,
                        lora_dropout=args.lora_dropout,
                        init_lora_weights=False,
                    )
                _ = model.load_state_dict(fedhera_default_lora_state, strict=False)

            client = GeneralClient(
                client_id,
                model,
                tokenizer,
                prompter,
                data_path,
                output_dir,
                cache_dir=args.cache_dir,
                hetero_lora=False,
                optim=optim,
                dataloader_num_workers=args.dataloader_num_workers,
            )

            logging.info("Preparing local dataset and trainer for Client_%s", client_id)
            client.preprare_local_dataset()

            local_eval_result = client.test(epoch, args.local_micro_batch_size)
            local_eval_results += float(local_eval_result["eval_loss"]) * local_dataset_len_dict[client_id]
            local_eval_rouge_1 += float(local_eval_result["eval_rouge1"]) * local_dataset_len_dict[client_id]
            local_eval_rouge_l += float(local_eval_result["eval_rougeL"]) * local_dataset_len_dict[client_id]

            logging.info("Initiating local training of Client_%s", client_id)
            client.build_local_trainer(
                tokenizer,
                args.local_micro_batch_size,
                args.local_batch_size // args.local_micro_batch_size,
                args.local_num_epochs,
                args.local_learning_rate,
                args.group_by_length,
                args.warmup,
                profile_system_costs=args.profile_system_costs,
                profile_warmup_steps=args.profile_warmup_steps,
            )
            client.initiate_local_training()

            logging.info("Local training starts")
            local_train_result = client.train()
            local_train_results += float(local_train_result["eval_loss"]) * local_dataset_len_dict[client_id]

            if args.profile_system_costs and client.latest_system_profile is not None:
                profile = client.latest_system_profile
                if profile.get("avg_step_latency_ms") is not None:
                    round_step_latency_values.append(float(profile["avg_step_latency_ms"]))
                    system_cost_profile["client_step_latency_ms_values"].append(
                        float(profile["avg_step_latency_ms"])
                    )
                if profile.get("peak_gpu_mem_mb") is not None:
                    round_peak_gpu_mem_values.append(float(profile["peak_gpu_mem_mb"]))
                    system_cost_profile["peak_gpu_mem_mb_values"].append(
                        float(profile["peak_gpu_mem_mb"])
                    )
                logging.info(
                    "[SystemProfile][epoch %d][client %s] avg_step_latency_ms=%s peak_gpu_mem_mb=%s measured_steps=%d warmup_steps=%d",
                    epoch,
                    str(client_id),
                    "None"
                    if profile.get("avg_step_latency_ms") is None
                    else f"{profile['avg_step_latency_ms']:.3f}",
                    "None"
                    if profile.get("peak_gpu_mem_mb") is None
                    else f"{profile['peak_gpu_mem_mb']:.3f}",
                    int(profile.get("num_measured_steps", 0)),
                    int(profile.get("warmup_steps", 0)),
                )

            if args.calc_cross_client_drift:
                try:
                    dense_update_dict, dense_update_norm = client.reconstruct_dense_residual_update(
                        lora_alpha=args.lora_alpha
                    )
                    cross_client_dense_updates[int(client_id)] = dense_update_dict
                    logging.info(
                        "[CrossClientDrift][epoch %d][client %s] dense_update_layers=%d update_norm=%.6f",
                        epoch,
                        str(client_id),
                        len(dense_update_dict),
                        dense_update_norm,
                    )
                except Exception as exc:
                    logging.error(
                        "[CrossClientDrift][epoch %d][client %s] Failed to reconstruct dense update: %s",
                        epoch,
                        str(client_id),
                        str(exc),
                    )

            if epoch > 0 and args.calc_drift and client_index == 0:
                try:
                    drift_val = client.compute_oracle_drift(
                        global_params=None,
                        oracle_r=args.oracle_rank,
                        lora_alpha=args.lora_alpha,
                    )
                    logging.info("[DriftMetrics] Epoch %d Algorithm fedhera Drift %s", epoch, drift_val)
                except RuntimeError as exc:
                    logging.error("OOM during Oracle training: %s", str(exc))
                    torch.cuda.empty_cache()

            logging.info("Terminating local training of Client_%s", client_id)
            model, local_dataset_len_dict, previously_selected_clients_set, _ = (
                client.terminate_local_training(epoch, local_dataset_len_dict, previously_selected_clients_set)
            )
            if hera_hooks is not None:
                for hook in hera_hooks:
                    try:
                        hook.remove()
                    except Exception:
                        pass
            del client

        if args.calc_cross_client_drift and cross_client_dense_updates:
            try:
                dispersion_stats = compute_pairwise_directional_dispersion(cross_client_dense_updates)
                if dispersion_stats is not None:
                    logging.info(
                        "[CrossClientDrift] Epoch %d Algorithm fedhera Dispersion %.6f NumClients %d NumPairs %d NumLayers %d ZeroNormPairs %d MeanUpdateNorm %.6f MinUpdateNorm %.6f MaxUpdateNorm %.6f",
                        epoch,
                        dispersion_stats["dispersion"],
                        dispersion_stats["num_clients"],
                        dispersion_stats["num_pairs"],
                        dispersion_stats["num_layers"],
                        dispersion_stats["zero_norm_pairs"],
                        dispersion_stats["mean_client_update_norm"],
                        dispersion_stats["min_client_update_norm"],
                        dispersion_stats["max_client_update_norm"],
                    )
                    logging.info(
                        "[CrossClientDrift][TopLayers] Epoch %d %s",
                        epoch,
                        [
                            {
                                "layer": item["layer"],
                                "dispersion": round(item["dispersion"], 6),
                                "weight": round(item["weight"], 6),
                            }
                            for item in dispersion_stats["layer_stats_top"]
                        ],
                    )
            except Exception as exc:
                logging.error("[CrossClientDrift][epoch %d] Failed to compute dispersion: %s", epoch, str(exc))

        if args.profile_system_costs:
            if round_step_latency_values:
                round_latency_mean = float(np.mean(round_step_latency_values))
                system_cost_profile["round_client_step_latency_ms"].append(round_latency_mean)
            if round_peak_gpu_mem_values:
                round_peak_mean = float(np.mean(round_peak_gpu_mem_values))
                system_cost_profile["round_peak_gpu_mem_mb"].append(round_peak_mean)
            logging.info(
                "[SystemProfile][epoch %d] round_avg_step_latency_ms=%s round_avg_peak_gpu_mem_mb=%s",
                epoch,
                "None" if not round_step_latency_values else f"{np.mean(round_step_latency_values):.3f}",
                "None" if not round_peak_gpu_mem_values else f"{np.mean(round_peak_gpu_mem_values):.3f}",
            )

        new_dense_params = FedHera(
            selected_clients_set,
            output_dir,
            local_dataset_len_dict,
            epoch,
            client_budgets=FL_training.client_budgets,
            layer_specs=FL_training.layer_specs,
            fixed_client_ranks=None,
            quant_scheme=("bfloat16", "nf4"),
            use_gpu_svd=True,
            basis_update_every=args.basis_update_every,
            ablation=args.ablation,
            lora_alpha=args.lora_alpha,
            server_agg=args.fedhera_server_agg,
            use_atw=args.use_atw,
            atw_temperature=args.atw_temperature,
            all_client_ids=list(range(args.num_clients)),
            prev_global_params=dense_global_params if args.fedhera_server_agg == "unbiased" else None,
            profile_metrics=system_cost_profile if args.profile_system_costs else None,
            force_coupled=args.fedhera_coupled,
        )

        if args.fedhera_server_agg == "unbiased":
            dense_global_params = new_dense_params
        if args.save_model:
            torch.save(new_dense_params, os.path.join(output_dir, "adapter_model.bin"))

        global_eval_rouge_l = local_eval_rouge_l / total_data_num

        if args.early_stop:
            if best_rouge_l < global_eval_rouge_l:
                best_rouge_l = global_eval_rouge_l
                best_round = epoch
                current_count = 0
            else:
                current_count += 1

            if current_count > patience:
                logging.info("Best round is %s with test_rouge_L %s", best_round, best_rouge_l)
                try:
                    stats = get_traffic_stats()
                    logging.info("[TrafficSummary] %s", stats)
                except Exception:
                    pass
                if args.profile_system_costs:
                    _summarize_system_costs(args, output_dir, system_cost_profile)
                return

        local_dataset_len_dict = {}
        gc.collect()

    try:
        stats = get_traffic_stats()
        logging.info("[TrafficSummary] %s", stats)
    except Exception:
        pass
    if args.profile_system_costs:
        _summarize_system_costs(args, output_dir, system_cost_profile)


def main():
    args = read_options()
    seed_torch(args.seed, deterministic=args.deterministic)

    if args.dirichlet_alpha is not None:
        raw_path = args.data_path.rstrip("/")
        args.data_path = f"{raw_path}_noniid_{args.dirichlet_alpha}"

    session_root = os.path.join(args.output_dir, args.session_name) if args.output_dir else args.session_name
    os.makedirs(session_root, exist_ok=True)

    log_dir = os.path.join(session_root, "logs")
    os.makedirs(log_dir, exist_ok=True)
    model_tag = os.path.basename(str(args.global_model)).replace("/", "_")
    dataset_tag = os.path.basename(os.path.normpath(args.data_path))
    log_name = f"fedhera_{args.hetero_mode}_{model_tag}_{dataset_tag}_r{args.lora_r}_{args.dirichlet_alpha}.log"
    log_path = os.path.join(log_dir, log_name)
    logging.basicConfig(filename=log_path, level=logging.INFO, format="%(message)s")
    logging.info("Logging to %s", log_path)
    logging.info("Initial training parameters %s", args)
    logging.info(str(socket.gethostbyname(socket.gethostname())))
    print(args)

    data_path = os.path.join(args.data_path, str(args.num_clients))
    output_dir = os.path.join(session_root, args.aggregation)

    model, tokenizer = model_and_tokenizer(global_model=args.global_model, device_map=args.device_map)
    prompter = Prompter(args.prompt_template_name)

    lora_target_modules = resolve_lora_target_modules(model, user_target_modules=args.lora_target_modules)
    layer_specs = extract_lora_layer_specs(model, lora_target_modules)
    client_budgets = calculate_dynamic_budgets(
        layer_specs,
        args.num_clients,
        args.hetero_mode,
        seed=args.seed,
    )
    FL_training.layer_specs = layer_specs
    FL_training.client_budgets = client_budgets
    FL_training.lora_target_modules = lora_target_modules

    config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=lora_target_modules,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, config, adapter_name="local")

    FL_training(model, tokenizer, prompter, data_path, output_dir, args, config=config)


if __name__ == "__main__":
    main()
