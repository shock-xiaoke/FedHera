from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
)
from fed_utils import FedAvg, client_selection, seed_torch, GeneralClient, FlexLoRA, HetLoRA, \
    load_weight_local, load_weight_hetlora, distribute_weight_fast, modify_adapter, FedHera, FedHeLLo, FLoRA, FedHL
from fed_utils.model_aggregation import reset_traffic_stats, get_traffic_stats, TRAFFIC_STATS

import datasets
from datasets import load_dataset
from utils.prompter import Prompter
import socket

datasets.utils.logging.set_verbosity_error()

import numpy as np
import random
import os
import torch
import logging
import argparse
import math
os.environ["WANDB_MODE"]="disabled"

import json

# Canonical LoRA ranks associated with client resource levels.
# These are used both for heterogeneous FlexLoRA ranks and, via
# Fed-Hera, for computing client resource budgets.
RESOURCE_RANKS = {
    "low": 4,
    "medium": 8,
    "high": 16,
}



def calculate_dynamic_budgets(layer_specs, num_clients, hetero_mode, seed=42):
    """
    Dynamically derive per-client bandwidth/compute budgets that yield
    fixed target ranks regardless of how many LoRA layers are active.
    """
    layer_specs = layer_specs or {}
    unit_params = 0
    for spec in layer_specs.values():
        d_in = spec.get("d_in")
        d_out = spec.get("d_out")
        if d_in is None or d_out is None:
            continue
        unit_params += (d_in + d_out)

    unit_comm_cost_bytes = unit_params * 2.0  # BF16/FP16 bytes per param
    unit_comp_cost_ms = unit_params * 1.7e-04  # matched to model_aggregation.py
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
        "setting_A": {"probs": [1/3, 1/3, 1/3], "tiers": ["low", "medium", "high"]},
        "setting_B": {"probs": [0.3, 0.5, 0.2], "tiers": ["low", "medium", "high"]},
    }

    client_budgets = {}
    dist = distributions.get(hetero_mode, distributions["setting_A"])
    probs = dist["probs"]
    tier_names = dist["tiers"]
    rng = np.random.default_rng(seed)
    for i in range(num_clients):
        tier = str(rng.choice(tier_names, p=probs))
        base = tier_bases[tier]
        client_budgets[i] = {
            "tier": tier,
            "B_down_MB": float(base["B_down_MB"]),
            "VRAM_MB": float(base["VRAM_MB"]),
            "step_ms": float(base["step_ms"]),
        }
    return client_budgets

def extract_lora_layer_specs(model, target_modules):
    """
    Estimate LoRA layer (d_out, d_in) pairs from base model modules that will
    receive adapters.
    """
    layer_specs = {}
    targets = tuple(target_modules or [])
    for name, module in model.named_modules():
        if not targets or not any(str(name).endswith(t) for t in targets):
            continue
        weight = getattr(module, "weight", None)
        if weight is None or not hasattr(weight, "shape") or len(weight.shape) < 2:
            continue
        
        if module.__class__.__name__ == 'Conv1D':
            d_out, d_in = int(weight.shape[1]), int(weight.shape[0])
        else:
            d_out, d_in = int(weight.shape[0]), int(weight.shape[1])
            
        base_key = f"base_model.model.{name}.lora"  # Mirror PEFT naming consumed downstream.
        layer_specs[base_key] = {"d_out": d_out, "d_in": d_in}
    return layer_specs

def calculate_unified_rank_from_budget(client_budgets, layer_specs, max_rank=64):
    """
    Map resource tiers directly to fixed LoRA ranks (low/medium/high -> 4/8/16).
    """
    tier_to_rank = {
        "low": RESOURCE_RANKS["low"],
        "medium": RESOURCE_RANKS["medium"],
        "high": RESOURCE_RANKS["high"],
    }
    rank_map = {}
    for client_id, budget in client_budgets.items():
        tier = str(budget.get("tier", "")).lower()
        r_comm = tier_to_rank.get(tier, RESOURCE_RANKS["medium"])
        r_comp = r_comm
        final_r = max(1, min(max_rank, r_comm, r_comp))
        rank_map[client_id] = final_r
    return rank_map


def calculate_active_layers_from_budget(client_budgets, layer_specs, lora_rank):
    """
    Estimate how many LoRA layers each client can actively train under
    Fed-HeLLo based on bandwidth and step-time budgets.
    Uses calibrated per-parameter compute/communication costs to gate active layers.
    """
    layer_specs = layer_specs or {}
    total_layers = len(layer_specs)
    if total_layers == 0:
        return {}, 0.0

    params_per_layer = []
    for _, spec in layer_specs.items():
        d_in = spec.get("d_in")
        d_out = spec.get("d_out")
        if d_in is None or d_out is None:
            continue
        params_per_layer.append((d_in + d_out) * lora_rank)

    avg_params_per_layer = float(np.mean(params_per_layer)) if params_per_layer else 0.0
    comm_cost_per_layer = avg_params_per_layer * 2.0  # BF16 bytes per param
    comp_cost_per_layer = avg_params_per_layer * 1.7e-04  # ms per layer scaled by params
    MB = 1024 * 1024

    num_active_layers = {}
    for client_id, budget in client_budgets.items():
        b_down_bytes = float(budget.get("B_down_MB", 0.0)) * MB
        step_ms = float(budget.get("step_ms", 0.0))
        l_comm = (b_down_bytes / comm_cost_per_layer) if comm_cost_per_layer > 0 else total_layers
        l_comp = (step_ms / comp_cost_per_layer) if comp_cost_per_layer > 0 else total_layers
        count = int(min(l_comm, l_comp))
        count = max(1, min(total_layers, count))
        num_active_layers[int(client_id)] = count

    return num_active_layers, avg_params_per_layer

def parse_lora_target_modules(s):
    # Accept JSON list or comma-separated string
    try:
        v = json.loads(s)
        if isinstance(v, list):
            return v
    except Exception:
        pass
    return [x.strip() for x in s.split(",") if x.strip()]



def read_options():
    parser = argparse.ArgumentParser()

    parser.add_argument('--global_model', default='data_juicer', type=str, help='ifle path to the LLaMA model')
    parser.add_argument('--data_path', default='./data', type=str,
                        help='file path to data')
    parser.add_argument('--cache_dir', default=None, type=str,
                        help='file path for caching data')
    parser.add_argument('--output_dir', default=None, type=str,
                        help='output directory to store model and experiment result')
    parser.add_argument('--session_name', default='test', type=str,
                        help='name for your experiment')
    parser.add_argument('--seed', default=42, type=int,
                        help='random seed')
    parser.add_argument('--save_model', action='store_true', default=False,
                        help='If set, save aggregated adapter_model.bin; otherwise only logs are kept')
    parser.add_argument('--deterministic', default=False, type=bool,
                        help='Enable deterministic CUDA kernels (slower, disables TF32/cuDNN benchmark)')
    parser.add_argument('--device_map', default='cuda', type=str,
                        help='HuggingFace device_map, e.g., "cuda", "auto", or "balanced"')
    parser.add_argument('--ablation', default=None, type=str,
                        choices=[None, 'uniform', 'random'],
                        help='FedHera ablation: None -> water-filling, uniform -> equal ranks, random -> random ranks within budgets')

    ## FL parameters
    parser.add_argument('--aggregation', default='homo', type=str,
                        help='aggregation method',
                        choices=['homo', 'flexlora', 'hetlora', 'fedhera', 'fedhello', 'flora', 'fedhl'])
    parser.add_argument('--hetero_mode', default='setting_B', type=str,
                        choices=['setting_A', 'setting_B'],
                        help='resource heterogeneity preset for Fed-Hera/FlexLoRA')
    parser.add_argument('--basis_update_every', default=1, type=int)
    parser.add_argument('--baseline', default='fedavg', type=str,
                        help='type of FL baseline to choose', choices=['fedavg', 'fedit'])
    parser.add_argument('--client_selection_frac', default=0.05, type=float,
                        help='ratio of how many clients participate in each round')
    parser.add_argument('--num_clients', default=1613, type=int,
                        help='total number of clients')
    parser.add_argument('--num_communication_rounds', default=50, type=int,
                        help='total number of communication rounds')
    parser.add_argument('--R_1', default=5, type=int,
                        help='Parameter for SLoRA. Total number of rounds for stage 1 sparse finetuning.')
    parser.add_argument('--early_stop', default=True, type=bool,
                        help='Early stop for FL training. If True, will apply early stop.')
    parser.add_argument('--patience', default=10, type=int,
                        help='Early stop patience.')
    parser.add_argument('--resume_epoch', default=None, type=int,
                        help='continue training from an existing experiment, specifying which comm round to resume')
    parser.add_argument('--dirichlet_alpha', default=None, type=float,
                        help='Optional Dirichlet alpha for non-IID data selection. If set (e.g., 0.5), it appends "_noniid_0.5" to data_path.')
    
    ## Local training parameters
    parser.add_argument('--local_batch_size', default=4, type=int,
                        help='local_batch_size')
    parser.add_argument('--local_micro_batch_size', default=2, type=int,
                        help='local_micro_batch_size')
    parser.add_argument('--dataloader_num_workers', default=4, type=int,
                        help='Number of worker processes for data loading')
    parser.add_argument('--local_num_epochs', default=5, type=int,
                        help='local epochs for local client training')
    parser.add_argument('--local_learning_rate', default=1e-6, type=float,
                        help='local training rate for local client training')
    parser.add_argument('--cutoff_len', default=512, type=int,
                        help='cut off len for tokenizing text')
    parser.add_argument('--warmup', default=0, type=int,
                        help='warm up steps for local training')
    parser.add_argument('--lr_decay', default=True, type=bool,
                        help='Learning rate decay. If true, will divide learning rate by 2 after 15-th comm round')
    parser.add_argument('--train_on_inputs', default=True, type=bool,
                        help='Whether training on input text')
    parser.add_argument('--group_by_length', default=False, type=bool,
                        help='')
    parser.add_argument('--prompt_template_name', default='alpaca', type=str,
                        help='template to generate prompt')

    ## LoRA Parameters
    parser.add_argument('--lora_r', default=8, type=int,
                        help='LoRA rank')
    parser.add_argument('--lora_alpha', default=16, type=int,
                        help='LoRA alpha')
    parser.add_argument('--lora_dropout', default=0.05, type=float,
                        help='LoRA dropout')
    parser.add_argument('--lora_target_modules',
                        default=None,
                        type=parse_lora_target_modules,
                        help='lora_target_modules (JSON list or comma-separated); omit for model-specific defaults',
                        )
    parser.add_argument('--use_atw', action='store_true', default=False,
                        help='Enable Adaptive Tail Warm-up (ATW) for FedHera. '
                             'If False, lambda is fixed to 1.0 (Static Tail).')
    parser.add_argument('--atw_temperature', type=float, default=2.0, 
                        help='Temperature for FedHera ATW (default: 2.0). Higher value = Slower lambda warmup.')
    parser.add_argument('--calc_drift', action='store_true', default=False,
                        help='Whether to calculate drift against a high-rank Oracle (Very slow!).')
    parser.add_argument('--oracle_rank', default=512, type=int, help='Rank for the Oracle baseline.')
    parser.add_argument('--calc_cross_client_drift', action='store_true', default=False,
                        help='Whether to calculate cross-client drift via pairwise directional dispersion.')
    parser.add_argument('--profile_system_costs', action='store_true', default=False,
                        help='Enable lightweight system-cost profiling for local training and server SVD.')
    parser.add_argument('--profile_warmup_steps', default=3, type=int,
                        help='Number of local train steps to ignore as warmup when profiling step latency.')
    parser.add_argument('--fedhera_coupled', action='store_true', default=False,
                        help='Force coupled FedHera setting with r_tot = r_train (i.e., r_tot = r_main).')
    parser.add_argument('--fedhera_server_agg',
                        default='original',
                        type=str,
                        choices=['original', 'unbiased'],
                        help=("FedHera server-side aggregation. "
                            "original: W_{t+1}=Σ p_i W_i. "
                            "unbiased: FedHL-style W_{t+1}=W_t+Σ p_i (W_i^{t+1}-W_t^{r_i}), "
                            "where W_t^{r_i} is the last-round server_push for each client."))


    args = parser.parse_args()
    if isinstance(args.ablation, str) and args.ablation.lower() == "none":
        args.ablation = None
    return args


# [修改] main.py 中的 model_and_tokenizer 函数

def model_and_tokenizer(global_model, device_map='cuda'):
    # 处理 device_map 参数
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
        model = AutoModelForCausalLM.from_pretrained(
            global_model,
            **model_load_kwargs,
        )
    except Exception as e:
        if model_load_kwargs.get("attn_implementation") == "flash_attention_2":
            logging.warning(
                "Failed to load %s with flash_attention_2 (%s). Falling back to default attention.",
                global_model,
                str(e),
            )
            model_load_kwargs.pop("attn_implementation", None)
            model = AutoModelForCausalLM.from_pretrained(
                global_model,
                **model_load_kwargs,
            )
        else:
            raise

    logging.info(
        "Loaded model %s with dtype=%s attn_implementation=%s",
        global_model,
        str(model_dtype).replace("torch.", ""),
        model_load_kwargs.get("attn_implementation", "default"),
    )
    
    # 启用梯度检查点时的参数更新
    # use_reentrant=False 是新版推荐设置，避免警告和潜在显存问题
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.config.use_cache = False
    model.enable_input_require_grads()

    # 2. 加载 Tokenizer
    # Llama-3 的 tokenizer 不需要 use_fast=False 强制降级，新版 fast tokenizer 已经很稳定
    tokenizer = AutoTokenizer.from_pretrained(
        global_model,
        trust_remote_code=True,
        use_fast=True, # [建议] 改为 True，除非显存非常紧张
        padding_side="left" # Decoder-only 模型通常左填充
    )
    
    # 3. 修复 Pad Token (关键)
    # Llama-3.2 的 tokenizer 通常没有默认 pad_token_id，或者 pad_token_id 是一个特殊保留位
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is not None:
            tokenizer.pad_token_id = tokenizer.eos_token_id
            print(f"Warning: pad_token_id was None, set to eos_token_id: {tokenizer.eos_token_id}")
        else:
            # 万一连 EOS 都没有（极少见），设为 0
            tokenizer.pad_token_id = 0
            
    return model, tokenizer


def resolve_lora_targets_and_config_types(model, user_target_modules=None):
    """
    Choose sensible default LoRA target modules and heterogeneity configs
    based on the underlying model architecture.
    """
    model_type = getattr(model.config, "model_type", "").lower()

    # If the user explicitly provided a list, always respect it verbatim.
    if user_target_modules:
        target_modules = user_target_modules
    else:
        if model_type in ["llama", "mistral", "gemma"]:
            # LLaMA/Mistral/Gemma use the same projection names.
            target_modules = [
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ]
        elif model_type in ["gpt2"]:
            # GPT-2 blocks: attn.c_attn / attn.c_proj / mlp.c_fc / mlp.c_proj
            target_modules = ["c_attn", "c_proj", "c_fc"]
        else:
            # Fallback to the original default.
            target_modules = ['q_proj', 'v_proj']

    # Heterogeneous PEFT type presets.
    # Tie the three tiers directly to the canonical
    # resource levels so that FlexLoRA always picks
    # ranks from {4, 8, 16}.
    small_r = RESOURCE_RANKS["low"]
    medium_r = RESOURCE_RANKS["medium"]
    large_r = RESOURCE_RANKS["high"]

    if model_type in ["llama", "mistral", "gemma"] and set(
        ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "down_proj", "up_proj"]
    ).issubset(set(target_modules)):
        config_types = {
            'Type_0': {
                'q_proj': small_r, 'v_proj': small_r, 'k_proj': small_r, 'o_proj': small_r,
                'gate_proj': small_r, 'down_proj': small_r, 'up_proj': small_r,
            },
            'Type_1': {
                'q_proj': large_r, 'v_proj': large_r, 'k_proj': large_r, 'o_proj': large_r,
                'gate_proj': large_r, 'down_proj': large_r, 'up_proj': large_r,
            },
            'Type_2': {
                'q_proj': medium_r, 'v_proj': medium_r, 'k_proj': medium_r, 'o_proj': medium_r,
                'gate_proj': medium_r, 'down_proj': medium_r, 'up_proj': medium_r,
            },
            'Type_3': {
                'q_proj': medium_r, 'v_proj': medium_r, 'k_proj': medium_r, 'o_proj': medium_r,
                'gate_proj': medium_r, 'down_proj': medium_r, 'up_proj': medium_r,
            },
        }
    else:
        # Generic patterns for other architectures (including GPT-2).
        config_types = {
            'Type_0': {m: small_r for m in target_modules},
            'Type_1': {m: large_r for m in target_modules},
            'Type_2': {m: medium_r for m in target_modules},
            'Type_3': {m: medium_r for m in target_modules},
        }

    return target_modules, config_types


def _resource_probabilities(mode: str):
    """
    Map hetero mode to (low, medium, high) probabilities for FlexLoRA-style rank sampling.
    """
    if mode == 'setting_A':
        return [1/3, 1/3, 1/3]  # uniform
    if mode == 'setting_B':
        return [0.3, 0.5, 0.2]  # 30% low, 50% mid, 20% high
    return [1/3, 1/3, 1/3]



def get_peft(config_types, num_clients, strategy=None, hetero_mode="setting_B", seed=42, fixed_ranks=None):
    """
    Get each client's unique LoRA configuration based on the aggregation strategy.
    """
    if strategy in ['homo', 'fedhera', 'fedhello']:
        return {'alpha': 16, 'lora_dropout': 0.05}
    module_template = next(iter(config_types.values()), {})
    base_modules = list(module_template.keys())
    if fixed_ranks is not None:
        config_local = {'alpha': 16, 'lora_dropout': 0.05}
        for i in range(num_clients):
            rank = int(fixed_ranks.get(i, RESOURCE_RANKS["medium"]))
            config_local['Client_' + str(i)] = {m: rank for m in base_modules}
        return config_local
    rng = np.random.default_rng(seed)
    probs = _resource_probabilities(hetero_mode or "setting_B")
    tier_to_type = {"low": "Type_0", "medium": "Type_2", "high": "Type_1"}
    tiers = ["low", "medium", "high"]

    config_local = {'alpha': 16, 'lora_dropout': 0.05}
    for i in range(num_clients):
        tier = rng.choice(tiers, p=probs)
        type_key = tier_to_type[tier]
        config_local['Client_' + str(i)] = config_types[type_key]
    return config_local


def local_client_load_weight(args, model, epoch, global_params=None):
    """
    Load local client weight for non-FedHera strategies.
    """
    if args.aggregation in ['homo', 'fedhello']:
        _ = model.load_state_dict(global_params, strict=False)
    elif args.aggregation == 'hetlora':
        local_weight = load_weight_hetlora(global_params, model)
        _ = model.load_state_dict(local_weight, strict=False)
    else:
        local_weight = load_weight_local(global_params, model)
        _ = model.load_state_dict(local_weight, strict=False)




def local_client_modify_layer(args, epoch, config_local, model, client_id):
    """
    Modify local client's LoRA layers based on local config.
    """
    if args.aggregation in ['fedhera', 'fedhello']:
        return
    if args.aggregation != 'homo':
        local_lora_config = config_local['Client_' + str(client_id)]
        modify_adapter(model, 'local', modify_module_rank=local_lora_config,
                       lora_alpha=config_local['alpha'], lora_dropout=config_local['lora_dropout'],
                       init_lora_weights=True)


def resume(args, data_path, output_dir, config_local):
    """
    Resume experiment from an existing study.
    """
    selected_clients_set = client_selection(args.num_clients, args.client_selection_frac,
                                            seed=args.seed, other_info=args.resume_epoch-1)
    local_dataset_len_dict = {}
    for client_id in tqdm(selected_clients_set):
        train_path = data_path + '/local_training_' + str(client_id) + '.json'
        train_data = load_dataset("json", data_files=train_path, cache_dir=args.cache_dir)
        local_dataset_len_dict[client_id] = len(train_data['train'])
    if args.aggregation == 'homo':
        global_params = FedAvg(
            selected_clients_set,
            output_dir,
            local_dataset_len_dict,
            args.resume_epoch-1,
            client_budgets=FL_training.client_budgets,
            layer_specs=FL_training.layer_specs,
        )
    elif args.aggregation == 'flexlora':
        global_params = FlexLoRA(
            selected_clients_set,
            output_dir,
            local_dataset_len_dict,
            args.resume_epoch-1,
            client_budgets=FL_training.client_budgets,
            layer_specs=FL_training.layer_specs,
        )
        global_params = distribute_weight_fast(global_params, config_local)
    elif args.aggregation == 'hetlora':
        global_params = HetLoRA(
            selected_clients_set,
            output_dir,
            local_dataset_len_dict,
            args.resume_epoch - 1,
            client_rank_map=config_local,
            target_global_rank=getattr(FL_training, "hetlora_global_rank", args.lora_r),
            prev_global_params=None,
            client_budgets=FL_training.client_budgets,
            layer_specs=FL_training.layer_specs,
        )
    elif args.aggregation == 'flora':
            # 1. Aggregate via Stacking (FLoRA specific)
            global_params = FLoRA(selected_clients_set,
                                  output_dir,
                                  local_dataset_len_dict,
                                  epoch,
                                  client_budgets=FL_training.client_budgets,
                                  layer_specs=FL_training.layer_specs)
            
            if args.save_model:
                torch.save(global_params, os.path.join(output_dir, "adapter_model.bin"))
                
            # 2. Distribute via SVD (Reuse FlexLoRA's distribution logic)
            global_params = distribute_weight_fast(global_params, config_local)
    else:
        global_params = None
    return global_params


def get_density(args, config_local, client_id, config_types):
    """
    get sparsity for slora sparse finetuning stage 1
    """
    if args.aggregation == 'homo':
        density = 0.0012
    else:
        if config_local['Client_' + str(client_id)] == config_types['Type_0']:
            density = 0.0012
        if config_local['Client_' + str(client_id)] == config_types['Type_1']:
            density = 0.1222
        if config_local['Client_' + str(client_id)] == config_types['Type_2']:
            density = 0.0822
        if config_local['Client_' + str(client_id)] == config_types['Type_3']:
            density = 0.0246
    return density


def compute_pairwise_directional_dispersion(client_dense_updates, eps=1e-12):
    """
    Compute cross-client drift in reconstructed dense update space.

    Instead of flattening all layers at once, compute pairwise directional
    dispersion per LoRA layer first, then aggregate layer scores using the
    layer's mean update Frobenius norm as weight. This reduces domination by
    a few large layers while still emphasizing layers with meaningful updates.
    """
    client_ids = list(client_dense_updates.keys())
    if len(client_ids) < 2:
        return None

    all_layer_keys = sorted({
        layer_key
        for updates in client_dense_updates.values()
        for layer_key in updates.keys()
    })
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
        layer_sq_norm_sum = 0.0
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
            layer_sq_norm_sum += norm_val * norm_val

        pair_dispersion_sum = 0.0
        pair_count = 0
        layer_zero_norm_pairs = 0
        for i in range(len(client_ids)):
            for j in range(i + 1, len(client_ids)):
                cid_i = client_ids[i]
                cid_j = client_ids[j]
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

                pair_dispersion_sum += (1.0 - cosine)
                pair_count += 1

        if pair_count == 0:
            continue

        layer_dispersion = pair_dispersion_sum / float(pair_count)
        layer_weight = sum(layer_norms.values()) / float(len(client_ids))
        weighted_dispersion_sum += layer_weight * layer_dispersion
        total_weight += layer_weight
        total_pairs += pair_count
        zero_norm_pairs += layer_zero_norm_pairs
        layer_stats.append({
            "layer": layer_key,
            "dispersion": layer_dispersion,
            "weight": layer_weight,
            "mean_update_norm": layer_weight,
            "param_count": layer_numel,
        })

    if not layer_stats:
        return None

    layer_stats.sort(key=lambda x: x["weight"], reverse=True)
    if total_weight <= eps:
        dispersion = sum(x["dispersion"] for x in layer_stats) / float(len(layer_stats))
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

    def _mean(vals):
        return float(np.mean(vals)) if vals else None

    def _std(vals):
        return float(np.std(vals)) if vals else None

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


# training for FL setting
def FL_training(model, tokenizer, prompter, data_path, output_dir, args, config_local, config=None, config_types=None):
    logging.info("The process of federated instruction-tuning has started..")
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

    local_dataset_len_dict = dict()
    best_rouge_L = 0
    patience = args.patience
    current_count = 0
    dense_global_params = None
    if args.resume_epoch:
        global_params = resume(args, data_path, output_dir, config_local)
        start_epoch = args.resume_epoch
    else:
        start_epoch = 0
        global_params = None
    if (args.aggregation == 'fedhl' or (args.aggregation == 'fedhera' and args.fedhera_server_agg == 'unbiased')) and dense_global_params is None:
        logging.info(f"[{args.aggregation}] Initializing dense global LoRA-update W0 as ZEROS (delta adapter).")
        dense_global_params = {}

        for key, spec in FL_training.layer_specs.items():
            d_out = int(spec.get("d_out", 0))
            d_in  = int(spec.get("d_in", 0))
            dense_global_params[key] = torch.zeros((d_out, d_in), dtype=torch.float32)
    if args.aggregation == 'fedhl' and dense_global_params is None:
        logging.info("[FedHL] Initializing dense global LoRA-update W0 as ZEROS (delta adapter).")
        dense_global_params = {}

        for key, spec in FL_training.layer_specs.items():
            d_out = int(spec.get("d_out", 0))
            d_in  = int(spec.get("d_in", 0))
            assert d_out > 0 and d_in > 0, f"Bad layer spec for {key}: {spec}"
            dense_global_params[key] = torch.zeros((d_out, d_in), dtype=torch.float32)

        logging.info("[FedHL] Performing initial SVD distribution for Round 0.")
        global_params = distribute_weight_fast(dense_global_params, config_local)

    optim = 'sgd' if args.baseline == 'fedavg' else 'adamw_torch'
    fedhera_default_lora_state = None
    if args.aggregation == 'fedhera':
        # Snapshot initial LoRA weights as a global default for cases where no server push exists yet
        # (e.g., epoch 0 or resumed runs with missing push folders).
        fedhera_default_lora_state = {
            k: v.detach().cpu().clone()
            for k, v in model.state_dict().items()
            if k.endswith('_A.local.weight') or k.endswith('_B.local.weight')
        }

    fedhello_layer_keys = sorted(FL_training.layer_specs.keys()) if args.aggregation == 'fedhello' else []
    fedhello_active_counts = FL_training.fedhello_active_layer_counts if args.aggregation == 'fedhello' else {}
    if fedhello_active_counts is None:
        fedhello_active_counts = {}
    for epoch in tqdm(range(start_epoch, args.num_communication_rounds)):
        local_train_results = 0
        local_eval_results = 0
        local_eval_rouge_1 = 0
        local_eval_rouge_L = 0
        total_data_num = 0
        cross_client_dense_updates = {} if args.calc_cross_client_drift else None
        round_step_latency_values = [] if args.profile_system_costs else None
        round_peak_gpu_mem_values = [] if args.profile_system_costs else None
        logging.info("\In Epoch " + str(epoch))
        logging.info("\nConducting the client selection")

        selected_clients_set = client_selection(args.num_clients, args.client_selection_frac,
                                                seed=args.seed, other_info=epoch)
        if epoch == 15 and args.lr_decay:
            args.local_learning_rate = args.local_learning_rate / 2

        fedhello_masks = None
        if args.aggregation == 'fedhello':
            fedhello_masks = {}
            rng = np.random.default_rng(args.seed + epoch)
            total_layers = len(fedhello_layer_keys)
            for client_id in selected_clients_set:
                active_count = int(fedhello_active_counts.get(int(client_id), 1))
                if total_layers > 0:
                    active_count = max(1, min(total_layers, active_count))
                    chosen = rng.choice(fedhello_layer_keys, size=active_count, replace=False)
                    active_layers = [str(x) for x in chosen]
                else:
                    active_layers = []
                fedhello_masks[client_id] = active_layers

        for k, client_id in enumerate(selected_clients_set):
            train_path = data_path + '/local_training_' + str(client_id) + '.json'
            train_data = load_dataset("json", data_files=train_path, cache_dir=args.cache_dir)
            local_dataset_len_dict[client_id] = len(train_data['train'])
            del train_data
            total_data_num += local_dataset_len_dict[client_id]

            local_client_modify_layer(args, epoch, config_local, model, client_id)
            hera_hooks = None
            if args.aggregation == 'fedhera':
                from fed_utils.adaptive_peft import load_weight_fedhera_if_exists, apply_lora_prefix_mask, modify_adapter
                # Find the latest available server push for this client
                pkg, meta = None, None
                for e in range(epoch - 1, -1, -1):
                    pkg_tmp, meta_tmp = load_weight_fedhera_if_exists(output_dir, client_id, e)
                    if pkg_tmp is not None and meta_tmp is not None:
                        pkg, meta = pkg_tmp, meta_tmp
                        break

                if pkg is not None and meta is not None:
                    # 有 Push：加载 Push
                    per_layer_r_tot = {}
                    for base_key, info in meta.items():
                        if info.get("skip", False): continue
                        rt = int(info.get("r_tot", 0))
                        if rt <= 0: continue
                        module_key = base_key.rsplit(".", 1)[0]
                        per_layer_r_tot[module_key] = rt

                    if per_layer_r_tot:
                        modify_adapter(
                            model, 'local', modify_module_rank=per_layer_r_tot,
                            lora_alpha=16, lora_dropout=0.05, init_lora_weights=False,
                        )
                    _ = model.load_state_dict(pkg, strict=False)

                    per_layer_r_main = {k: int(v.get("r_main", 0)) for k, v in meta.items() if not v.get("skip", False)}
                    hera_hooks = apply_lora_prefix_mask(model, per_layer_r_main)
                else:
                    # 无 Push：重置为默认 (修复了这里的 NoneType 错误)
                    # 使用 config.target_modules 作为更可靠的来源
                    base_rank = int(getattr(args, 'lora_r', 8))
                    
                    # [修复] 优先使用 config 中的 target_modules，因为 args.lora_target_modules 可能是 None
                    targets = config.target_modules if config and hasattr(config, "target_modules") else (args.lora_target_modules or [])
                    if isinstance(targets, str): targets = [targets] # 防止是字符串
                    
                    base_rank_map = {m: base_rank for m in targets}
                    
                    if base_rank_map:
                        modify_adapter(
                            model, 'local', modify_module_rank=base_rank_map,
                            lora_alpha=getattr(args, 'lora_alpha', 16),
                            lora_dropout=getattr(args, 'lora_dropout', 0.05),
                            init_lora_weights=(fedhera_default_lora_state is None),
                        )
                    if fedhera_default_lora_state is not None:
                        _ = model.load_state_dict(fedhera_default_lora_state, strict=False)
                    hera_hooks = None


            if (epoch > 0 or args.aggregation == 'fedhl') and args.aggregation != 'fedhera' and global_params is not None:
                local_client_load_weight(args, model, epoch, global_params=global_params)

            active_layers = None
            if fedhello_masks is not None:
                active_layers = fedhello_masks.get(client_id, [])
            client = GeneralClient(client_id, model, tokenizer, prompter, data_path, output_dir, cache_dir=args.cache_dir,
                                   hetero_lora=False, optim=optim, dataloader_num_workers=args.dataloader_num_workers,
                                   active_lora_layers=active_layers)

            logging.info("\nPreparing the local dataset and trainer for Client_{}".format(client_id))
            client.preprare_local_dataset()

            local_eval_result = client.test(epoch, args.local_micro_batch_size)
            local_eval_results += float(local_eval_result['eval_loss']) * local_dataset_len_dict[client_id]
            local_eval_rouge_1 += float(local_eval_result['eval_rouge1']) * local_dataset_len_dict[client_id]
            local_eval_rouge_L += float(local_eval_result['eval_rougeL']) * local_dataset_len_dict[client_id]

            logging.info("Initiating the local training of Client_{}".format(client_id))

            client.build_local_trainer(tokenizer,
                                       args.local_micro_batch_size,
                                       args.local_batch_size // args.local_micro_batch_size,
                                       args.local_num_epochs,
                                       args.local_learning_rate,
                                       args.group_by_length,
                                       args.warmup,
                                       profile_system_costs=args.profile_system_costs,
                                       profile_warmup_steps=args.profile_warmup_steps)
            client.initiate_local_training()

            logging.info("Local training starts ... ")
            local_train_result = client.train()
            local_train_results += float(local_train_result['eval_loss']) * local_dataset_len_dict[client_id]
            if args.profile_system_costs and client.latest_system_profile is not None:
                prof = client.latest_system_profile
                if prof.get("avg_step_latency_ms") is not None:
                    round_step_latency_values.append(float(prof["avg_step_latency_ms"]))
                    system_cost_profile["client_step_latency_ms_values"].append(float(prof["avg_step_latency_ms"]))
                if prof.get("peak_gpu_mem_mb") is not None:
                    round_peak_gpu_mem_values.append(float(prof["peak_gpu_mem_mb"]))
                    system_cost_profile["peak_gpu_mem_mb_values"].append(float(prof["peak_gpu_mem_mb"]))
                logging.info(
                    "[SystemProfile][epoch %d][client %s] avg_step_latency_ms=%s peak_gpu_mem_mb=%s measured_steps=%d warmup_steps=%d",
                    epoch,
                    str(client_id),
                    "None" if prof.get("avg_step_latency_ms") is None else f"{prof['avg_step_latency_ms']:.3f}",
                    "None" if prof.get("peak_gpu_mem_mb") is None else f"{prof['peak_gpu_mem_mb']:.3f}",
                    int(prof.get("num_measured_steps", 0)),
                    int(prof.get("warmup_steps", 0)),
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
                except Exception as e:
                    logging.error(
                        "[CrossClientDrift][epoch %d][client %s] Failed to reconstruct dense update: %s",
                        epoch,
                        str(client_id),
                        str(e),
                    )

            if (epoch > 0) and args.calc_drift:
                if k == 0: 
                    try:
                        drift_val = client.compute_oracle_drift(
                            global_params=global_params, 
                            oracle_r=args.oracle_rank,
                            lora_alpha=args.lora_alpha
                        )
                        logging.info(f"[DriftMetrics] Epoch {epoch}, Algorithm {args.aggregation}, Drift {drift_val}")
                        # 你可以将这个值存到 list 里最后画图
                    except RuntimeError as e:
                        logging.error(f"OOM during Oracle training: {e}")
                        torch.cuda.empty_cache()

            logging.info("\nTerminating the local training of Client_{}".format(client_id))
            model, local_dataset_len_dict, previously_selected_clients_set, last_client_id = client.terminate_local_training(
                epoch, local_dataset_len_dict, previously_selected_clients_set)
            if 'hera_hooks' in locals() and hera_hooks is not None:
                for _h in hera_hooks:
                    try:
                        _h.remove()
                    except Exception:
                        pass
            del client

            logging.info("Collecting the weights of clients and performing aggregation")
        if args.calc_cross_client_drift and cross_client_dense_updates:
            try:
                dispersion_stats = compute_pairwise_directional_dispersion(cross_client_dense_updates)
                if dispersion_stats is not None:
                    logging.info(
                        "[CrossClientDrift] Epoch %d Algorithm %s Dispersion %.6f NumClients %d NumPairs %d NumLayers %d ZeroNormPairs %d MeanUpdateNorm %.6f MinUpdateNorm %.6f MaxUpdateNorm %.6f",
                        epoch,
                        args.aggregation,
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
            except Exception as e:
                logging.error("[CrossClientDrift][epoch %d] Failed to compute dispersion: %s", epoch, str(e))
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
        if args.aggregation == 'homo':
            global_params = FedAvg(selected_clients_set,
                                   output_dir,
                                   local_dataset_len_dict,
                                   epoch,
                                   client_budgets=FL_training.client_budgets,
                                   layer_specs=FL_training.layer_specs,
                                   )
            if args.save_model:
                torch.save(global_params, os.path.join(output_dir, "adapter_model.bin"))
        elif args.aggregation == 'fedhera':
            new_global_params = FedHera(
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
                prev_global_params=dense_global_params if args.fedhera_server_agg == 'unbiased' else None,
                profile_metrics=system_cost_profile if args.profile_system_costs else None,
                force_coupled=args.fedhera_coupled,
            )
            if args.fedhera_server_agg == 'unbiased':
                if new_global_params is not None:
                    dense_global_params = new_global_params
                else:
                    logging.warning("FedHera returned None in unbiased mode! Check fed_utils implementation.")
            # adapter_model.bin 可存聚合Wg，便于可视化/对照
            # torch.save(_, os.path.join(output_dir, "adapter_model.bin"))
        elif args.aggregation == 'fedhello':
            global_params = FedHeLLo(
                selected_clients_set,
                output_dir,
                local_dataset_len_dict,
                epoch,
                active_layers_map=fedhello_masks,
                prev_global_params=global_params,
                layer_specs=FL_training.layer_specs,
            )
            if args.save_model:
                torch.save(global_params, os.path.join(output_dir, "adapter_model.bin"))
        elif args.aggregation == 'flexlora':
            global_params = FlexLoRA(selected_clients_set,
                                   output_dir,
                                   local_dataset_len_dict,
                                   epoch,
                                   client_budgets=FL_training.client_budgets,
                                   layer_specs=FL_training.layer_specs,
                                   )
            if args.save_model:
                torch.save(global_params, os.path.join(output_dir, "adapter_model.bin"))
            global_params = distribute_weight_fast(global_params, config_local)
        elif args.aggregation == 'hetlora':
            global_params = HetLoRA(
                selected_clients_set,
                output_dir,
                local_dataset_len_dict,
                epoch,
                client_rank_map=config_local,
                target_global_rank=getattr(FL_training, "hetlora_global_rank", args.lora_r),
                prev_global_params=global_params,
                client_budgets=FL_training.client_budgets,
                layer_specs=FL_training.layer_specs,
            )
            if args.save_model:
                torch.save(global_params, os.path.join(output_dir, "adapter_model.bin"))
        elif args.aggregation == 'flora':
            global_params = FLoRA(selected_clients_set,
                                  output_dir,
                                  local_dataset_len_dict,
                                  epoch,
                                  client_budgets=FL_training.client_budgets,
                                  layer_specs=FL_training.layer_specs,
                                  )
            if args.save_model:
                torch.save(global_params, os.path.join(output_dir, "adapter_model.bin"))
            # FLoRA in this codebase aggregates to dense weights, so we must redistribute via SVD
            global_params = distribute_weight_fast(global_params, config_local)
        elif args.aggregation == 'fedhl':
            
            new_dense_params = FedHL(
                selected_clients_set,
                output_dir,
                local_dataset_len_dict,
                epoch,
                prev_global_params=dense_global_params, 
                layer_specs=FL_training.layer_specs
            )
            
            dense_global_params = new_dense_params
            
            if args.save_model:
                torch.save(dense_global_params, os.path.join(output_dir, "dense_model.bin"))
                
            global_params = distribute_weight_fast(dense_global_params, config_local)
        else:
            raise ValueError(f"Unsupported aggregation mode: {args.aggregation}")

        global_eval_rouge_L = local_eval_rouge_L / total_data_num

        ### early stop
        if args.early_stop:
            if best_rouge_L < global_eval_rouge_L:
                best_rouge_L = global_eval_rouge_L
                best_round = epoch
                current_count = 0
            else:
                current_count += 1
            if current_count > patience:
                logging.info(f"Best round is {best_round} with test_rouge_L {best_rouge_L}")
                # Log final communication / compute statistics before exiting.
                try:
                    stats = get_traffic_stats()
                    logging.info("[TrafficSummary] %s", stats)
                except Exception:
                    pass
                if args.profile_system_costs:
                    _summarize_system_costs(args, output_dir, system_cost_profile)
                return
        local_dataset_len_dict = {}
        import gc
        gc.collect()

    # Training finished without early stopping: record final traffic stats.
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
    if not os.path.exists(args.session_name):
        os.makedirs(args.session_name)
    log_dir = "/root/nfs/fedhera"
    os.makedirs(log_dir, exist_ok=True)
    model_tag = os.path.basename(str(args.global_model)).replace("/", "_")
    dataset_tag = os.path.basename(os.path.normpath(args.data_path))
    log_name = f"{args.aggregation}_{args.hetero_mode}_{model_tag}_{dataset_tag}_r{args.lora_r}_{args.dirichlet_alpha}.log"
    log_path = os.path.join(log_dir, log_name)
    logging.basicConfig(filename=log_path,
                        level=logging.INFO,
                        format='%(message)s')
    logging.info("Logging to %s", log_path)
    logging.info("Initial training parameters %s", args)
    if args.dirichlet_alpha is not None:
        raw_path = args.data_path.rstrip('/')
        args.data_path = f"{raw_path}_noniid_{args.dirichlet_alpha}"
        logging.info(f"Dirichlet alpha={args.dirichlet_alpha} detected. Switching data_path to: {args.data_path}")
    print(args)


    logging.info(str(socket.gethostbyname(socket.gethostname())))

    data_path = os.path.join(args.data_path, str(args.num_clients))
    if args.output_dir:
        output_dir = os.path.join(args.output_dir, args.session_name, args.aggregation)
    else:
        output_dir = os.path.join(args.session_name, args.aggregation)

    # set up the global model & tokenizer
    model, tokenizer = model_and_tokenizer(global_model=args.global_model, device_map=args.device_map)

    prompter = Prompter(args.prompt_template_name)

    # Choose model-appropriate LoRA target modules and heterogeneity configs.
    lora_target_modules, config_types = resolve_lora_targets_and_config_types(
        model,
        user_target_modules=args.lora_target_modules,
    )

    # Build layer specs and budgets once for all aggregation strategies.
    layer_specs = extract_lora_layer_specs(model, lora_target_modules)
    client_budgets = calculate_dynamic_budgets(layer_specs, args.num_clients, args.hetero_mode, seed=args.seed)
    calculated_ranks = calculate_unified_rank_from_budget(client_budgets, layer_specs)
    FL_training.layer_specs = layer_specs
    FL_training.client_budgets = client_budgets
    if args.aggregation == 'fedhello':
        fedhello_counts, avg_params = calculate_active_layers_from_budget(
            client_budgets, layer_specs, args.lora_r
        )
        FL_training.fedhello_active_layer_counts = fedhello_counts
        FL_training.fedhello_avg_params_per_layer = avg_params
        logging.info(
            "[FedHeLLo] avg_params_per_layer=%.1f num_active_layers_sample=%s",
            avg_params,
            dict(list(fedhello_counts.items())[:3]),
        )
    else:
        FL_training.fedhello_active_layer_counts = None
        FL_training.fedhello_avg_params_per_layer = None

    if args.aggregation == 'fedhera':
        fixed_ranks = None
    elif args.aggregation == 'flexlora':
        logging.info("Calculating FlexLoRA ranks based on client budgets...")
        fixed_ranks = calculated_ranks
    elif args.aggregation == 'hetlora':
        logging.info("Calculating HetLoRA ranks based on client budgets...")
        fixed_ranks = calculated_ranks
    elif args.aggregation == 'flora':
        logging.info("Calculating FLoRA ranks based on client budgets (Same as FlexLoRA)...")
        fixed_ranks = calculated_ranks
    elif args.aggregation == 'fedhl':
        logging.info("Calculating FedHL ranks based on client budgets...")
        fixed_ranks = calculated_ranks
    elif args.aggregation == 'homo':
        min_rank = min(calculated_ranks.values()) if calculated_ranks else 1
        logging.info(f"Homo: Bottleneck detected. Setting unified rank to {min_rank} for all clients.")
        fixed_ranks = {i: min_rank for i in range(args.num_clients)}
    elif args.aggregation == 'fedhello':
        fixed_ranks = None
    else:
        raise ValueError(f"Unsupported aggregation method: {args.aggregation}")
    config_local = get_peft(
        config_types,
        num_clients=args.num_clients,
        strategy=args.aggregation,
        hetero_mode=args.hetero_mode,
        seed=args.seed,
        fixed_ranks=fixed_ranks,
    )
    if args.aggregation == 'hetlora':
        FL_training.hetlora_global_rank = max(calculated_ranks.values()) if calculated_ranks else int(args.lora_r)

    logging.info(config_local)

    config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=lora_target_modules,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )
    if args.baseline != 'slora':
        model = get_peft_model(model, config, adapter_name = 'local')
    
    # world_size = int(os.environ.get("WORLD_SIZE", 1))
    # ddp = world_size != 1
    # if not ddp and torch.cuda.device_count() > 1:
    #     model.is_parallelizable = True
    #     model.model_parallel = True

    FL_training(model, tokenizer, prompter, data_path, output_dir, args, config_local=config_local, config=config, config_types=config_types)


if __name__ == "__main__":
    main()
