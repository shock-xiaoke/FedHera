import json
import os
import random

import numpy as np
import peft
import torch


def seed_torch(seed, deterministic=False):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = not deterministic
    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cuda.matmul.allow_tf32 = not deterministic
    torch.backends.cudnn.allow_tf32 = not deterministic


def tokenize(tokenizer, prompt, cutoff_len=512, add_eos_token=True):
    result = tokenizer(
        prompt,
        truncation=True,
        max_length=cutoff_len,
        padding=False,
        return_tensors=None,
    )
    if (
        result["input_ids"][-1] != tokenizer.eos_token_id
        and len(result["input_ids"]) < cutoff_len
        and add_eos_token
    ):
        result["input_ids"].append(tokenizer.eos_token_id)
        result["attention_mask"].append(1)

    result["labels"] = result["input_ids"].copy()
    return result


def modify_adapter(
    peft_model,
    adapter_name,
    modify_module_rank=None,
    layer_dict=None,
    lora_alpha=16,
    lora_dropout=0.05,
    init_lora_weights=True,
):
    """
    Update LoRA ranks for modules whose names contain keys in ``modify_module_rank``.
    If ``layer_dict`` is empty, all matching layers are updated.
    """
    if modify_module_rank is None:
        modify_module_rank = {}
    if layer_dict is None:
        layer_dict = []

    for name, module in peft_model.named_modules():
        if layer_dict and not any(f".{layer}." in name for layer in layer_dict):
            continue

        for key, rank in modify_module_rank.items():
            alpha = rank if lora_alpha == 0 else lora_alpha
            is_lora_linear = isinstance(module, peft.tuners.lora.Linear)
            is_lora_8bit = isinstance(module, peft.tuners.lora.Linear8bitLt)
            if key not in name or not (is_lora_linear or is_lora_8bit):
                continue

            use_rslora = False
            try:
                if hasattr(peft_model, "peft_config") and adapter_name in peft_model.peft_config:
                    use_rslora = bool(
                        getattr(peft_model.peft_config[adapter_name], "use_rslora", False)
                    )
            except Exception:
                use_rslora = False

            try:
                module.update_layer(
                    adapter_name,
                    rank,
                    alpha,
                    lora_dropout,
                    init_lora_weights,
                    use_rslora,
                )
            except TypeError:
                module.update_layer(
                    adapter_name,
                    rank,
                    alpha,
                    lora_dropout,
                    init_lora_weights,
                )


def apply_lora_prefix_mask(peft_model, per_layer_r_main):
    """
    Register gradient masks so each layer only updates the first ``r_main`` columns/rows.
    """
    hooks = []
    for name, param in peft_model.named_parameters():
        if "lora_A" not in name and "lora_B" not in name:
            continue

        base_key = ".".join(name.split(".")[:-3]) + ".lora"
        r_main = int(per_layer_r_main.get(base_key, 0))
        if r_main <= 0:
            mask = torch.zeros_like(param, dtype=param.dtype, device=param.device)
        elif "lora_A" in name:
            mask = torch.zeros_like(param)
            mask[:r_main, :] = 1
        else:
            mask = torch.zeros_like(param)
            mask[:, :r_main] = 1

        def _make_hook(local_mask):
            def hook_fn(grad):
                if grad is None:
                    return None
                return grad * local_mask.to(grad.device)

            return hook_fn

        hooks.append(param.register_hook(_make_hook(mask)))
    return hooks


def load_weight_fedhera_if_exists(output_dir, client_id, epoch):
    """
    Load the last server push package for a client if it exists.

    If ATW stored a per-client lambda in the push metadata, scale the frozen tail
    so the effective dense update is multiplied by lambda.
    """
    push_dir = os.path.join(output_dir, str(client_id), f"server_push_epoch_{epoch}")
    model_path = os.path.join(push_dir, "pytorch_model.bin")
    meta_path = os.path.join(push_dir, "meta.json")

    if not (os.path.exists(model_path) and os.path.exists(meta_path)):
        return None, None

    state = torch.load(model_path, map_location="cpu")
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    for layer_key, info in meta.items():
        if info.get("skip", False):
            continue

        lambda_val = float(info.get("lambda", 1.0))
        if lambda_val >= 0.999:
            continue

        r_main = int(info.get("r_main", 0))
        r_tot = int(info.get("r_tot", 0))
        if r_main >= r_tot:
            continue

        key_a = layer_key + "_A.local.weight"
        key_b = layer_key + "_B.local.weight"
        if key_a not in state or key_b not in state:
            continue

        tensor_a = state[key_a]
        tensor_b = state[key_b]
        if r_main >= tensor_a.shape[0] or r_main >= tensor_b.shape[1]:
            continue

        scale = lambda_val ** 0.5
        tensor_a[r_main:, :].mul_(scale)
        tensor_b[:, r_main:].mul_(scale)

    return state, meta
