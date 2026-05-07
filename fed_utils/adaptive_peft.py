import random
import numpy as np
import os
import torch
import peft
from tqdm import tqdm

def seed_torch(seed, deterministic=False):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
    # Let cuDNN/TF32 stay fast by default; flip deterministic on only if requested.
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

def load_weight_local(weighted_single_weights, model):
    weight_dict = {}
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(param.shape)
            print(name)
            rank = min(param.shape[0], param.shape[1])
            if name + '.' + str(rank) in weighted_single_weights.keys():
                weight_dict[name] = weighted_single_weights[name + '.' + str(rank)]
    return weight_dict


def load_weight_hetlora(global_params, model):
    """
    Truncate a shared HetLoRA global adapter to the current client's local rank.
    """
    weight_dict = {}
    if global_params is None:
        return weight_dict
    for name, param in model.named_parameters():
        if not param.requires_grad or name not in global_params:
            continue
        src = global_params[name].detach().to(device='cpu', dtype=param.dtype)
        if "lora_A" in name:
            local_rank = int(param.shape[0])
            copy_rank = min(local_rank, int(src.shape[0]))
            tensor = torch.zeros_like(param, device='cpu')
            if copy_rank > 0:
                tensor[:copy_rank, :] = src[:copy_rank, :]
            weight_dict[name] = tensor
        elif "lora_B" in name:
            local_rank = int(param.shape[1])
            copy_rank = min(local_rank, int(src.shape[1]))
            tensor = torch.zeros_like(param, device='cpu')
            if copy_rank > 0:
                tensor[:, :copy_rank] = src[:, :copy_rank]
            weight_dict[name] = tensor
        elif tuple(src.shape) == tuple(param.shape):
            weight_dict[name] = src.clone()
    return weight_dict


def distribute_weight(weighted_single_weights, model):
    # mode is local model
    # around 15 min for one client
    weight_dict = {}
    for key in tqdm(weighted_single_weights.keys()):
        # _, target, target_name = peft.utils.other._get_submodules(model, key + '_A.local')
        rank = 2048
        merge_rate = 16 / rank
        W_cpu = (weighted_single_weights[key] / merge_rate).detach().to('cpu')
        u, s, vT = torch.linalg.svd(W_cpu, full_matrices=False)
        u = u[:, :rank]
        s = s[:rank]
        v = vT[:rank, :]
        lora_B = u @ torch.diag(s)
        lora_A = v
        weight_dict[key + '_A.local.weight'] = lora_A
        weight_dict[key + '_B.local.weight'] = lora_B
        # print(key + '_A.local.weight', lora_A.shape)
        # print(key + '_B.local.weight', lora_B.shape)
    return weight_dict

def distribute_weight_fast(weighted_single_weights, config_local):
    # mode is local model, model needs to load local weights first
    weight_dict = {}
    rank_dict = {}
    alpha = config_local['alpha']
    for client, val in config_local.items():
        if 'Client' in client:
            for key in val.keys():
                if key in rank_dict.keys():
                    rank_dict[key].append(val[key])
                else:
                    rank_dict[key] = [val[key]]

    for key in tqdm(weighted_single_weights.keys()):
        W_cpu = weighted_single_weights[key].detach().to(device='cpu', dtype=torch.float32)
        u, s, vT = torch.linalg.svd(W_cpu, full_matrices=False)
        for layer, rank_lst in rank_dict.items():
            if layer in key:
                break
        for rank in rank_lst:
            if rank != 0:
                U = u[:, :rank]
                S = s[:rank]
                V = vT[:rank, :]
                lora_B = U @ torch.diag(S)
                lora_A = V
                # merge_rate = 2
                merge_rate = alpha/rank
                weight_dict[key + '_A.local.weight.' + str(rank)] = lora_A
                weight_dict[key + '_B.local.weight.' + str(rank)] = lora_B/ merge_rate
    return weight_dict


def modify_adapter(peft_model, adapter_name, modify_module_rank=None, layer_dict=None,
                   lora_alpha=16, lora_dropout=0.05, init_lora_weights=True):
    """
    Update LoRA ranks for modules whose names contain keys in ``modify_module_rank``.
    If ``layer_dict`` is None or empty, all layers are considered; otherwise only
    layers whose name contains ``.{layer}.`` for some layer in ``layer_dict`` are updated.
    """
    if modify_module_rank is None:
        modify_module_rank = {}
    if layer_dict is None:
        layer_dict = []

    for name, module in peft_model.named_modules():
        # If layer_dict is empty, match all layers; otherwise restrict to the given list.
        if layer_dict and not any(f".{layer}." in name for layer in layer_dict):
            continue
        for key, r in modify_module_rank.items():
            if lora_alpha == 0:
                alpha = r
            else:
                alpha = lora_alpha
            
            if key in name and (isinstance(module, peft.tuners.lora.Linear) or isinstance(module, peft.tuners.lora.Linear8bitLt)):
                # Try PEFT new signature (requires use_rslora); fallback to old signature.
                use_rslora = False
                try:
                    # If available, read from peft config
                    if hasattr(peft_model, "peft_config") and adapter_name in peft_model.peft_config:
                        use_rslora = bool(getattr(peft_model.peft_config[adapter_name], "use_rslora", False))
                except Exception:
                    use_rslora = False

                try:
                    module.update_layer(adapter_name, r, alpha, lora_dropout, init_lora_weights, use_rslora)
                except TypeError:
                    module.update_layer(adapter_name, r, alpha, lora_dropout, init_lora_weights)



# fed_utils/adaptive_peft.py (append)
import json
import torch.nn as nn

def apply_lora_prefix_mask(peft_model, per_layer_r_main):
    """
    peft_model: PEFT LoRA 模型
    per_layer_r_main: Dict[layer_key] -> int
    对 A/B 注册梯度hook：只让前 r_main 列/行产生梯度。
    """
    hooks = []
    for name, param in peft_model.named_parameters():
        if "lora_A" in name or "lora_B" in name:
            # name 示例: model.layers.0.self_attn.q_proj.lora_A.local.weight
            base_key = '.'.join(name.split('.')[:-3]) + '.lora'
            r_main = int(per_layer_r_main.get(base_key, 0))
            if r_main <= 0:
                mask = torch.zeros_like(param, dtype=param.dtype, device=param.device)
            else:
                if "lora_A" in name:
                    # A: [r, d_in] -> 只保留前 r_main 行
                    mask = torch.zeros_like(param)
                    mask[:r_main, :] = 1
                else:
                    # B: [d_out, r] -> 只保留前 r_main 列
                    mask = torch.zeros_like(param)
                    mask[:, :r_main] = 1

            def _make_hook(msk):
                def hook_fn(grad):
                    if grad is None: return None
                    return grad * msk.to(grad.device)
                return hook_fn

            hooks.append(param.register_hook(_make_hook(mask)))
    return hooks

# [修改] fed_utils/adaptive_peft.py

def load_weight_fedhera_if_exists(output_dir, client_id, epoch):
    """
    若存在 server_push 包，读取并返回 (state_dict, meta)；否则返回 (None, None)
    新增逻辑：如果 meta 中包含 lambda 且 lambda < 1，则对 Frozen Tail 进行缩放。
    """
    import os, json, torch
    push_dir = os.path.join(output_dir, str(client_id), f"server_push_epoch_{epoch}")
    model_path = os.path.join(push_dir, "pytorch_model.bin")
    meta_path  = os.path.join(push_dir, "meta.json")
    
    if os.path.exists(model_path) and os.path.exists(meta_path):
        state = torch.load(model_path, map_location="cpu")
        with open(meta_path, "r") as f:
            meta = json.load(f)
            
        # [新增] 应用 ATW Lambda Scaling
        # 我们需要在加载前修改 state 中的权重
        # 逻辑：对于每一层，获取 r_main 和 lambda。
        # A: [r_tot, d_in] -> Tail 是 row[r_main:]
        # B: [d_out, r_tot] -> Tail 是 col[:, r_main:]
        # 将 Tail 部分乘以 sqrt(lambda) (因为 W = B@A，两边各乘 sqrt(lambda) 等于整体乘 lambda)
        # 或者只乘一边。为了对称，通常各乘 sqrt(lambda)。
        
        # 检查是否所有层的 lambda 都一样（目前的实现是 client 级 lambda）
        # 直接遍历 meta 即可
        
        for layer_key, info in meta.items():
            if info.get("skip", False):
                continue
            
            lambda_val = info.get("lambda", 1.0)
            if lambda_val >= 0.999: # 接近 1 则不处理
                continue
                
            r_main = int(info.get("r_main", 0))
            r_tot = int(info.get("r_tot", 0))
            
            if r_main >= r_tot: # 没有 tail
                continue

            # 找到对应的 tensor key
            # meta key 是 base key (e.g. ...lora), state key 是 ...lora_A.local.weight
            key_A = layer_key + "_A.local.weight"
            key_B = layer_key + "_B.local.weight"
            
            if key_A in state and key_B in state:
                tensor_A = state[key_A]
                tensor_B = state[key_B]
                
                current_r_A = tensor_A.shape[0] # [r, d_in]
                current_r_B = tensor_B.shape[1] # [d_out, r]
                
                if r_main >= current_r_A or r_main >= current_r_B:
                    continue

                scale = lambda_val ** 0.5
                tensor_A[r_main:, :].mul_(scale)
                tensor_B[:, r_main:].mul_(scale)
                
        return state, meta
        
    return None, None
