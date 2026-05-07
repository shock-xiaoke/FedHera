import numpy as np
import torch
import math
import copy
import os
from torch.nn.functional import normalize
import gc
import logging
import time
from tqdm import tqdm


# Simple global traffic counters to let the training loop
# summarise total communication and compute volume.
TRAFFIC_STATS = {
    "FlexLoRA": {"transmit_MB": 0.0, "compute_MB": 0.0},
    "HetLoRA": {"transmit_MB": 0.0, "compute_MB": 0.0},
    "FedHera": {"transmit_MB": 0.0, "compute_MB": 0.0},
    "FedHeLLo": {"transmit_MB": 0.0, "compute_MB": 0.0},
    "FLoRA": {"transmit_MB": 0.0, "compute_MB": 0.0},
}
# [NEW] Global cache for ATW stats: {client_id: {"s": float, "t": int}}
FEDHERA_CLIENT_STATS = {}


def reset_traffic_stats():
    for method in TRAFFIC_STATS:
        TRAFFIC_STATS[method]["transmit_MB"] = 0.0
        TRAFFIC_STATS[method]["compute_MB"] = 0.0


def get_traffic_stats():
    # Return a shallow copy so callers cannot mutate internals.
    return {
        name: stats.copy()
        for name, stats in TRAFFIC_STATS.items()
    }


def _resolve_client_rank(client_rank_map, client_id):
    if client_rank_map is None:
        return 0
    rank_info = client_rank_map.get(int(client_id), client_rank_map.get(client_id, 0))
    if isinstance(rank_info, dict):
        vals = [int(v) for v in rank_info.values() if int(v) > 0]
        return max(vals) if vals else 0
    if isinstance(rank_info, (list, tuple)):
        vals = [int(v) for v in rank_info if int(v) > 0]
        return max(vals) if vals else 0
    try:
        return int(rank_info)
    except Exception:
        return 0


def HetLoRA(
    selected_clients_set,
    output_dir,
    local_dataset_len_dict,
    epoch,
    client_rank_map,
    target_global_rank,
    prev_global_params=None,
    client_budgets=None,
    layer_specs=None,
):
    """
    HetLoRA baseline:
    1. Compute each participating client's sparsity score ||B_k A_k||_F.
    2. Normalize scores to aggregation weights.
    3. Zero-pad heterogeneous LoRA A/B into a shared global rank.
    4. Aggregate directly in factor space without SVD redistribution.
    """
    if not selected_clients_set:
        return prev_global_params if prev_global_params is not None else {}

    selected_clients = list(selected_clients_set)
    target_global_rank = int(max(1, target_global_rank))
    client_loaded_weights = {}
    client_scores = {}

    round_transmit_bytes = 0.0
    round_compute_bytes = 0.0
    compute_ms_per_param = 1.7e-04
    MB = 1024.0 * 1024.0

    for client_id in tqdm(selected_clients, desc="HetLoRA Aggregation"):
        rank = _resolve_client_rank(client_rank_map, client_id)
        model_path = os.path.join(output_dir, str(client_id), f"local_output_epoch_{epoch}", "pytorch_model.bin")
        if not os.path.exists(model_path):
            logging.warning("[HetLoRA][epoch %d] Missing model for client %s: %s", epoch, str(client_id), model_path)
            client_scores[int(client_id)] = 0.0
            continue

        state = torch.load(model_path, map_location="cpu")
        client_loaded_weights[int(client_id)] = state

        total_norm_squared = 0.0
        for key in list(state.keys()):
            if 'local' not in key or 'bias' in key or 'lora_A' not in key:
                continue
            B_key = key.replace('lora_A', 'lora_B')
            if B_key not in state:
                continue

            lora_A = state[key].float()
            lora_B = state[B_key].float()
            client_rank = int(lora_A.shape[0])
            if client_rank <= 0 or client_rank != int(lora_B.shape[1]):
                continue

            BT_B = torch.matmul(lora_B.T, lora_B)
            BT_B_A = torch.matmul(BT_B, lora_A)
            total_norm_squared += torch.sum(lora_A * BT_B_A).item()

            d_in = int(lora_A.shape[1])
            d_out = int(lora_B.shape[0])
            elem_bytes = lora_A.element_size()
            bytes_this = float((d_in + d_out) * client_rank * elem_bytes)
            round_transmit_bytes += bytes_this
            round_compute_bytes += bytes_this

        client_scores[int(client_id)] = math.sqrt(max(total_norm_squared, 0.0))

        if client_budgets is not None:
            budget = client_budgets.get(int(client_id)) if isinstance(client_budgets, dict) else None
            if budget is not None:
                comm_bytes = 0.0
                comp_time_ms = 0.0
                for key in list(state.keys()):
                    if 'local' not in key or 'bias' in key or 'lora_A' not in key:
                        continue
                    B_key = key.replace('lora_A', 'lora_B')
                    if B_key not in state:
                        continue
                    A_w = state[key]
                    B_w = state[B_key]
                    layer_rank = int(A_w.shape[0])
                    base_key = '.'.join(key.split('.')[:-3]) + '.lora'
                    d_out = int(B_w.shape[0])
                    d_in = int(A_w.shape[1])
                    if layer_specs is not None and base_key in layer_specs:
                        spec = layer_specs[base_key]
                        d_out = int(spec.get("d_out", d_out))
                        d_in = int(spec.get("d_in", d_in))
                    comm_bytes += float(layer_rank * (d_in + d_out) * 2.0)
                    comp_time_ms += float(layer_rank * (d_in + d_out) * compute_ms_per_param)

                B_down_bytes = float(budget.get("B_down_MB", 0.0)) * MB
                T_ms = float(budget.get("step_ms", 0.0))
                comm_util = (comm_bytes / B_down_bytes) if B_down_bytes > 0 else 0.0
                comp_util = (comp_time_ms / T_ms) if T_ms > 0 else 0.0
                logging.info(
                    "[HetLoRA][epoch %d][client %s] score=%.6f Comm Util: %.1f%%, Comp Util: %.1f%%",
                    epoch,
                    str(client_id),
                    client_scores[int(client_id)],
                    comm_util * 100.0,
                    comp_util * 100.0,
                )

    valid_client_ids = [cid for cid in selected_clients if int(cid) in client_loaded_weights]
    if not valid_client_ids:
        return prev_global_params if prev_global_params is not None else {}

    denom = sum(client_scores.get(int(cid), 0.0) for cid in valid_client_ids)
    if denom <= 0:
        agg_weights = {int(cid): 1.0 / len(valid_client_ids) for cid in valid_client_ids}
        logging.info("[HetLoRA][epoch %d] All client scores are zero; using uniform weights.", epoch)
    else:
        agg_weights = {
            int(cid): client_scores.get(int(cid), 0.0) / denom
            for cid in valid_client_ids
        }

    aggregated_params = {}
    template_state = prev_global_params if prev_global_params else client_loaded_weights[int(valid_client_ids[0])]
    for key, param in template_state.items():
        if 'lora_A' in key:
            aggregated_params[key] = torch.zeros((target_global_rank, int(param.shape[1])), dtype=param.dtype)
        elif 'lora_B' in key:
            aggregated_params[key] = torch.zeros((int(param.shape[0]), target_global_rank), dtype=param.dtype)
        else:
            aggregated_params[key] = torch.zeros_like(param, device='cpu')

    for client_id in valid_client_ids:
        state = client_loaded_weights[int(client_id)]
        weight = float(agg_weights[int(client_id)])
        for key, param in state.items():
            if key not in aggregated_params:
                continue
            tensor = param.detach().to('cpu')
            if 'lora_A' in key:
                rank = min(int(tensor.shape[0]), target_global_rank)
                aggregated_params[key][:rank, :] += tensor[:rank, :] * weight
            elif 'lora_B' in key:
                rank = min(int(tensor.shape[1]), target_global_rank)
                aggregated_params[key][:, :rank] += tensor[:, :rank] * weight
            else:
                if aggregated_params[key].shape == tensor.shape:
                    aggregated_params[key] += tensor * weight

    logging.info(
        "[HetLoRA][epoch %d] target_global_rank=%d weights=%s",
        epoch,
        target_global_rank,
        {cid: round(w, 6) for cid, w in sorted(agg_weights.items())},
    )
    TRAFFIC_STATS["HetLoRA"]["transmit_MB"] += round_transmit_bytes / MB
    TRAFFIC_STATS["HetLoRA"]["compute_MB"] += round_compute_bytes / MB
    logging.info(
        "[HetLoRA][epoch %d] transmit_MB=%.3f compute_MB=%.3f",
        epoch,
        round_transmit_bytes / MB,
        round_compute_bytes / MB,
    )
    return aggregated_params

def FLoRA(selected_clients_set, output_dir, local_dataset_len_dict, epoch, client_budgets=None, layer_specs=None):
    """
    FLoRA Aggregation: Stack heterogeneous LoRA modules.
    Ref: FLoRA: Federated Fine-Tuning Large Language Models with Heterogeneous Low-Rank Adaptations
    
    Implementation:
    1. Collect A and B matrices from all clients.
    2. Apply weighting p_k to Matrix A.
    3. Stack A (vertically) and B (horizontally).
    4. Compute B_stack @ A_stack to get the global dense update.
    5. Return the dense update so 'distribute_weight_fast' can redistribute it via SVD.
    """
    # Calculate p_k (weights_array)
    weights_array = torch.tensor(
        [local_dataset_len_dict[client_id] for client_id in selected_clients_set], dtype=torch.float32
    )
    weights_array = torch.nn.functional.normalize(weights_array, p=1, dim=0)
    
    # Storage for stacking
    # Structure: { 'module_name': { 'A': [list_of_tensors], 'B': [list_of_tensors] } }
    stacking_buffer = {}
    
    round_transmit_bytes = 0.0
    round_compute_bytes = 0.0
    compute_ms_per_param = 1.7e-04
    MB = 1024.0 * 1024.0

    with torch.no_grad():
        for k, client_id in tqdm(enumerate(selected_clients_set)):
            single_output_dir = os.path.join(
                output_dir, str(client_id), f"local_output_epoch_{epoch}", "pytorch_model.bin"
            )
            single_weights = torch.load(single_output_dir, map_location='cpu')
            
            comm_bytes = 0.0
            comp_time_ms = 0.0
            
            for key in list(single_weights.keys()):
                # Identify valid LoRA A keys
                if 'local' in key and 'bias' not in key and 'lora_A' in key:
                    B_key = key.replace('lora_A', 'lora_B')
                    
                    # Base module name (e.g., base_model.model.model.layers.0.self_attn.q_proj.lora)
                    base_key = '.'.join(key.split('.')[:-3]) + '.lora'
                    
                    if base_key not in stacking_buffer:
                        stacking_buffer[base_key] = {'A': [], 'B': []}
                    
                    A_w = single_weights[key]
                    B_w = single_weights[B_key]
                    rank = A_w.shape[0] # LoRA A is [r, d_in]
                    
                    # Apply Scaling Factor p_k to A only (Eq 11 in FLoRA paper)
                    # Note: We also handle the scaling factor `merge_rate` (alpha/r) here if needed, 
                    # but typically merge_rate is baked into forward. 
                    # For consistency with FlexLoRA code in this repo, we apply merge_rate sqrt adjustment or standard logic.
                    # FlexLoRA code does: (B @ A) * merge_rate * weight.
                    # To achieve equivalence via stacking: A_new = A * weight * merge_rate, B_new = B.
                    # Or split merge_rate between them. Let's follow the standard:
                    merge_rate = 16 / max(rank, 1)
                    
                    # We apply the full scalar weight to A for simplicity of stacking
                    A_weighted = A_w * weights_array[k] * merge_rate
                    
                    stacking_buffer[base_key]['A'].append(A_weighted)
                    stacking_buffer[base_key]['B'].append(B_w)

                    # --- Traffic Stats Tracking ---
                    d_in = A_w.shape[1]
                    d_out = B_w.shape[0]
                    elem_bytes = A_w.element_size()
                    bytes_this = (d_in + d_out) * rank * elem_bytes
                    round_transmit_bytes += bytes_this
                    round_compute_bytes += bytes_this
                    
                    if client_budgets is not None:
                        # Client-side utility tracking
                        if layer_specs is not None and base_key in layer_specs:
                            spec = layer_specs[base_key]
                            d_out_spec = int(spec.get("d_out", d_out))
                            d_in_spec = int(spec.get("d_in", d_in))
                        else:
                            d_out_spec, d_in_spec = d_out, d_in
                        
                        comm_bytes += float(rank * (d_in_spec + d_out_spec) * 2.0)
                        comp_time_ms += float(rank * (d_in_spec + d_out_spec) * compute_ms_per_param)

            if client_budgets is not None:
                budget = client_budgets.get(int(client_id)) if isinstance(client_budgets, dict) else None
                if budget is not None:
                    B_down_bytes = float(budget.get("B_down_MB", 0.0)) * MB
                    T_ms = float(budget.get("step_ms", 0.0))
                    comm_util = (comm_bytes / B_down_bytes) if B_down_bytes > 0 else 0.0
                    comp_util = (comp_time_ms / T_ms) if T_ms > 0 else 0.0
                    logging.info(
                        "[FLoRA][epoch %d][client %s] Comm Util: %.1f%%, Comp Util: %.1f%%",
                        epoch,
                        str(client_id),
                        comm_util * 100.0,
                        comp_util * 100.0,
                    )

            del single_weights
            torch.cuda.empty_cache()

    # Perform Stacking and Multiplication
    weighted_single_weights = {}
    for base_key, matrices in stacking_buffer.items():
        if not matrices['A']: 
            continue
            
        # Stack A vertically (dim 0 for [r, d_in]) -> Result [Total_R, d_in]
        A_stack = torch.cat(matrices['A'], dim=0)
        
        # Stack B horizontally (dim 1 for [d_out, r]) -> Result [d_out, Total_R]
        B_stack = torch.cat(matrices['B'], dim=1)
        
        # Compute global dense weight W = B_stack @ A_stack
        # This effectively sums B_k @ (A_k * p_k)
        merged_weight = B_stack @ A_stack
        
        weighted_single_weights[base_key] = merged_weight
        
        del A_stack, B_stack, merged_weight
        torch.cuda.empty_cache()

    TRAFFIC_STATS["FLoRA"]["transmit_MB"] += round_transmit_bytes / MB
    TRAFFIC_STATS["FLoRA"]["compute_MB"] += round_compute_bytes / MB
    logging.info(
        "[FLoRA][epoch %d] transmit_MB=%.3f compute_MB=%.3f",
        epoch,
        round_transmit_bytes / MB,
        round_compute_bytes / MB,
    )
    
    return weighted_single_weights

def FedAvg(selected_clients_set, output_dir, local_dataset_len_dict, epoch, client_budgets=None, layer_specs=None):
    weights_array = normalize(
        torch.tensor([local_dataset_len_dict[client_id] for client_id in selected_clients_set],
                     dtype=torch.float32),
        p=1, dim=0)
    for k, client_id in enumerate(selected_clients_set):
        single_output_dir = os.path.join(output_dir, str(client_id), "local_output_epoch_{}".format(epoch),
                                         "pytorch_model.bin")
        single_weights = torch.load(single_output_dir, map_location='cpu')
        # delete_lst = []
        # for key in single_weights.keys():
        #     if 'bias' in key and 'lora_B' in key:
        #         delete_lst.append(key)
        # for key in delete_lst:
        #     del single_weights[key]
        with torch.no_grad():
            if k == 0:
                weighted_single_weights = {key: 0 for key in
                                           single_weights.keys()}

            weighted_single_weights = {key: weighted_single_weights[key] + single_weights[key] * (weights_array[k])
                                       for key in
                                       single_weights.keys()}
        if client_budgets is not None:
            budget = client_budgets.get(int(client_id)) if isinstance(client_budgets, dict) else None
            if budget is not None:
                comm_bytes = 0.0
                comp_time_ms = 0.0
                for key in list(single_weights.keys()):
                    if 'local' not in key or 'bias' in key or 'lora_A' not in key:
                        continue
                    B_key = key.replace('lora_A', 'lora_B')
                    if B_key not in single_weights:
                        continue
                    A_w = single_weights[key]
                    B_w = single_weights[B_key]
                    rank = int(B_w.shape[1])
                    base_key = '.'.join(key.split('.')[:-3]) + '.lora'
                    d_out = int(B_w.shape[0])
                    d_in = int(A_w.shape[1]) if A_w.ndim >= 2 else int(A_w.shape[-1])
                    if layer_specs is not None and base_key in layer_specs:
                        spec = layer_specs[base_key]
                        d_out = int(spec.get("d_out", d_out))
                        d_in = int(spec.get("d_in", d_in))
                    comm_bytes += float(rank * (d_in + d_out) * 2.0)
                    comp_time_ms += float(rank * (d_in + d_out) * 1.7e-04)
                B_down_bytes = float(budget.get("B_down_MB", 0.0)) * 1024.0 * 1024.0
                T_ms = float(budget.get("step_ms", 0.0))
                comm_util = (comm_bytes / B_down_bytes) if B_down_bytes > 0 else 0.0
                comp_util = (comp_time_ms / T_ms) if T_ms > 0 else 0.0
                logging.info(
                    "[FedAvg][epoch %d][client %s] Comm Util: %.1f%%, Comp Util: %.1f%%",
                    epoch,
                    str(client_id),
                    comm_util * 100.0,
                    comp_util * 100.0,
                )
        del single_weights
        gc.collect()
        torch.cuda.empty_cache()

    # set_peft_model_state_dict(model, weighted_single_weights, "default")
    torch.cuda.empty_cache()
    return weighted_single_weights

def FedHeLLo(selected_clients_set, output_dir, local_dataset_len_dict, epoch,
             active_layers_map=None, prev_global_params=None, layer_specs=None):
    """
    Fed-HeLLo 聚合（最小侵入版）：
    - client 侧已经通过 active_lora_layers 冻结了未分配的 LoRA 层，只保存训练过的 LoRA 参数。
    - 这里对所有上传的 LoRA 参数做“按样本数加权的 FedAvg”。
    - 某个参数 key 只在存在该 key 的客户端上做加权平均（其它客户端不参与该 key 的聚合）。
    """
    import os
    from torch.nn.functional import normalize

    # 本函数不再使用 active_layers_map / layer_specs，保留参数仅为兼容现有调用接口
    del active_layers_map
    total_layers = len(layer_specs or {})

    # 按本地样本数计算 client 级权重
    lens = [local_dataset_len_dict[client_id] for client_id in selected_clients_set]
    weights_array = normalize(
        torch.tensor(lens, dtype=torch.float32),
        p=1, dim=0
    )

    accum = {}         # key -> 加权和
    weight_sums = {}   # key -> 对应该 key 的权重之和（仅来自有该 key 的客户端）
    round_transmit_bytes = 0.0
    MB = 1024.0 * 1024.0

    with torch.no_grad():
        for idx, client_id in enumerate(selected_clients_set):
            single_output_dir = os.path.join(
                output_dir,
                str(client_id),
                f"local_output_epoch_{epoch}",
                "pytorch_model.bin",
            )
            if not os.path.exists(single_output_dir):
                continue

            state = torch.load(single_output_dir, map_location="cpu")
            w = float(weights_array[idx])

            for key, tensor in state.items():
                if key not in accum:
                    accum[key] = torch.zeros_like(tensor)
                    weight_sums[key] = 0.0
                accum[key] += tensor * w
                weight_sums[key] += w

                round_transmit_bytes += float(tensor.numel() * tensor.element_size())

            del state
            torch.cuda.empty_cache()

    # 先把上一轮的全局参数拷贝过来（主要是为了保留未出现 key 的旧值）
    aggregated = {}
    if prev_global_params is not None:
        aggregated.update({
            k: v.clone() if isinstance(v, torch.Tensor) else v
            for k, v in prev_global_params.items()
        })

    # 对每个出现过的 key，根据该 key 的权重和做加权平均
    for key, summed in accum.items():
        wsum = weight_sums.get(key, 0.0)
        if wsum > 0.0:
            aggregated[key] = summed / wsum
        else:
            # 理论上不会出现 wsum=0，如果出现就直接当作简单相加结果
            aggregated[key] = summed

    TRAFFIC_STATS["FedHeLLo"]["transmit_MB"] += round_transmit_bytes / MB
    logging.info(
        "[FedHeLLo][epoch %d] round_transmit_MB=%.3f trained_params=%d total_layers=%d",
        epoch,
        round_transmit_bytes / MB,
        len(accum),
        total_layers,
    )
    return aggregated

def truncate(selected_clients_set, output_dir, local_dataset_len_dict, epoch, handle_alpha = False):

    weights_array = torch.tensor(
        [local_dataset_len_dict[client_id] for client_id in selected_clients_set], dtype=torch.float32
    )
    weights_array = torch.nn.functional.normalize(weights_array, p=1, dim=0)
    weighted_single_weights = {}
    with torch.no_grad():
        for k, client_id in tqdm(enumerate(selected_clients_set)):
            single_output_dir = os.path.join(output_dir, str(client_id), f"local_output_epoch_{epoch}", "pytorch_model.bin")
            single_weights = torch.load(single_output_dir, map_location='cpu')
            for key in list(single_weights.keys()):
                if 'local' in key and 'bias' not in key:
                    if 'lora_A' in key:
                        B_key = key.replace('lora_A', 'lora_B')
                        rank = single_weights[B_key].shape[1]
                        if handle_alpha:
                            merge_rate = 16 / rank
                        else:
                            merge_rate = 1
                        # new_key = '.'.join(key.split('.')[:-3]) + '.lora'
                        if key not in weighted_single_weights.keys():
                            weighted_single_weights[key] = torch.zeros(360, int(single_weights[key].shape[1])).to('cpu')
                            weighted_single_weights[B_key] = torch.zeros(int(single_weights[B_key].shape[0]), 360).to('cpu')
                        weighted_single_weights[key][:rank, :] += single_weights[key] * weights_array[k] * np.sqrt(merge_rate)
                        weighted_single_weights[B_key][:, :rank] += single_weights[B_key] * weights_array[k] * np.sqrt(merge_rate)
                        torch.cuda.empty_cache()
            del single_weights
            # gc.collect()
            torch.cuda.empty_cache()
    torch.cuda.empty_cache()
    return weighted_single_weights

def FlexLoRA(selected_clients_set, output_dir, local_dataset_len_dict, epoch, client_budgets=None, layer_specs=None):
    """
    Aggregate heterogeneous LoRA adapters from clients.

    In addition to the merged weights, this function also updates
    global TRAFFIC_STATS with the communication/compute volume for
    this round. In FlexLoRA, transmit and compute data are the same
    because clients train on all transmitted ranks.
    """
    weights_array = torch.tensor(
        [local_dataset_len_dict[client_id] for client_id in selected_clients_set], dtype=torch.float32
    )
    weights_array = torch.nn.functional.normalize(weights_array, p=1, dim=0)
    weighted_single_weights = {}

    round_transmit_bytes = 0.0
    round_compute_bytes = 0.0
    compute_ms_per_param = 1.7e-04
    MB = 1024.0 * 1024.0

    with torch.no_grad():
        for k, client_id in tqdm(enumerate(selected_clients_set)):
            comm_bytes = 0.0
            comp_time_ms = 0.0
            single_output_dir = os.path.join(
                output_dir, str(client_id), f"local_output_epoch_{epoch}", "pytorch_model.bin"
            )
            single_weights = torch.load(single_output_dir, map_location='cpu')
            for key in list(single_weights.keys()):
                if 'local' in key and 'bias' not in key and 'lora_A' in key:
                    B_key = key.replace('lora_A', 'lora_B')
                    rank = single_weights[B_key].shape[1]
                    merge_rate = 16 / max(rank, 1)
                    new_key = '.'.join(key.split('.')[:-3]) + '.lora'
                    if new_key not in weighted_single_weights.keys():
                        weighted_single_weights[new_key] = 0
                    merged_weight = (single_weights[B_key] @ single_weights[key]) * merge_rate * weights_array[k]
                    weighted_single_weights[new_key] += merged_weight

                    # Track per-round communication/compute volume for this LoRA pair.
                    d_in = single_weights[key].shape[1]
                    d_out = single_weights[B_key].shape[0]
                    elem_bytes = single_weights[key].element_size()
                    bytes_this = (d_in + d_out) * rank * elem_bytes
                    round_transmit_bytes += bytes_this
                    round_compute_bytes += bytes_this  # same for FlexLoRA
                    # Track per-client comm/comp usage
                    base_key = '.'.join(key.split('.')[:-3]) + '.lora'
                    if layer_specs is not None and base_key in layer_specs:
                        spec = layer_specs[base_key]
                        d_out = int(spec.get("d_out", d_out))
                        d_in = int(spec.get("d_in", d_in))
                    comm_bytes += float(rank * (d_in + d_out) * 2.0)
                    comp_time_ms += float(rank * (d_in + d_out) * compute_ms_per_param)

                    del merged_weight
                    torch.cuda.empty_cache()
            del single_weights
            # gc.collect()
            torch.cuda.empty_cache()

            if client_budgets is not None:
                budget = client_budgets.get(int(client_id)) if isinstance(client_budgets, dict) else None
                if budget is not None:
                    B_down_bytes = float(budget.get("B_down_MB", 0.0)) * MB
                    T_ms = float(budget.get("step_ms", 0.0))
                    comm_util = (comm_bytes / B_down_bytes) if B_down_bytes > 0 else 0.0
                    comp_util = (comp_time_ms / T_ms) if T_ms > 0 else 0.0
                    logging.info(
                        "[FlexLoRA][epoch %d][client %s] Comm Util: %.1f%%, Comp Util: %.1f%%",
                        epoch,
                        str(client_id),
                        comm_util * 100.0,
                        comp_util * 100.0,
                    )

    TRAFFIC_STATS["FlexLoRA"]["transmit_MB"] += round_transmit_bytes / MB
    TRAFFIC_STATS["FlexLoRA"]["compute_MB"] += round_compute_bytes / MB
    logging.info(
        "[FlexLoRA][epoch %d] transmit_MB=%.3f compute_MB=%.3f",
        epoch,
        round_transmit_bytes / MB,
        round_compute_bytes / MB,
    )

    torch.cuda.empty_cache()
    return weighted_single_weights


# fed_utils/model_aggregation.py (append)

import json
from .rank_allocator import allocate_r_tot_for_client, allocate_r_main_for_client
def _fedhera_dense_global_path(output_dir: str, epoch: int) -> str:
    return os.path.join(output_dir, f"fedhera_dense_global_epoch_{epoch}.pt")

def _load_fedhera_dense_global(output_dir: str, epoch: int):
    path = _fedhera_dense_global_path(output_dir, epoch)
    if os.path.exists(path):
        return torch.load(path, map_location="cpu")
    return None

def _save_fedhera_dense_global(output_dir: str, epoch: int, dense_dict: dict):
    path = _fedhera_dense_global_path(output_dir, epoch)
    torch.save({k: v.detach().cpu() for k, v in dense_dict.items()}, path)


def FedHera(selected_clients_set, output_dir, local_dataset_len_dict, epoch,
            client_budgets,
            layer_specs,
            quant_scheme=("fp16", "nf4"),
            use_gpu_svd=False,
            basis_update_every=5,
            fixed_client_ranks=None,
            ablation=None,
            lora_alpha=16,
            use_atw=False,
            atw_temperature=2.0, 
            all_client_ids=None, 
            server_agg: str = "original", 
            prev_global_params=None,
            profile_metrics=None,
            force_coupled=False):
    """
    Fed-Hera aggregation:
    1) Merge client adapters into W_global.
    2) [ATW] Compute alignment scores s_i and update cache.
    3) Run SVD to refresh basis.
    4) Allocate r_tot/r_main per client.
    5) [ATW] Calculate lambda and Push truncated A/B plus meta back to clients.
    """
    compute_ms_per_param = 1.7e-04
    
    def _uniform_allocation(layers, bytes_per_col, c_mem_per_col, c_time_per_col,
                            B_down_bytes, M_bytes, T_ms, target_rank=None):
        total_bytes = max(sum(bytes_per_col.values()), 1)
        uniform_cap = B_down_bytes // total_bytes
        if target_rank is not None and target_rank > 0:
            uniform_cap = min(uniform_cap, target_rank)
        r_tot = {L: int(min(uniform_cap, len(meta["sigma"]))) for L, meta in layers.items()}
        mem_denom = max(sum(c_mem_per_col.values()), 1)
        time_denom = max(sum(c_time_per_col.values()), 1)
        r_main_cap = int(min(uniform_cap, M_bytes // mem_denom, T_ms // time_denom))
        r_main = {L: int(min(rt, r_main_cap)) for L, rt in r_tot.items()}
        return r_tot, r_main

    def _random_allocation(layers, bytes_per_col, c_mem_per_col, c_time_per_col,
                           B_down_bytes, M_bytes, T_ms, target_rank=None, rng=None):
        rng = rng or np.random.default_rng()
        caps = {
            L: int(min(len(meta["sigma"]), target_rank)) if target_rank else int(len(meta["sigma"]))
            for L, meta in layers.items()
        }
        r_tot = {L: 0 for L in layers}
        remaining = B_down_bytes
        layer_keys = list(layers.keys())
        while True:
            candidates = [k for k in layer_keys if r_tot[k] < caps[k] and remaining >= bytes_per_col[k]]
            if not candidates:
                break
            choice = rng.choice(candidates)
            r_tot[choice] += 1
            remaining -= bytes_per_col[choice]
        mem_denom = max(sum(c_mem_per_col.values()), 1)
        time_denom = max(sum(c_time_per_col.values()), 1)
        r_main_cap = int(min(M_bytes // mem_denom, T_ms // time_denom))
        r_main = {L: int(min(rt, r_main_cap)) for L, rt in r_tot.items()}
        return r_tot, r_main

    weights_array = torch.tensor([local_dataset_len_dict[c] for c in selected_clients_set], dtype=torch.float32)
    weights_array = torch.nn.functional.normalize(weights_array, p=1, dim=0)
    if profile_metrics is not None and use_gpu_svd and torch.cuda.is_available():
        torch.cuda.synchronize()
    server_pipeline_start = time.perf_counter() if profile_metrics is not None else None

    # 1. Aggregation Phase
    server_agg_mode = str(server_agg or "original").lower()
    if server_agg_mode not in ["original", "unbiased"]:
        raise ValueError(f"Unsupported FedHera server_agg mode: {server_agg_mode}")

    prev_dense = None
    if server_agg_mode == "unbiased":
        if prev_global_params is not None:
            prev_dense = prev_global_params
        elif epoch > 0:
            prev_dense = _load_fedhera_dense_global(output_dir, epoch - 1)
            if prev_dense is None:
                logging.warning(
                    "[FedHera][epoch %d] Missing dense cache for epoch %d; fallback to original.",
                    epoch, epoch - 1
                )
                server_agg_mode = "original"
        else:
            logging.info("[FedHera][epoch 0] unbiased requires prev dense cache; fallback to original.")
            server_agg_mode = "original"

    if server_agg_mode == "unbiased":
        assert prev_dense is not None, "prev_dense must exist in unbiased mode"


    with torch.no_grad():
        if server_agg_mode == "original":
            aggregated = {}
            for k, client_id in tqdm(enumerate(selected_clients_set)):
                local_path = os.path.join(output_dir, str(client_id),
                                        f"local_output_epoch_{epoch}", "pytorch_model.bin")
                state = torch.load(local_path, map_location="cpu")

                for key in list(state.keys()):
                    if ('local' in key) and ('bias' not in key) and ('lora_A' in key):
                        B_key = key.replace('lora_A', 'lora_B')
                        rank = int(state[B_key].shape[1])
                        merge_rate = float(lora_alpha) / float(max(rank, 1))  # 修正：别写死16
                        base_key = '.'.join(key.split('.')[:-3]) + '.lora'
                        merged = (state[B_key] @ state[key]) * merge_rate * weights_array[k]
                        aggregated[base_key] = aggregated.get(base_key, 0) + merged.to("cpu")

                del state
                torch.cuda.empty_cache()

        else:
            # unbiased: W_{t+1} = W_t + Σ p_i (W_i^{t+1} - W_t^{r_i})
            delta_sum = {}
            for k, client_id in tqdm(enumerate(selected_clients_set)):
                local_path = os.path.join(output_dir, str(client_id),
                                        f"local_output_epoch_{epoch}", "pytorch_model.bin")
                init_path = os.path.join(output_dir, str(client_id),
                                        f"server_push_epoch_{epoch - 1}", "pytorch_model.bin")

                local_state = torch.load(local_path, map_location="cpu")
                init_state = torch.load(init_path, map_location="cpu") if os.path.exists(init_path) else {}

                for key in list(local_state.keys()):
                    if ('local' in key) and ('bias' not in key) and ('lora_A' in key):
                        B_key = key.replace('lora_A', 'lora_B')
                        rank = int(local_state[B_key].shape[1])
                        merge_rate = float(lora_alpha) / float(max(rank, 1))
                        base_key = '.'.join(key.split('.')[:-3]) + '.lora'
                        merged_local = (local_state[B_key] @ local_state[key]) * merge_rate

                        A_init_key = base_key + "_A.local.weight"
                        B_init_key = base_key + "_B.local.weight"
                        if (A_init_key in init_state) and (B_init_key in init_state):
                            r_init = int(init_state[B_init_key].shape[1])
                            merge_rate_init = float(lora_alpha) / float(max(r_init, 1))
                            merged_init = (init_state[B_init_key] @ init_state[A_init_key]) * merge_rate_init
                        else:
                            merged_init = torch.zeros_like(merged_local)

                        delta = (merged_local - merged_init) * weights_array[k]
                        delta_sum[base_key] = delta_sum.get(base_key, 0) + delta.to("cpu")

                del local_state, init_state
                torch.cuda.empty_cache()

            aggregated = {k: v.clone().to("cpu") for k, v in prev_dense.items()}
            for base_key, d in delta_sum.items():
                aggregated[base_key] = aggregated.get(base_key, 0) + d


    # 2. [ATW Logic] Compute s_i and update cache
    if use_atw:
        logging.info("[FedHera] Computing ATW alignment scores with Trace Optimization...")

        # 1. 计算全局 Global Norm
        global_sq_norm = 0.0
        for g_tensor in aggregated.values():
            global_sq_norm += torch.linalg.norm(g_tensor.float()) ** 2
        global_norm = math.sqrt(global_sq_norm)

        push_client_ids = all_client_ids if all_client_ids is not None else selected_clients_set
        for client_id in push_client_ids:
            single_output = os.path.join(output_dir, str(client_id), f"local_output_epoch_{epoch}", "pytorch_model.bin")
            if not os.path.exists(single_output):
                continue
                
            state = torch.load(single_output, map_location="cpu")
            dot_product = 0.0
            client_sq_norm = 0.0
            
            for base_key, g_tensor in aggregated.items():
                prefix = base_key.rsplit('.lora', 1)[0]
                key_A = None
                key_B = None
                
                # 寻找对应的 A 和 B
                for k in state.keys():
                    if prefix in k and 'lora_A' in k:
                        key_A = k
                        key_B = k.replace('lora_A', 'lora_B')
                        break
                
                if key_A and key_B:
                    A_mat = state[key_A].float() # (r, k)
                    B_mat = state[key_B].float() # (d, r)
                    G_mat = g_tensor.float()     # (d, k)
                    
                    rank = B_mat.shape[1]
                    merge_rate = 16 / max(rank, 1)
                    
                    # --- 优化1: Dot Product ---
                    # 计算 <BA, G> = Tr(A^T B^T G)
                    # 先算 (B^T G) -> (r, k)
                    temp_res = B_mat.T @ G_mat 
                    # 再算 sum(A * temp_res)
                    contribution = torch.sum(A_mat * temp_res).item()
                    dot_product += contribution * merge_rate
                    
                    # --- 优化2: Client Norm ---
                    # 计算 ||BA||^2 = Tr((B^T B)(A A^T))
                    # 这一步避免了生成 (d, k) 的大矩阵，全程只处理 (r, r) 的小矩阵
                    BT_B = B_mat.T @ B_mat   # (r, r)
                    A_AT = A_mat @ A_mat.T   # (r, r)
                    trace_norm = torch.sum(BT_B * A_AT).item()
                    
                    client_sq_norm += trace_norm * (merge_rate ** 2)

            client_norm = math.sqrt(client_sq_norm)
            
            # Cosine Sim
            if global_norm > 1e-6 and client_norm > 1e-6:
                s_i = dot_product / (global_norm * client_norm)
            else:
                s_i = 0.0
            
            # Update Cache
            FEDHERA_CLIENT_STATS[int(client_id)] = {"s": s_i, "t": epoch}
            del state

    # 3. SVD Phase
    basis_version = epoch // max(basis_update_every, 1)
    per_layer_USV = {}
    for layer_key in list(aggregated.keys()):
        Wg = aggregated[layer_key]
        device = "cuda" if (use_gpu_svd and torch.cuda.is_available()) else "cpu"
        W = Wg.to(device=device, dtype=torch.float32)
        U, S, Vh = torch.linalg.svd(W, full_matrices=False)
        per_layer_USV[layer_key] = {
            "U": U.to("cpu"),
            "S": S.to("cpu"),
            "Vh": Vh.to("cpu"),
            "sigma": S.detach().cpu().numpy(),
        }
        del Wg, W, U, S, Vh
        torch.cuda.empty_cache()

    gc.collect()

    # 4. Allocation & Push Phase
    quant_main, quant_res = quant_scheme
    BYTE_MAP = {"fp16": 2, "bfloat16": 2, "int8": 1, "nf4": 0.5, "int4": 0.5}
    bytes_down_main = BYTE_MAP.get(quant_main, 2.0)
    MB = 1024.0 * 1024.0
    rng = np.random.default_rng()
    round_transmit_bytes = 0.0
    round_compute_bytes = 0.0

    push_target_ids = all_client_ids if all_client_ids is not None else selected_clients_set
    for client_id in push_target_ids:
        budgets = client_budgets[int(client_id)]
        B_down_bytes = int(budgets["B_down_MB"] * MB)
        M_bytes = int(budgets["VRAM_MB"] * MB)
        T_ms = float(budgets["step_ms"])
        target_rank = None
        if fixed_client_ranks is not None:
            target_rank = int(fixed_client_ranks.get(int(client_id), 0))
            if target_rank < 0:
                target_rank = 0

        layers = {}
        bytes_per_col = {}
        c_mem_per_col = {}
        c_time_per_col = {}
        for layer_key, usv in per_layer_USV.items():
            if layer_specs is not None and layer_key in layer_specs:
                spec = layer_specs[layer_key]
                d_out, d_in = spec["d_out"], spec["d_in"]
            else:
                d_out = int(usv["U"].shape[0])
                d_in = int(usv["Vh"].shape[1])
            sigma = usv["sigma"]
            layers[layer_key] = {"sigma": sigma, "d_out": d_out, "d_in": d_in}
            bytes_per_col[layer_key] = int((d_out + d_in) * bytes_down_main)
            c_mem_per_col[layer_key] = int((d_out + d_in) * 2.0 * 3.5)
            compute_ms_per_param = 1.7e-04  
            c_time_per_col[layer_key] = float((d_out + d_in) * compute_ms_per_param)

        if ablation == "uniform":
            r_tot, r_main = _uniform_allocation(layers, bytes_per_col, c_mem_per_col, c_time_per_col,
                                                B_down_bytes, M_bytes, T_ms, target_rank)
        elif ablation == "random":
            r_tot, r_main = _random_allocation(layers, bytes_per_col, c_mem_per_col, c_time_per_col,
                                               B_down_bytes, M_bytes, T_ms, target_rank, rng)
        else:
            if target_rank is not None and target_rank > 0:
                r_tot = {L: min(target_rank, len(meta["sigma"])) for L, meta in layers.items()}
            else:
                r_tot, _ = allocate_r_tot_for_client(layers, B_down_bytes, bytes_per_col)
            r_main, _, _ = allocate_r_main_for_client(layers, r_tot, M_bytes, T_ms, c_mem_per_col, c_time_per_col)

        if force_coupled:
            r_tot = {
                layer_key: int(min(r_tot.get(layer_key, 0), r_main.get(layer_key, 0)))
                for layer_key in layers.keys()
            }
            r_main = {
                layer_key: int(r_tot.get(layer_key, 0))
                for layer_key in layers.keys()
            }

        # [ATW Logic] Calculate Lambda for this client
        lambda_val = 1.0 # Default Static
        if use_atw:
            # Retrieve cached stats
            stat = FEDHERA_CLIENT_STATS.get(int(client_id), {"s": 0.0, "t": -1})
            s_stored = stat["s"]
            t_hat = stat["t"]
            
            beta = 0.9 # Hardcoded per requirement
            
            # Note: For selected clients, t_hat == epoch because we just updated it.
            # So decay factor beta^(epoch - epoch) == 1. This uses the fresh score.
            # The formula works for stale clients too if we were to serve them.
            
            current_round = epoch + 1
            decay = beta ** (epoch - t_hat)
            
            # Formula: 1 - exp( - (1 + s * beta^(t-that)) * t / T_warmup )
            exponent = -1.0 * (1.0 + s_stored * decay) * current_round / atw_temperature
            lambda_val = 1.0 - math.exp(exponent)
            
            # Clip for safety
            lambda_val = max(0.0, min(1.0, lambda_val))
            
            if client_id == sorted(list(selected_clients_set))[0]: # Log once per round
                logging.info(f"[ATW] Client {client_id}: s={s_stored:.4f}, lambda={lambda_val:.4f}")

        push_dir = os.path.join(output_dir, str(client_id), f"server_push_epoch_{epoch}")
        os.makedirs(push_dir, exist_ok=True)
        pkg = {}
        meta = {}
        ablation_mode = ablation if ablation is not None else "water_filling"

        for layer_key, usv in per_layer_USV.items():
            rt = int(r_tot.get(layer_key, 0))
            if rt <= 0:
                meta[layer_key] = {"skip": True, "basis_version": basis_version, "r_tot": 0, "r_main": 0,
                                   "ablation": ablation_mode}
                continue
            
            U = usv["U"][:, :rt]
            S = usv["S"][:rt]
            Vh = usv["Vh"][:rt, :]
            
            scale_up = float(rt) / float(max(lora_alpha, 1e-12))
            S = S * scale_up
            sroot = torch.sqrt(S)
            B = (U * sroot.unsqueeze(0))
            A = (sroot.unsqueeze(1) * Vh)
            
            Akey = layer_key + "_A.local.weight"
            Bkey = layer_key + "_B.local.weight"
            pkg[Akey] = A.float().cpu()
            pkg[Bkey] = B.float().cpu()
            
            meta[layer_key] = {
                "skip": False,
                "basis_version": basis_version,
                "r_tot": rt,
                "r_main": int(r_main.get(layer_key, 0)),
                "quant_main": quant_main,
                "quant_res": quant_res,
                "ablation": ablation_mode,
                "lambda": lambda_val # Write lambda to meta
            }
            del U, S, Vh, B, A
        
        torch.save(pkg, os.path.join(push_dir, "pytorch_model.bin"))
        with open(os.path.join(push_dir, "meta.json"), "w") as f:
            json.dump(meta, f)
        torch.cuda.empty_cache()

        # Track communication/compute stats...
        client_transmit_bytes = 0.0
        client_compute_bytes = 0.0
        comp_time_ms = 0.0
        rank_summary = {}
        for layer_key, info in meta.items():
            if info.get("skip", False):
                continue
            rt = int(info.get("r_tot", 0))
            rm = int(info.get("r_main", 0))
            rank_summary[layer_key] = {"r_tot": rt, "r_main": rm}

            d_out = d_in = None
            if layer_specs and layer_key in layer_specs:
                spec = layer_specs[layer_key]
                d_out = spec.get("d_out")
                d_in = spec.get("d_in")
            # Fallback to shapes from the stored tensors if specs are missing.
            if (d_out is None or d_in is None) and pkg:
                Akey = layer_key + "_A.local.weight"
                Bkey = layer_key + "_B.local.weight"
                if d_out is None and Bkey in pkg:
                    d_out = int(pkg[Bkey].shape[0])
                if d_in is None and Akey in pkg:
                    d_in = int(pkg[Akey].shape[1])
            if d_out is None or d_in is None:
                continue

            elem_bytes = bytes_down_main
            bytes_per_rank = (d_out + d_in) * elem_bytes
            client_transmit_bytes += bytes_per_rank * rt
            client_compute_bytes += bytes_per_rank * rm
            comp_time_ms += float(rm * (d_in + d_out) * compute_ms_per_param)

        if rank_summary:
            rank_summary_top = dict(list(rank_summary.items())[:2])
            logging.info("[FedHera][epoch %d][client %s] ranks(top)=%s", epoch, str(client_id), rank_summary_top)
        if force_coupled:
            logging.info("[FedHera][epoch %d][client %s] coupled_mode=True (r_tot=r_main)", epoch, str(client_id))
        comm_util = (client_transmit_bytes / float(B_down_bytes)) if B_down_bytes > 0 else 0.0
        comp_util = (comp_time_ms / float(T_ms)) if T_ms > 0 else 0.0
        logging.info(
            "[FedHera][epoch %d][client %s] Comm Util: %.1f%%, Comp Util: %.1f%%",
            epoch,
            str(client_id),
            comm_util * 100.0,
            comp_util * 100.0,
        )
        round_transmit_bytes += client_transmit_bytes
        round_compute_bytes += client_compute_bytes

    TRAFFIC_STATS["FedHera"]["transmit_MB"] += round_transmit_bytes / MB
    TRAFFIC_STATS["FedHera"]["compute_MB"] += round_compute_bytes / MB
    logging.info(
        "[FedHera][epoch %d] round_transmit_MB=%.3f round_compute_MB=%.3f",
        epoch,
        round_transmit_bytes / MB,
        round_compute_bytes / MB,
    )
    if profile_metrics is not None:
        if use_gpu_svd and torch.cuda.is_available():
            torch.cuda.synchronize()
        server_pipeline_ms = (time.perf_counter() - server_pipeline_start) * 1000.0
        profile_metrics.setdefault("server_svd_time_ms_per_round", []).append(float(server_pipeline_ms))
        logging.info(
            "[SystemProfile][FedHera][epoch %d] server_svd_pipeline_ms=%.3f",
            epoch,
            server_pipeline_ms,
        )
    aggregated = {k: v.detach().cpu() for k, v in aggregated.items()}
    _save_fedhera_dense_global(output_dir, epoch, aggregated)
    return aggregated


def FedHL(selected_clients_set, output_dir, local_dataset_len_dict, epoch, prev_global_params, layer_specs=None):
    """
    FedHL: Federated Learning for Heterogeneous LoRA via Unbiased Aggregation.
    Paper: arXiv:2505.18494v1

    Formula: W_{t+1} = W_t + sum( p_i * (W_{client_i} - W_t^{rank_i}) )
    Where p_i is optimized based on truncation error.
    """
    logging.info(f"[FedHL] Starting aggregation for epoch {epoch}")
    
    # 1. 预计算全局模型 W_t 的 SVD 信息 (用于计算 truncation error 和 W_t^{r_i})
    # prev_global_params 是 W_t (Dense weights)
    # 我们按层缓存 SVD 结果： {layer_key: (U, S, Vh, S_squared_sum)}
    layer_svd_cache = {}
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # 筛选出需要 LoRA 的层 (即存在于 global_params 中的层)
    # 注意：prev_global_params 通常是完整的 state_dict 或仅包含 adapter 对应的 dense weight
    # 这里假设 prev_global_params 是 dense weight 字典
    
    # 2. 第一遍循环：收集所有客户端的 Rank 信息，计算 Truncation Error (\hat{r}_i)
    client_ranks = {}      # client_id -> {layer_key: rank}
    client_errors = {}     # client_id -> total_truncation_error
    client_updates_cache = {} # 缓存加载的客户端模型，避免重复 IO
    
    epsilon = 1e-6 # 防止除零
    
    for client_id in tqdm(selected_clients_set, desc="[FedHL] Computing Errors"):
        single_output_dir = os.path.join(output_dir, str(client_id), f"local_output_epoch_{epoch}", "pytorch_model.bin")
        if not os.path.exists(single_output_dir):
            continue
            
        single_weights = torch.load(single_output_dir, map_location='cpu')
        client_updates_cache[client_id] = single_weights
        
        current_client_ranks = {}
        total_error = 0.0
        
        for key in single_weights.keys():
            # 识别 LoRA A 矩阵
            if 'local' in key and 'lora_A' in key:
                base_key = '.'.join(key.split('.')[:-3]) + '.lora'
                B_key = key.replace('lora_A', 'lora_B')
                
                # 获取 Client Rank
                rank = single_weights[key].shape[0] # A is [r, d_in]
                current_client_ranks[base_key] = rank
                
                # 确保全局模型中有该层，并计算 SVD
                if base_key not in prev_global_params:
                    # 可能是新层或者命名不匹配，暂时跳过
                    continue
                
                if base_key not in layer_svd_cache:
                    # 计算 SVD 并缓存
                    W_t_layer = prev_global_params[base_key].to(device).float()
                    # 使用 full_matrices=False 以节省显存
                    U, S, Vh = torch.linalg.svd(W_t_layer, full_matrices=False)
                    layer_svd_cache[base_key] = (U.cpu(), S.cpu(), Vh.cpu())
                    del W_t_layer
                    torch.cuda.empty_cache()
                
                # 计算 Truncation Error: || W_t - W_t^r ||^2
                # 等价于 sum(sigma_{r+1}^2 ... sigma_{end}^2)
                _, S, _ = layer_svd_cache[base_key]
                # S 是奇异值向量，从大到小排列
                truncated_singular_values = S[rank:]
                layer_error = torch.sum(truncated_singular_values ** 2).item()
                total_error += layer_error
                
        client_ranks[client_id] = current_client_ranks
        client_errors[client_id] = total_error

    # 3. 计算最优聚合权重 p_i (Theorem 3)
    # p_i* = (1 / (error_i^2 + eps)) / sum(...)
    # 注意：论文中 error 定义为 \hat{r}_i = ||...||^2，所以这里直接用 total_error
    
    p_numerators = {}
    p_denom_sum = 0.0
    
    for cid in selected_clients_set:
        err = client_errors.get(cid, 0.0)
        # 论文公式 (15): p_i = (1 / (r_i^2 + epsilon)) ... 这里的 r_i 指的是 truncation error \hat{r}_i
        # 因为我们上面算的 total_error 已经是 ||...||^2 了，所以这里直接用 total_error
        # 但论文写的是 \hat{r}_i^2，我们需要确认 \hat{r}_i 是定义为范数还是范数平方。
        # 论文 Eq(8): \hat{r}_i(t) = || W - W^r ||^2. 
        # Theorem 3 Eq(15) 分母是 \hat{r}_i^2 + epsilon. 这里的 \hat{r}_i 指代上面的定义。
        # 所以是 (Error)^2.
        
        val = 1.0 / ( (err ** 2) + epsilon )
        p_numerators[cid] = val
        p_denom_sum += val
        
    optimal_weights = {cid: val / p_denom_sum for cid, val in p_numerators.items()}
    
    # 记录权重信息
    log_weights = {cid: f"{w:.4f}" for cid, w in optimal_weights.items()}
    logging.info(f"[FedHL] Aggregation Weights: {str(log_weights)}")

    # 4. 执行聚合: W_{t+1} = W_t + sum( p_i * (W_{t+1}^i - W_t^{r_i}) )
    # 初始化 Delta accumulator
    global_delta = {k: torch.zeros_like(v, device='cpu') for k, v in prev_global_params.items()}
    
    for client_id in tqdm(selected_clients_set, desc="[FedHL] Aggregating"):
        weight = optimal_weights[client_id]
        single_weights = client_updates_cache[client_id]
        ranks = client_ranks.get(client_id, {})
        
        for key in single_weights.keys():
            if 'local' in key and 'lora_A' in key:
                base_key = '.'.join(key.split('.')[:-3]) + '.lora'
                B_key = key.replace('lora_A', 'lora_B')
                
                if base_key not in layer_svd_cache:
                    continue
                    
                # 1. 重构客户端模型 W_{t+1}^i = B @ A
                B_mat = single_weights[B_key].float()
                A_mat = single_weights[key].float()
                
                rank = ranks[base_key]
                # 注意：Adaptive PEFT 代码中，forward 包含 scaling (alpha/r)。
                # 但 distribute_weight_fast 通常把 scaling 并在 B 或 A 里。
                # 检查 FlexLoRA/FLoRA 实现，通常需要在聚合时乘 merge_rate。
                # 在本代码库的 FL_training 中，distribute_weight_fast 使用 weight_dict[..._B...] = lora_B / merge_rate
                # 意味着存储的权重被除过了，forward 时乘回来。
                # 这里我们需要恢复其数学上的值用于加减。
                merge_rate = 16.0 / max(rank, 1) # assuming alpha=16
                
                # W_client = (B @ A) * scale
                W_client = (B_mat @ A_mat) * merge_rate
                
                # 2. 重构初始截断模型 W_t^{r_i} = U_r S_r V_r^T
                U, S, Vh = layer_svd_cache[base_key]
                U_r = U[:, :rank]
                S_r = S[:rank]
                Vh_r = Vh[:rank, :]
                
                W_truncated = (U_r @ torch.diag(S_r) @ Vh_r)
                
                # 3. 计算差异 Delta = p_i * (W_client - W_truncated)
                diff = (W_client - W_truncated) * weight
                
                # 累加到 Global Delta
                if base_key in global_delta:
                    global_delta[base_key] += diff
                
    # 5. 更新全局模型: W_{t+1} = W_t + Delta
    new_global_params = {}
    for k, v in prev_global_params.items():
        if k in global_delta:
            # 关键：使用 .detach().clone() 确保新 Tensor 是 leaf tensor，且内存独立
            # 将 W_t + Delta 的结果剥离出计算图
            updated_tensor = (v + global_delta[k]).detach().clone()
            new_global_params[k] = updated_tensor
        else:
            # 对于没有更新的参数，也进行 detach clone 以防万一
            new_global_params[k] = v.detach().clone()
            
    # 清理缓存
    del layer_svd_cache
    del client_updates_cache
    gc.collect()
    torch.cuda.empty_cache()
    
    return new_global_params
