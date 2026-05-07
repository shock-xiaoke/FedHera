import gc
import json
import logging
import math
import os
import time

import numpy as np
import torch
from tqdm import tqdm

from .rank_allocator import allocate_r_main_for_client, allocate_r_tot_for_client


TRAFFIC_STATS = {
    "FedHera": {"transmit_MB": 0.0, "compute_MB": 0.0},
}

FEDHERA_CLIENT_STATS = {}


def reset_traffic_stats():
    for method in TRAFFIC_STATS:
        TRAFFIC_STATS[method]["transmit_MB"] = 0.0
        TRAFFIC_STATS[method]["compute_MB"] = 0.0


def get_traffic_stats():
    return {name: stats.copy() for name, stats in TRAFFIC_STATS.items()}


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


def FedHera(
    selected_clients_set,
    output_dir,
    local_dataset_len_dict,
    epoch,
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
    server_agg="original",
    prev_global_params=None,
    profile_metrics=None,
    force_coupled=False,
):
    """
    FedHera aggregation.

    1. Merge client adapters into a dense global update.
    2. Optionally compute ATW alignment scores.
    3. Refresh the global basis with SVD.
    4. Allocate per-client total/trainable ranks under resource budgets.
    5. Push truncated adapters and metadata back to clients.
    """
    compute_ms_per_param = 1.7e-04

    def _uniform_allocation(
        layers,
        bytes_per_col,
        c_mem_per_col,
        c_time_per_col,
        b_down_bytes,
        m_bytes,
        t_ms,
        target_rank=None,
    ):
        total_bytes = max(sum(bytes_per_col.values()), 1)
        uniform_cap = b_down_bytes // total_bytes
        if target_rank is not None and target_rank > 0:
            uniform_cap = min(uniform_cap, target_rank)
        r_tot = {layer: int(min(uniform_cap, len(meta["sigma"]))) for layer, meta in layers.items()}
        mem_denom = max(sum(c_mem_per_col.values()), 1)
        time_denom = max(sum(c_time_per_col.values()), 1)
        r_main_cap = int(min(uniform_cap, m_bytes // mem_denom, t_ms // time_denom))
        r_main = {layer: int(min(rt, r_main_cap)) for layer, rt in r_tot.items()}
        return r_tot, r_main

    def _random_allocation(
        layers,
        bytes_per_col,
        c_mem_per_col,
        c_time_per_col,
        b_down_bytes,
        m_bytes,
        t_ms,
        target_rank=None,
        rng=None,
    ):
        rng = rng or np.random.default_rng()
        caps = {
            layer: int(min(len(meta["sigma"]), target_rank)) if target_rank else int(len(meta["sigma"]))
            for layer, meta in layers.items()
        }
        r_tot = {layer: 0 for layer in layers}
        remaining = b_down_bytes
        layer_keys = list(layers.keys())
        while True:
            candidates = [
                layer
                for layer in layer_keys
                if r_tot[layer] < caps[layer] and remaining >= bytes_per_col[layer]
            ]
            if not candidates:
                break
            choice = rng.choice(candidates)
            r_tot[choice] += 1
            remaining -= bytes_per_col[choice]

        mem_denom = max(sum(c_mem_per_col.values()), 1)
        time_denom = max(sum(c_time_per_col.values()), 1)
        r_main_cap = int(min(m_bytes // mem_denom, t_ms // time_denom))
        r_main = {layer: int(min(rt, r_main_cap)) for layer, rt in r_tot.items()}
        return r_tot, r_main

    if not selected_clients_set:
        logging.warning("[FedHera][epoch %d] No selected clients.", epoch)
        return prev_global_params if prev_global_params is not None else {}

    weights_array = torch.tensor(
        [local_dataset_len_dict[client_id] for client_id in selected_clients_set],
        dtype=torch.float32,
    )
    weights_array = torch.nn.functional.normalize(weights_array, p=1, dim=0)

    if profile_metrics is not None and use_gpu_svd and torch.cuda.is_available():
        torch.cuda.synchronize()
    server_pipeline_start = time.perf_counter() if profile_metrics is not None else None

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
                    epoch,
                    epoch - 1,
                )
                server_agg_mode = "original"
        else:
            logging.info("[FedHera][epoch 0] unbiased requires a previous dense cache; fallback to original.")
            server_agg_mode = "original"

    if server_agg_mode == "unbiased":
        assert prev_dense is not None, "prev_dense must exist in unbiased mode"

    with torch.no_grad():
        if server_agg_mode == "original":
            aggregated = {}
            for idx, client_id in tqdm(enumerate(selected_clients_set), total=len(selected_clients_set)):
                local_path = os.path.join(
                    output_dir,
                    str(client_id),
                    f"local_output_epoch_{epoch}",
                    "pytorch_model.bin",
                )
                state = torch.load(local_path, map_location="cpu")

                for key in list(state.keys()):
                    if "local" not in key or "bias" in key or "lora_A" not in key:
                        continue
                    b_key = key.replace("lora_A", "lora_B")
                    rank = int(state[b_key].shape[1])
                    merge_rate = float(lora_alpha) / float(max(rank, 1))
                    base_key = ".".join(key.split(".")[:-3]) + ".lora"
                    merged = (state[b_key] @ state[key]) * merge_rate * weights_array[idx]
                    aggregated[base_key] = aggregated.get(base_key, 0) + merged.to("cpu")

                del state
                torch.cuda.empty_cache()
        else:
            delta_sum = {}
            for idx, client_id in tqdm(enumerate(selected_clients_set), total=len(selected_clients_set)):
                local_path = os.path.join(
                    output_dir,
                    str(client_id),
                    f"local_output_epoch_{epoch}",
                    "pytorch_model.bin",
                )
                init_path = os.path.join(
                    output_dir,
                    str(client_id),
                    f"server_push_epoch_{epoch - 1}",
                    "pytorch_model.bin",
                )

                local_state = torch.load(local_path, map_location="cpu")
                init_state = torch.load(init_path, map_location="cpu") if os.path.exists(init_path) else {}

                for key in list(local_state.keys()):
                    if "local" not in key or "bias" in key or "lora_A" not in key:
                        continue
                    b_key = key.replace("lora_A", "lora_B")
                    rank = int(local_state[b_key].shape[1])
                    merge_rate = float(lora_alpha) / float(max(rank, 1))
                    base_key = ".".join(key.split(".")[:-3]) + ".lora"
                    merged_local = (local_state[b_key] @ local_state[key]) * merge_rate

                    a_init_key = base_key + "_A.local.weight"
                    b_init_key = base_key + "_B.local.weight"
                    if a_init_key in init_state and b_init_key in init_state:
                        r_init = int(init_state[b_init_key].shape[1])
                        merge_rate_init = float(lora_alpha) / float(max(r_init, 1))
                        merged_init = (init_state[b_init_key] @ init_state[a_init_key]) * merge_rate_init
                    else:
                        merged_init = torch.zeros_like(merged_local)

                    delta = (merged_local - merged_init) * weights_array[idx]
                    delta_sum[base_key] = delta_sum.get(base_key, 0) + delta.to("cpu")

                del local_state, init_state
                torch.cuda.empty_cache()

            aggregated = {key: value.clone().to("cpu") for key, value in prev_dense.items()}
            for base_key, delta in delta_sum.items():
                aggregated[base_key] = aggregated.get(base_key, 0) + delta

    if use_atw:
        logging.info("[FedHera] Computing ATW alignment scores.")
        global_sq_norm = 0.0
        for tensor in aggregated.values():
            global_sq_norm += torch.linalg.norm(tensor.float()) ** 2
        global_norm = math.sqrt(global_sq_norm)

        push_client_ids = all_client_ids if all_client_ids is not None else selected_clients_set
        for client_id in push_client_ids:
            single_output = os.path.join(
                output_dir,
                str(client_id),
                f"local_output_epoch_{epoch}",
                "pytorch_model.bin",
            )
            if not os.path.exists(single_output):
                continue

            state = torch.load(single_output, map_location="cpu")
            dot_product = 0.0
            client_sq_norm = 0.0

            for base_key, global_tensor in aggregated.items():
                prefix = base_key.rsplit(".lora", 1)[0]
                key_a = None
                key_b = None

                for state_key in state.keys():
                    if prefix in state_key and "lora_A" in state_key:
                        key_a = state_key
                        key_b = state_key.replace("lora_A", "lora_B")
                        break

                if key_a is None or key_b is None:
                    continue

                a_mat = state[key_a].float()
                b_mat = state[key_b].float()
                g_mat = global_tensor.float()

                rank = b_mat.shape[1]
                merge_rate = float(lora_alpha) / float(max(rank, 1))

                temp_res = b_mat.T @ g_mat
                contribution = torch.sum(a_mat * temp_res).item()
                dot_product += contribution * merge_rate

                bt_b = b_mat.T @ b_mat
                a_at = a_mat @ a_mat.T
                trace_norm = torch.sum(bt_b * a_at).item()
                client_sq_norm += trace_norm * (merge_rate ** 2)

            client_norm = math.sqrt(client_sq_norm)
            if global_norm > 1e-6 and client_norm > 1e-6:
                s_i = dot_product / (global_norm * client_norm)
            else:
                s_i = 0.0

            FEDHERA_CLIENT_STATS[int(client_id)] = {"s": s_i, "t": epoch}
            del state

    basis_version = epoch // max(basis_update_every, 1)
    per_layer_usv = {}
    for layer_key, dense_weight in aggregated.items():
        device = "cuda" if (use_gpu_svd and torch.cuda.is_available()) else "cpu"
        weight = dense_weight.to(device=device, dtype=torch.float32)
        u_mat, s_val, v_h = torch.linalg.svd(weight, full_matrices=False)
        per_layer_usv[layer_key] = {
            "U": u_mat.to("cpu"),
            "S": s_val.to("cpu"),
            "Vh": v_h.to("cpu"),
            "sigma": s_val.detach().cpu().numpy(),
        }
        del weight, u_mat, s_val, v_h
        torch.cuda.empty_cache()

    gc.collect()

    quant_main, quant_res = quant_scheme
    byte_map = {"fp16": 2, "bfloat16": 2, "int8": 1, "nf4": 0.5, "int4": 0.5}
    bytes_down_main = byte_map.get(quant_main, 2.0)
    mb = 1024.0 * 1024.0
    rng = np.random.default_rng()
    round_transmit_bytes = 0.0
    round_compute_bytes = 0.0

    push_target_ids = all_client_ids if all_client_ids is not None else selected_clients_set
    for client_id in push_target_ids:
        budgets = client_budgets[int(client_id)]
        b_down_bytes = int(budgets["B_down_MB"] * mb)
        m_bytes = int(budgets["VRAM_MB"] * mb)
        t_ms = float(budgets["step_ms"])

        target_rank = None
        if fixed_client_ranks is not None:
            target_rank = int(fixed_client_ranks.get(int(client_id), 0))
            if target_rank < 0:
                target_rank = 0

        layers = {}
        bytes_per_col = {}
        c_mem_per_col = {}
        c_time_per_col = {}
        for layer_key, usv in per_layer_usv.items():
            if layer_specs is not None and layer_key in layer_specs:
                spec = layer_specs[layer_key]
                d_out = spec["d_out"]
                d_in = spec["d_in"]
            else:
                d_out = int(usv["U"].shape[0])
                d_in = int(usv["Vh"].shape[1])
            sigma = usv["sigma"]
            layers[layer_key] = {"sigma": sigma, "d_out": d_out, "d_in": d_in}
            bytes_per_col[layer_key] = int((d_out + d_in) * bytes_down_main)
            c_mem_per_col[layer_key] = int((d_out + d_in) * 2.0 * 3.5)
            c_time_per_col[layer_key] = float((d_out + d_in) * compute_ms_per_param)

        if ablation == "uniform":
            r_tot, r_main = _uniform_allocation(
                layers,
                bytes_per_col,
                c_mem_per_col,
                c_time_per_col,
                b_down_bytes,
                m_bytes,
                t_ms,
                target_rank,
            )
        elif ablation == "random":
            r_tot, r_main = _random_allocation(
                layers,
                bytes_per_col,
                c_mem_per_col,
                c_time_per_col,
                b_down_bytes,
                m_bytes,
                t_ms,
                target_rank,
                rng,
            )
        else:
            if target_rank is not None and target_rank > 0:
                r_tot = {layer: min(target_rank, len(meta["sigma"])) for layer, meta in layers.items()}
            else:
                r_tot, _ = allocate_r_tot_for_client(layers, b_down_bytes, bytes_per_col)
            r_main, _, _ = allocate_r_main_for_client(
                layers,
                r_tot,
                m_bytes,
                t_ms,
                c_mem_per_col,
                c_time_per_col,
            )

        if force_coupled:
            r_tot = {
                layer_key: int(min(r_tot.get(layer_key, 0), r_main.get(layer_key, 0)))
                for layer_key in layers.keys()
            }
            r_main = {layer_key: int(r_tot.get(layer_key, 0)) for layer_key in layers.keys()}

        lambda_val = 1.0
        if use_atw:
            stat = FEDHERA_CLIENT_STATS.get(int(client_id), {"s": 0.0, "t": -1})
            s_stored = stat["s"]
            t_hat = stat["t"]
            beta = 0.9
            current_round = epoch + 1
            decay = beta ** (epoch - t_hat)
            exponent = -1.0 * (1.0 + s_stored * decay) * current_round / atw_temperature
            lambda_val = 1.0 - math.exp(exponent)
            lambda_val = max(0.0, min(1.0, lambda_val))
            if client_id == sorted(list(selected_clients_set))[0]:
                logging.info("[ATW] Client %s: s=%.4f, lambda=%.4f", client_id, s_stored, lambda_val)

        push_dir = os.path.join(output_dir, str(client_id), f"server_push_epoch_{epoch}")
        os.makedirs(push_dir, exist_ok=True)
        pkg = {}
        meta = {}
        ablation_mode = ablation if ablation is not None else "water_filling"

        for layer_key, usv in per_layer_usv.items():
            rt = int(r_tot.get(layer_key, 0))
            if rt <= 0:
                meta[layer_key] = {
                    "skip": True,
                    "basis_version": basis_version,
                    "r_tot": 0,
                    "r_main": 0,
                    "ablation": ablation_mode,
                }
                continue

            u_mat = usv["U"][:, :rt]
            s_val = usv["S"][:rt]
            v_h = usv["Vh"][:rt, :]

            scale_up = float(rt) / float(max(lora_alpha, 1e-12))
            s_val = s_val * scale_up
            s_root = torch.sqrt(s_val)
            b_mat = u_mat * s_root.unsqueeze(0)
            a_mat = s_root.unsqueeze(1) * v_h

            a_key = layer_key + "_A.local.weight"
            b_key = layer_key + "_B.local.weight"
            pkg[a_key] = a_mat.float().cpu()
            pkg[b_key] = b_mat.float().cpu()

            meta[layer_key] = {
                "skip": False,
                "basis_version": basis_version,
                "r_tot": rt,
                "r_main": int(r_main.get(layer_key, 0)),
                "quant_main": quant_main,
                "quant_res": quant_res,
                "ablation": ablation_mode,
                "lambda": lambda_val,
            }

        torch.save(pkg, os.path.join(push_dir, "pytorch_model.bin"))
        with open(os.path.join(push_dir, "meta.json"), "w", encoding="utf-8") as f:
            json.dump(meta, f)
        torch.cuda.empty_cache()

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
            if (d_out is None or d_in is None) and pkg:
                a_key = layer_key + "_A.local.weight"
                b_key = layer_key + "_B.local.weight"
                if d_out is None and b_key in pkg:
                    d_out = int(pkg[b_key].shape[0])
                if d_in is None and a_key in pkg:
                    d_in = int(pkg[a_key].shape[1])
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

        comm_util = (client_transmit_bytes / float(b_down_bytes)) if b_down_bytes > 0 else 0.0
        comp_util = (comp_time_ms / float(t_ms)) if t_ms > 0 else 0.0
        logging.info(
            "[FedHera][epoch %d][client %s] Comm Util: %.1f%%, Comp Util: %.1f%%",
            epoch,
            str(client_id),
            comm_util * 100.0,
            comp_util * 100.0,
        )
        round_transmit_bytes += client_transmit_bytes
        round_compute_bytes += client_compute_bytes

    TRAFFIC_STATS["FedHera"]["transmit_MB"] += round_transmit_bytes / mb
    TRAFFIC_STATS["FedHera"]["compute_MB"] += round_compute_bytes / mb
    logging.info(
        "[FedHera][epoch %d] round_transmit_MB=%.3f round_compute_MB=%.3f",
        epoch,
        round_transmit_bytes / mb,
        round_compute_bytes / mb,
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

    aggregated = {key: value.detach().cpu() for key, value in aggregated.items()}
    _save_fedhera_dense_global(output_dir, epoch, aggregated)
    return aggregated
