# fed_utils/rank_allocator.py
import math
import numpy as np

def allocate_r_tot_for_client(layers, B_down_bytes, bytes_per_col):
    """
    Corrected allocation strategy for FedHera download phase.
    """
    r_tot = {L: 0 for L in layers}
    cost_used = 0
    
    layer_energy = {}
    for L, meta in layers.items():
        sigma = meta.get("sigma", np.array([]))
        layer_energy[L] = sigma.astype(float)**2

    while True:
        best_gain = -1.0
        best_L = None
        found_candidate = False
        
        for L, energy_array in layer_energy.items():
            r = r_tot[L]
            
            if r >= len(energy_array):
                continue
            
            cost_increase = bytes_per_col.get(L, 1.0)
            
            if cost_used + cost_increase > B_down_bytes:
                continue 
            
            current_energy = energy_array[r]
            gain = current_energy / max(cost_increase, 1e-9)
            
            if gain > best_gain:
                best_gain = gain
                best_L = L
                found_candidate = True
    
        if not found_candidate:
            break
        
        r_tot[best_L] += 1
        cost_used += bytes_per_col.get(best_L, 1.0)
        
    return r_tot, cost_used

def allocate_r_main_for_client(layers, r_tot, M_bytes, T_ms, c_mem_per_col, c_time_per_col, alpha=None, beta=None):
    """
    Fixed allocation strategy for FedHera.
    """
    r_main = {L: 0 for L in layers}
    M_left, T_left = float(M_bytes), float(T_ms)
    
    layer_energy_map = {}
    for L, meta in layers.items():
        sigma = meta.get("sigma", np.array([]))
        sigma_sq = sigma.astype(float)**2
        layer_energy_map[L] = sigma_sq

    while True:
        current_T_ratio = max(T_left, 1e-9) / max(T_ms, 1e-9)
        current_M_ratio = max(M_left, 1e-9) / max(M_bytes, 1e-9)
        
        invT = 1.0 / current_T_ratio
        invM = 1.0 / current_M_ratio

        # [Fix 3: correct the parentheses precedence bug]
        sum_inv = invT + invM
        a = (invT / sum_inv) if alpha is None else alpha
        b = (invM / sum_inv) if beta  is None else beta

        best = None
        
        for L, sigma_sq in layer_energy_map.items():
            r = r_main[L]
            limit = min(r_tot.get(L, 0), len(sigma_sq))
            
            if r >= limit:
                continue

            cost_t = c_time_per_col.get(L, 1.0)
            cost_m = c_mem_per_col.get(L, 1.0)
            
            if (M_left < cost_m) or (T_left < cost_t):
                continue

            unit_cost = a * cost_t + b * cost_m
            
            gain = sigma_sq[r] / max(unit_cost, 1e-12)

            if (best is None) or (gain > best[0]):
                best = (gain, L, cost_m, cost_t)

        if best is None:
            break

        _, Lbest, cost_m, cost_t = best
        
        r_main[Lbest] += 1
        M_left -= cost_m
        T_left -= cost_t

    return r_main, (M_bytes - M_left), (T_ms - T_left)
