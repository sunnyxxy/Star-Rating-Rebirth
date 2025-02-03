import osu_file_parser as osu_parser
from collections import defaultdict
import numpy as np
import heapq
import matplotlib.pyplot as plt
import pandas as pd
import bisect
import math

# --- Helper functions
def cumulative_sum(x, f):
    """
    Given sorted positions x (length N) and function values f defined piecewise constant on [x[i], x[i+1]),
    return an array F of cumulative integrals such that F[0]=0 and for i>=1:
      F[i] = sum_{j=0}^{i-1} f[j]*(x[j+1]-x[j])
    """
    F = np.zeros(len(x))
    for i in range(1, len(x)):
        F[i] = F[i-1] + f[i-1]*(x[i] - x[i-1])
    return F

def query_cumsum(q, x, F, f):
    """
    Given cumulative data (x, F, f) as above, return the cumulative sum at an arbitrary point q.
    Here we assume that f is constant on each interval.
    """
    if q <= x[0]:
        return 0.0
    if q >= x[-1]:
        return F[-1]
    # Find index i such that x[i] <= q < x[i+1]
    i = np.searchsorted(x, q) - 1
    return F[i] + f[i]*(q - x[i])

def smooth_on_corners(x, f, window, scale=1.0, mode='sum'):
    """
    Given positions x (a sorted 1D array) and function values f (piecewise constant on intervals defined by x),
    return an array g defined at x by applying a symmetric sliding window:
      if mode=='sum': g(s) = scale * ∫[s-window, s+window] f(t) dt
      if mode=='avg': g(s) = (∫[s-window, s+window] f(t) dt) / (length of window actually used)
    This is computed exactly using the cumulative–sum technique.
    """
    F = cumulative_sum(x, f)
    g = np.empty_like(f)
    for i, s in enumerate(x):
        a = max(s - window, x[0])
        b = min(s + window, x[-1])
        val = query_cumsum(b, x, F, f) - query_cumsum(a, x, F, f)
        if mode == 'avg':
            g[i] = val / (b - a) if (b - a) > 0 else 0.0
        else:
            g[i] = scale * val
    return g

# --- Helper for linear interpolation from one set of sample points to another ---
def interp_values(new_x, old_x, old_vals):
    """Return new_vals at positions new_x using linear interpolation from old_x, old_vals."""
    return np.interp(new_x, old_x, old_vals)
    
def step_interp(new_x, old_x, old_vals):
    """
    For each position in new_x, return the value of old_vals corresponding to the greatest old_x
    that is less than or equal to new_x. This implements a step–function (zero–order hold)
    interpolation.
    """
    indices = np.searchsorted(old_x, new_x, side='right') - 1
    indices = np.clip(indices, 0, len(old_vals)-1)
    return old_vals[indices]
    
# --- Main function ---
def calculate(file_path, mod, lambda_2, lambda_4, w_0, w_1, p_1, w_2, p_0):
    # === Basic Setup and Parsing ===
    lambda_n = 5
    lambda_1 = 0.11
    lambda_3 = 24
    p_obj = osu_parser.parser(file_path)
    p_obj.process()
    p = p_obj.get_parsed_data()
    
    # Build note_seq as a list of tuples (column, head_time, tail_time)
    note_seq = []
    for i in range(len(p[1])):
        k = p[1][i]
        h = p[2][i]
        # Only set tail_time when p[4]==128; otherwise use -1.
        t = p[3][i] if p[4][i] == 128 else -1
        if mod == "DT":
            h = int(math.floor(h * 2/3))
            t = int(math.floor(t * 2/3)) if t >= 0 else t
        elif mod == "HT":
            h = int(math.floor(h * 4/3))
            t = int(math.floor(t * 4/3)) if t >= 0 else t
        note_seq.append((k, h, t))
    
    # Global scaling factor x
    x = 0.3 * ((64.5 - math.ceil(p[5] * 3)) / 500)**0.5
    note_seq.sort(key=lambda tup: (tup[1], tup[0]))
    
    # Group notes by column
    note_dict = defaultdict(list)
    for tup in note_seq:
        note_dict[tup[0]].append(tup)
    note_seq_by_column = sorted(list(note_dict.values()), key=lambda lst: lst[0][0])
    
    # Long notes (LN) are those with a tail (t>=0)
    LN_seq = [n for n in note_seq if n[2] >= 0]
    tail_seq = sorted(LN_seq, key=lambda tup: tup[2])
    
    LN_dict = defaultdict(list)
    for tup in LN_seq:
        LN_dict[tup[0]].append(tup)
    LN_seq_by_column = sorted(list(LN_dict.values()), key=lambda lst: lst[0][0])
    
    K = p[0]
    T = max( max(n[1] for n in note_seq),
             max(n[2] for n in note_seq)) + 1

    # === Determine Corner Times for “base” variables and for A ===
    # For Jbar, Xbar, Pbar, Rbar, C, and Ks, unsmoothed step functions change only at note boundaries.
    corners_base = set()
    for (_, h, t) in note_seq:
        corners_base.add(h)
        if t >= 0:
            corners_base.add(t)
    for s in list(corners_base):
        corners_base.add(s + 501)
        corners_base.add(s - 499)
        corners_base.add(s + 1) # To resolve the Dirac-Delta additions exactly at notes
    corners_base.add(0)
    corners_base.add(T)
    corners_base = sorted(s for s in corners_base if 0 <= s <= T)
    
    # For Abar, unsmoothed values (KU and A) may change at ±500 relative to note boundaries, hence ±1000 overall.
    corners_A = set()
    for (_, h, t) in note_seq:
        corners_A.add(h)
        if t >= 0:
            corners_A.add(t)
    for s in list(corners_A):
        corners_A.add(s + 1000)
        corners_A.add(s - 1000)
    corners_A.add(0)
    corners_A.add(T)
    corners_A = sorted(s for s in corners_A if 0 <= s <= T)
    
    # Finally, take the union of all corners for final interpolation
    all_corners = sorted(set(corners_base) | set(corners_A))
    all_corners = np.array(all_corners, dtype=float)
    base_corners = np.array(corners_base, dtype=float)
    A_corners = np.array(corners_A, dtype=float)
    
    # === Section 2.3: Compute Jbar ===
    # For each column, the unsmoothed “J” is constant on segments between consecutive notes.
    def jackNerfer(delta):
        return 1 - 7e-5 * (0.15 + abs(delta - 0.08))**(-4)
    
    # For each column k, we compute a step function on base_corners.
    J_ks = {k: np.zeros(len(base_corners)) for k in range(K)}
    delta_ks = {k: np.full(len(base_corners), 1e9) for k in range(K)}
    for k in range(K):
        notes = note_seq_by_column[k]
        for i in range(len(notes) - 1):
            start = notes[i][1]
            end = notes[i+1][1]
            # Find indices in base_corners that lie in [start, end)
            idx = np.where((base_corners >= start) & (base_corners < end))[0]
            if len(idx) == 0:
                continue
            delta = 0.001 * (end - start)
            val = (delta**(-1)) * (delta + lambda_1 * x**(1/4))**(-1)
            J_val = val * jackNerfer(delta)
            J_ks[k][idx] = J_val
            delta_ks[k][idx] = delta
        # For any base corner not assigned (i.e. still 0), leave J_ks as 0 and delta as 1e9.
    
    # Now smooth each column’s J_ks using a sliding window of ±500 and scale 0.001.
    Jbar_ks = {}
    for k in range(K):
        Jbar_ks[k] = smooth_on_corners(base_corners, J_ks[k], window=500, scale=0.001, mode='sum')
    
    # Now aggregate across columns using weighted average
    Jbar_base = np.empty(len(base_corners))
    for i, s in enumerate(base_corners):
        vals = [Jbar_ks[k][i] for k in range(K)]
        weights = [1 / delta_ks[k][i] for k in range(K)]
        # Use the lambda_n–power average as in the original code:
        num = sum((max(v, 0) ** lambda_n) * w for v, w in zip(vals, weights))
        den = sum(weights)
        Jbar_base[i] = num / max(1e-9, den)
        Jbar_base[i] = Jbar_base[i]**(1/lambda_n)
    
    # Interpolate Jbar from base_corners to all_corners:
    Jbar = interp_values(all_corners, base_corners, Jbar_base)
    
    # === Section 2.4: Compute Xbar ===
    # X_ks is computed for k=0,...,K from merged note sequences.
    cross_matrix = [
        [-1],
        [0.075, 0.075],
        [0.125, 0.05, 0.125],
        [0.125, 0.125, 0.125, 0.125],
        [0.175, 0.25, 0.05, 0.25, 0.175],
        [0.175, 0.25, 0.175, 0.175, 0.25, 0.175],
        [0.225, 0.35, 0.25, 0.05, 0.25, 0.35, 0.225],
        [0.225, 0.35, 0.25, 0.225, 0.225, 0.25, 0.35, 0.225],
        [0.275, 0.45, 0.35, 0.25, 0.05, 0.25, 0.35, 0.45, 0.275],
        [0.275, 0.45, 0.35, 0.25, 0.275, 0.275, 0.25, 0.35, 0.45, 0.275],
        [0.325, 0.55, 0.45, 0.35, 0.25, 0.05, 0.25, 0.35, 0.45, 0.55, 0.325]
    ]
    X_ks = {k: np.zeros(len(base_corners)) for k in range(K+1)}
    for k in range(K+1):
        if k == 0:
            notes_in_pair = note_seq_by_column[0]
        elif k == K:
            notes_in_pair = note_seq_by_column[K-1]
        else:
            notes_in_pair = list(heapq.merge(note_seq_by_column[k-1], note_seq_by_column[k], key=lambda tup: tup[1]))
        for i in range(1, len(notes_in_pair)):
            start = notes_in_pair[i-1][1]
            end = notes_in_pair[i][1]
            idx = np.where((base_corners >= start) & (base_corners < end))[0]
            if len(idx) == 0:
                continue
            delta = 0.001 * (notes_in_pair[i][1] - notes_in_pair[i-1][1])
            val = 0.16 * max(x, delta)**(-2)
            X_ks[k][idx] = val
    # Combine using cross_matrix coefficients for column K:
    cross_coeff = cross_matrix[K]
    X_base = np.zeros(len(base_corners))
    for i in range(len(base_corners)):
        X_base[i] = sum(X_ks[k][i] * cross_coeff[k] for k in range(K+1))
    # Smooth X_base with the same ±500, scale 0.001.
    Xbar_base = smooth_on_corners(base_corners, X_base, window=500, scale=0.001, mode='sum')
    Xbar = interp_values(all_corners, base_corners, Xbar_base)
    
    # === Section 2.5: Compute Pbar ===
    # Build LN_bodies exactly as in the original.
    LN_bodies = np.zeros(T)
    for (k, h, t) in LN_seq:
        t1 = min(h + 80, t)
        LN_bodies[h:t1] += 0.5
        LN_bodies[t1:t] += 1
    
    # Compute the cumulative sum for LN_bodies.
    # cumsum_LN[i] = sum(LN_bodies[0] ... LN_bodies[i-1])
    cumsum_LN = np.concatenate(([0], np.cumsum(LN_bodies)))
    
    def LN_sum(a, b):
        """Return the exact sum over LN_bodies[a:b]."""
        return cumsum_LN[b] - cumsum_LN[a]
    
    def b_func(delta):
        """Implements the Dirac delta multiplier function.
           For a given delta, if 7.5/delta lies in (160, 360), adjust the value."""
        val = 7.5 / delta
        if 160 < val < 360:
            return 1 + 1.7e-7 * (val - 160) * (val - 360)**2
        return 1
    
    # Prepare the unsmoothed step function for P on the base grid.
    P_step = np.zeros(len(base_corners))
    for i in range(len(note_seq) - 1):
        h_l = note_seq[i][1]
        h_r = note_seq[i+1][1]
        delta_time = h_r - h_l
        if delta_time < 1e-9:
            # Dirac delta case: when notes occur at the same time.
            # Add the spike exactly at the note head in the base grid.
            spike = 1000 * (0.02 * (4 / x - lambda_3))**(1/4)
            idx = np.where(base_corners == h_l)[0]
            if len(idx) > 0:
                P_step[idx] += spike
            # Continue so that we add a spike for each adjacent note pair at the same time.
            continue
        # For the regular case where delta_time > 0, identify the base grid indices in [h_l, h_r)
        idx = np.where((base_corners >= h_l) & (base_corners < h_r))[0]
        if len(idx) == 0:
            continue
        delta = 0.001 * delta_time
        # v is an amplification factor based on LN_bodies over [h_l, h_r)
        v = 1 + lambda_2 * 0.001 * LN_sum(h_l, h_r)
        # Multiply by the b(delta) factor
        b_val = b_func(delta)
        if delta < 2 * x / 3:
            inc = delta**(-1) * (0.08 * x**(-1) * (1 - lambda_3 * x**(-1) * (delta - x/2)**2))**(1/4) * b_val * v
        else:
            inc = delta**(-1) * (0.08 * x**(-1) * (1 - lambda_3 * x**(-1) * (x/6)**2))**(1/4) * b_val * v
        P_step[idx] += inc
    
    # Smooth the unsmoothed P_step over a ±500 window (using cumulative–sum integration).
    Pbar_base = smooth_on_corners(base_corners, P_step, window=500, scale=0.001, mode='sum')
    # Interpolate from the base grid to the overall grid.
    Pbar = interp_values(all_corners, base_corners, Pbar_base)

    # === Section 2.6: Compute Abar ===
    # For A, the unsmoothed value depends on local key usage.
    # Mark for each column k, at times in base_corners we record whether that key is “active”
    KU_ks = {k: np.zeros(len(base_corners), dtype=bool) for k in range(K)}
    for (k, h, t) in note_seq:
        # For each note, active from max(0, h-500) to (h+500) (or t+500 if t>=0)
        startTime = max(h - 500, 0)
        endTime = (h + 500) if t < 0 else min(t + 500, T-1)
        idx = np.where((base_corners >= startTime) & (base_corners < endTime))[0]
        KU_ks[k][idx] = True
    # At each time in base_corners, build a list of columns that are active:
    KU_s_cols = [ [k for k in range(K) if KU_ks[k][i]] for i in range(len(base_corners)) ]
    
    # Also compute a “difference” measure dks between adjacent columns.
    dks = {k: np.zeros(len(base_corners)) for k in range(K-1)}
    for i in range(len(base_corners)):
        cols = KU_s_cols[i]
        for j in range(len(cols) - 1):
            k0 = cols[j]
            k1 = cols[j+1]
            # Use the delta_ks computed before on base_corners
            dks[k0][i] = abs(delta_ks[k0][i] - delta_ks[k1][i]) + 0.4*max(0, max(delta_ks[k0][i], delta_ks[k1][i]) - 0.11)
    
    # Now compute A (unsmoothed) on A_corners (because the active regions for A vary on ±500; hence Abar becomes
    # piecewise linear with corners at ±1000 relative to note boundaries).
    A_step = np.ones(len(A_corners))
    # We “lift” A_step by scanning through A_corners: for each A_corners value, use the same logic as above,
    # but we need to know which columns are active at time s.
    # We use linear interpolation from the base_corners’ KU info.
    # (Here we simply approximate: we compute KU at time s by checking nearest base corner.)
    for i, s in enumerate(A_corners):
        # Find the nearest index in base_corners:
        idx = np.searchsorted(base_corners, s)
        if idx >= len(base_corners):
            idx = len(base_corners) - 1
        cols = KU_s_cols[idx]
        for j in range(len(cols) - 1):
            k0 = cols[j]
            k1 = cols[j+1]
            d_val = dks[k0][idx]
            if d_val < 0.02:
                A_step[i] *= min(0.75 + 0.5 * max(delta_ks[k0][idx], delta_ks[k1][idx]), 1)
            elif d_val < 0.07:
                A_step[i] *= min(0.65 + 5*d_val + 0.5 * max(delta_ks[k0][idx], delta_ks[k1][idx]), 1)
            # Otherwise leave A_step[i] unchanged.
    # Smooth A_step with average smoothing (smooth2) using ±500:
    Abar_A = smooth_on_corners(A_corners, A_step, window=500, mode='avg')
    Abar = interp_values(all_corners, A_corners, Abar_A)
    
    # === Section 2.7: Compute Rbar ===
    # Compute I and R on base_corners from tail_seq.
    I_arr = np.zeros(len(base_corners))
    R_step = np.zeros(len(base_corners))
    # For each pair of successive long notes (tail_seq), assign I and R on the interval [t_i, t_{i+1})
    def find_next_note_in_column(note, note_seq_by_column):
        k, h, t = note
        times = [n[1] for n in note_seq_by_column[k]]
        idx = bisect.bisect_left(times, h)
        return note_seq_by_column[k][idx+1] if idx+1 < len(note_seq_by_column[k]) else (0, 10**9, 10**9)
    
    I_list = []
    for i in range(len(tail_seq)):
        k, h_i, t_i = tail_seq[i]
        _, h_j, _ = find_next_note_in_column((k, h_i, t_i), note_seq_by_column)
        I_h = 0.001 * abs(t_i - h_i - 80) / x
        I_t = 0.001 * abs(h_j - t_i - 80) / x
        I_list.append(2 / (2 + math.exp(-5*(I_h-0.75)) + math.exp(-5*(I_t-0.75))))
    
    # For each interval between successive tail times, assign I and R.
    for i in range(len(tail_seq)-1):
        t_start = tail_seq[i][2]
        t_end = tail_seq[i+1][2]
        idx = np.where((base_corners >= t_start) & (base_corners < t_end))[0]
        if len(idx) == 0:
            continue
        I_arr[idx] = 1 + I_list[i]
        delta_r = 0.001 * (tail_seq[i+1][2] - tail_seq[i][2])
        R_step[idx] = 0.08 * (delta_r)**(-0.5) * x**(-1) * (1 + lambda_4*(I_list[i] + I_list[i+1]))
    Rbar_base = smooth_on_corners(base_corners, R_step, window=500, scale=0.001, mode='sum')
    Rbar = interp_values(all_corners, base_corners, Rbar_base)
    
    # === Section 3: Compute C and Ks ===
    # C(s): count of notes whose hit time lies in [s-500, s+500).
    note_hit_times = sorted(n[1] for n in note_seq)
    C_step = np.zeros(len(base_corners))
    for i, s in enumerate(base_corners):
        low = s - 500
        high = s + 500
        # Use binary search on note_hit_times:
        cnt = bisect.bisect_left(note_hit_times, high) - bisect.bisect_left(note_hit_times, low)
        C_step[i] = cnt
    C_base = C_step  # already a step function on base_corners.

    C_arr = step_interp(all_corners, base_corners, C_base)
    
    # Ks: local key usage count (minimum 1)
    Ks_step = np.array([max(sum(1 for k in range(K) if KU_ks[k][i]), 1) for i in range(len(base_corners))])
    Ks_base = Ks_step
    Ks_arr = step_interp(all_corners, base_corners, Ks_base)
    
    # === Final Computations (nonlinear combination) ===
    # Compute S, T, and D on all_corners:
    A_bar = Abar  # from above (from A_corners, now on all_corners)
    J_bar = Jbar
    X_bar = Xbar
    P_bar = Pbar
    R_bar = Rbar
    Ks_val = Ks_arr  # local key count
    # Compute S and T as in your original formula:
    S_all = ((w_0 * (A_bar**(3/ Ks_val) * J_bar)**1.5) + 
             ((1-w_0) * (A_bar**(2/3) * (0.8*P_bar + R_bar))**1.5))**(2/3)
    T_all = (A_bar**(3/ Ks_val) * X_bar) / (X_bar + S_all + 1)
    D_all = w_1 * (S_all**0.5) * (T_all**p_1) + S_all * w_2
    
    # Now perform the weighted–percentile calculation on D_all using C_arr as weights.
    # First, sort by D_all:
    df_corners = pd.DataFrame({
        'time': all_corners,
        'Jbar': J_bar,
        'Xbar': X_bar,
        'Pbar': P_bar,
        'Abar': A_bar,
        'Rbar': R_bar,
        'C': C_arr,
        'Ks': Ks_arr,
        'D': D_all
    })

    times = df_corners['time'].values  # array of corner times
    C_vals = df_corners['C'].values      # local density at each corner
    
    # Compute the gaps between consecutive times in a vectorised way.
    # For interior points, the effective gap is the average of the left and right gap.
    gaps = np.empty_like(times, dtype=float)
    gaps[0] = (times[1] - times[0]) / 2.0
    gaps[-1] = (times[-1] - times[-2]) / 2.0
    gaps[1:-1] = (times[2:] - times[:-2]) / 2.0
    
    # The effective weight for each corner is the product of its density and its gap.
    effective_weights = C_vals * gaps
    
    # Now, sort df_corners by D.
    df_sorted = df_corners.sort_values('D')
    D_sorted = df_sorted['D'].values
    # Retrieve the effective weights corresponding to the sorted indices.
    # (Using .loc ensures we are aligning by the index from the original df_corners.)
    sorted_indices = df_sorted.index.to_numpy()
    w_sorted = effective_weights[sorted_indices]
    
    # Compute the cumulative sum of the effective weights.
    cum_weights = np.cumsum(w_sorted)
    total_weight = cum_weights[-1]
    norm_cum_weights = cum_weights / total_weight
    
    target_percentiles = np.array([0.945, 0.935, 0.925, 0.915, 0.845, 0.835, 0.825, 0.815])
    
    indices = np.searchsorted(norm_cum_weights, target_percentiles, side='left')
    
    if len(indices) >= 8:
        percentile_93 = np.mean(D_sorted[indices[:4]])
        percentile_83 = np.mean(D_sorted[indices[4:8]])
    else:
        percentile_93 = np.mean(D_sorted)
        percentile_83 = percentile_93
    
    weighted_mean = (np.sum(D_sorted**lambda_n * w_sorted) / np.sum(w_sorted))**(1 / lambda_n)
    
    # Final SR calculation
    SR = (0.88 * percentile_93) * 0.25 + (0.94 * percentile_83) * 0.2 + weighted_mean * 0.55
    SR = SR**(p_0) / (8**p_0) * 8
    
    total_notes = len(note_seq) + 0.5 * len(LN_seq)
    SR *= total_notes / (total_notes + 60)
    if SR <= 2:
        SR = (SR * 2)**0.5
        
    SR *= 0.97
    return SR
    
    # --- Optionally: interpolate all variables from all_corners to full resolution ---
    # full_range = np.arange(T, dtype=float)
    # df_full = pd.DataFrame({
    #     'Jbar': interp_values(full_range, all_corners, J_bar),
    #     'Xbar': interp_values(full_range, all_corners, X_bar),
    #     'Pbar': interp_values(full_range, all_corners, P_bar),
    #     'Abar': interp_values(full_range, all_corners, A_bar),
    #     'Rbar': interp_values(full_range, all_corners, R_bar),
    #     'C':    step_interp(full_range, all_corners, C_arr),
    #     'Ks':   step_interp(full_range, all_corners, Ks_arr)
    # })
    # df_full = df_full.clip(lower=0)
    # # Recompute S, T, D at full resolution if desired:
    # df_full['S'] = ((w_0 * (df_full['Abar']**(3/df_full['Ks']) * df_full['Jbar'])**1.5) +
    #                 (1-w_0) * (df_full['Abar']**(2/3) * (0.8*df_full['Pbar'] + df_full['Rbar']))**1.5)**(2/3)
    # df_full['T'] = (df_full['Abar']**(3/df_full['Ks'])*df_full['Xbar'])/(df_full['Xbar']+df_full['S']+1)
    # df_full['D'] = w_1*df_full['S']**0.5*df_full['T']**p_1+df_full['S']*w_2
    
    # return (SR, df_full)
