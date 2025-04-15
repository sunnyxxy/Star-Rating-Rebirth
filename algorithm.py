import osu_file_parser as osu_parser
from collections import defaultdict
import numpy as np
import heapq
import pandas as pd
import bisect
import math

# -----Start of Helper methods--------

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
    
def rescale_high(sr):
    if sr <= 9:
        return sr
    return 9 + (sr - 9) * (1 / 1.2)

def find_next_note_in_column(note, times, note_seq_by_column):
    k, h, t = note
    idx = bisect.bisect_left(times, h)
    return note_seq_by_column[k][idx+1] if idx+1 < len(note_seq_by_column[k]) else (0, 10**9, 10**9)
    
# -----End of Helper methods--------

def preprocess_file(file_path, mod):
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
    
    # Hit leniency x
    x = 0.3 * ((64.5 - math.ceil(p[5] * 3)) / 500)**0.5
    x = min(x, 0.6*(x-0.09)+0.09)
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
    
    return x, K, T, note_seq, note_seq_by_column, LN_seq, tail_seq, LN_seq_by_column

def get_corners(T, note_seq):
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
    
    # For Abar, unsmoothed values (KU and A) usually change at ±500 relative to note boundaries, hence ±1000 overall.
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
    return all_corners, base_corners, A_corners
    
def get_key_usage(K, T, note_seq, base_corners):
    key_usage = {k: np.zeros(len(base_corners), dtype=bool) for k in range(K)}
    for (k, h, t) in note_seq:
        startTime = max(h - 150, 0)
        endTime = (h + 150) if t < 0 else min(t + 150, T-1)
        left_idx = np.searchsorted(base_corners, startTime, side='left')
        right_idx = np.searchsorted(base_corners, endTime, side='left')
        idx = np.arange(left_idx, right_idx)
        key_usage[k][idx] = True
    return key_usage

def get_key_usage_400(K, T, note_seq, base_corners):
    key_usage_400 = {k: np.zeros(len(base_corners), dtype=float) for k in range(K)}
    for (k, h, t) in note_seq:
        startTime = max(h, 0)
        endTime = h if t < 0 else min(t, T-1)
        left400_idx = np.searchsorted(base_corners, startTime - 400, side='left')
        left_idx = np.searchsorted(base_corners, startTime, side='left')
        right_idx = np.searchsorted(base_corners, endTime, side='left')
        right400_idx = np.searchsorted(base_corners, endTime + 400, side='left')
        idx = np.arange(left_idx, right_idx)
        key_usage_400[k][idx] += 3.75 + np.minimum(endTime - startTime, 1500)/150
        idx = np.arange(left400_idx, left_idx)
        key_usage_400[k][idx] += 3.75 - 3.75/400**2*(base_corners[idx] - np.array(startTime))**2
        idx = np.arange(right_idx, right400_idx)
        key_usage_400[k][idx] += 3.75 - 3.75/400**2*np.abs(base_corners[idx] - np.array(endTime))**2
    return key_usage_400

def compute_anchor(K, key_usage_400, base_corners):
    anchor = np.zeros(len(base_corners))
    for idx in range(len(base_corners)):
        # Collect the counts for each group at this base corner
        counts = np.array([key_usage_400[k][idx] for k in range(K)])
        counts[::-1].sort() # e.g.  8, 5, 2, 2, 0
        nonzero_counts = counts[counts != 0]
        if nonzero_counts.size > 1:
            walk = np.sum(nonzero_counts[:-1]*(1-4*(0.5-nonzero_counts[1:]/nonzero_counts[:-1])**2))
            max_walk = np.sum(nonzero_counts[:-1])
            anchor[idx] = walk/max_walk
        else:
            anchor[idx] = 0
    anchor = 1 + np.minimum(anchor-0.18, 5*(anchor-0.22)**3)
    return anchor

def LN_bodies_count_sparse_representation(LN_seq, T):
    diff = {}  # dictionary: index -> change in LN_bodies (before transformation)
    
    for (k, h, t) in LN_seq:
        t0 = min(h + 60, t)
        t1 = min(h + 120, t)
        diff[t0] = diff.get(t0, 0) + 1.3
        diff[t1] = diff.get(t1, 0) + (-1.3 + 1)  # net change at t1: -1.3 from first part, then +1
        diff[t]  = diff.get(t, 0) - 1
    
    # The breakpoints are the times where changes occur.
    points = sorted(set([0, T] + list(diff.keys())))
        
    # Build piecewise constant values (after transformation) and a cumulative sum.
    values = []
    cumsum = [0]  # cumulative sum at the breakpoints
    curr = 0.0
    
    for i in range(len(points) - 1):
        t = points[i]
        # If there is a change at t, update the running value.
        if t in diff:
            curr += diff[t]

        v = min(curr, 2.5 + 0.5 * curr)
        values.append(v)
        # Compute cumulative sum on the interval [points[i], points[i+1])
        seg_length = points[i+1] - points[i]
        cumsum.append(cumsum[-1] + seg_length * v)
    return points, cumsum, values

def LN_sum(a, b, LN_rep):
    points, cumsum, values = LN_rep
    # Locate the segments that contain a and b.
    i = bisect.bisect_right(points, a) - 1
    j = bisect.bisect_right(points, b) - 1
    
    total = 0.0
    if i == j:
        # Both a and b lie in the same segment.
        total = (b - a) * values[i]
    else:
        # First segment: from a to the end of the i-th segment.
        total += (points[i+1] - a) * values[i]
        # Full segments between i+1 and j-1.
        total += cumsum[j] - cumsum[i+1]
        # Last segment: from start of segment j to b.
        total += (b - points[j]) * values[j]
    return total
        
def compute_Jbar(K, T, x, note_seq_by_column, base_corners):
    J_ks = {k: np.zeros(len(base_corners)) for k in range(K)}
    delta_ks = {k: np.full(len(base_corners), 1e9) for k in range(K)}
    jack_nerfer = lambda delta: 1 - 7e-5 * (0.15 + abs(delta - 0.08))**(-4)
    for k in range(K):
        notes = note_seq_by_column[k]
        for i in range(len(notes) - 1):
            start = notes[i][1]
            end = notes[i+1][1]
            # Find indices in base_corners that lie in [start, end)
            left_idx = np.searchsorted(base_corners, start, side='left')
            right_idx = np.searchsorted(base_corners, end, side='left')
            idx = np.arange(left_idx, right_idx)
            if len(idx) == 0:
                continue
            delta = 0.001 * (end - start)
            val = (delta**(-1)) * (delta + 0.11 * x**(1/4))**(-1)
            J_val = val * jack_nerfer(delta)
            J_ks[k][idx] = J_val
            delta_ks[k][idx] = delta
    
    # Now smooth each column's J_ks
    Jbar_ks = {}
    for k in range(K):
        Jbar_ks[k] = smooth_on_corners(base_corners, J_ks[k], window=500, scale=0.001, mode='sum')
    
    # Aggregate across columns using weighted average
    Jbar = np.empty(len(base_corners))
    for i, s in enumerate(base_corners):
        vals = [Jbar_ks[k][i] for k in range(K)]
        weights = [1 / delta_ks[k][i] for k in range(K)]
        num = sum((max(v, 0) ** 5) * w for v, w in zip(vals, weights))
        den = sum(weights)
        Jbar[i] = num / max(1e-9, den)
        Jbar[i] = Jbar[i]**(1/5)
    
    return delta_ks, Jbar

def compute_Xbar(K, T, x, note_seq_by_column, active_columns, base_corners):
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

    fast_cross = {k: np.zeros(len(base_corners)) for k in range(K+1)}
    cross_coeff = cross_matrix[K]
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
            idx_start = np.searchsorted(base_corners, start, side='left')
            idx_end = np.searchsorted(base_corners, end, side='left')
            idx = np.arange(idx_start, idx_end)
            if len(idx) == 0:
                continue
            delta = 0.001 * (notes_in_pair[i][1] - notes_in_pair[i-1][1])
            val = 0.16 * max(x, delta)**(-2)
            if ((k - 1) not in active_columns[idx_start] and (k - 1) not in active_columns[idx_end]) or (k not in active_columns[idx_start] and k not in active_columns[idx_end]):
                val*=(1-cross_coeff[k])
            X_ks[k][idx] = val
            fast_cross[k][idx] = max(0, 0.4*max(delta, 0.06, 0.75*x)**(-2) - 80)
    X_base = np.zeros(len(base_corners))
    for i in range(len(base_corners)):
        X_base[i] = sum(X_ks[k][i] * cross_coeff[k] for k in range(K+1)) + sum(np.sqrt(fast_cross[k][i]*cross_coeff[k]*fast_cross[k+1][i]*cross_coeff[k+1]) for k in range(0, K))

    Xbar = smooth_on_corners(base_corners, X_base, window=500, scale=0.001, mode='sum')
    return Xbar

def compute_Pbar(K, T, x, note_seq, LN_rep, anchor, base_corners):
    stream_booster = lambda delta: 1 + 1.7e-7 * ((7.5 / delta) - 160) * ((7.5 / delta) - 360)**2 if 160 < (7.5 / delta) < 360 else 1
    
    P_step = np.zeros(len(base_corners))
    for i in range(len(note_seq) - 1):
        h_l = note_seq[i][1]
        h_r = note_seq[i+1][1]
        delta_time = h_r - h_l
        if delta_time < 1e-9:
            # Dirac delta case: when notes occur at the same time.
            # Add the spike exactly at the note head in the base grid.
            spike = 1000 * (0.02 * (4 / x - 24))**(1/4)
            left_idx = np.searchsorted(base_corners, h_l, side='left')
            right_idx = np.searchsorted(base_corners, h_l, side='right')
            idx = np.arange(left_idx, right_idx)
            if len(idx) > 0:
                P_step[idx] += spike
            # Continue so that we add a spike for each additional simultaneous note.
            continue
        # For the regular case where delta_time > 0, identify the base grid indices in [h_l, h_r)
        left_idx = np.searchsorted(base_corners, h_l, side='left')
        right_idx = np.searchsorted(base_corners, h_r, side='left')
        idx = np.arange(left_idx, right_idx)
        if len(idx) == 0:
            continue
        delta = 0.001 * delta_time
        v = 1 + 6 * 0.001 * LN_sum(h_l, h_r, LN_rep)
        b_val = stream_booster(delta)
        if delta < 2 * x / 3:
            inc = delta**(-1) * (0.08 * x**(-1) * (1 - 24 * x**(-1) * (delta - x/2)**2))**(1/4) * max(b_val, v)
        else:
            inc = delta**(-1) * (0.08 * x**(-1) * (1 - 24 * x**(-1) * (x/6)**2))**(1/4) * max(b_val, v)
        P_step[idx] += np.minimum(inc * anchor[idx], np.maximum(inc, inc*2-10))
    
    Pbar = smooth_on_corners(base_corners, P_step, window=500, scale=0.001, mode='sum')
    return Pbar

def compute_Abar(K, T, x, note_seq_by_column, active_columns, delta_ks, A_corners, base_corners):
    dks = {k: np.zeros(len(base_corners)) for k in range(K-1)}
    for i in range(len(base_corners)):
        cols = active_columns[i]
        for j in range(len(cols) - 1):
            k0 = cols[j]
            k1 = cols[j+1]
            # Use the delta_ks computed before on base_corners
            dks[k0][i] = abs(delta_ks[k0][i] - delta_ks[k1][i]) + 0.4*max(0, max(delta_ks[k0][i], delta_ks[k1][i]) - 0.11)
    
    A_step = np.ones(len(A_corners))
    
    for i, s in enumerate(A_corners):
        idx = np.searchsorted(base_corners, s)
        if idx >= len(base_corners):
            idx = len(base_corners) - 1
        cols = active_columns[idx]
        for j in range(len(cols) - 1):
            k0 = cols[j]
            k1 = cols[j+1]
            d_val = dks[k0][idx]
            if d_val < 0.02:
                A_step[i] *= min(0.75 + 0.5 * max(delta_ks[k0][idx], delta_ks[k1][idx]), 1)
            elif d_val < 0.07:
                A_step[i] *= min(0.65 + 5*d_val + 0.5 * max(delta_ks[k0][idx], delta_ks[k1][idx]), 1)
            # Otherwise leave A_step[i] unchanged.

    Abar = smooth_on_corners(A_corners, A_step, window=250, mode='avg')
    return Abar

def compute_Rbar(K, T, x, note_seq_by_column, tail_seq, base_corners):    
    I_arr = np.zeros(len(base_corners))
    R_step = np.zeros(len(base_corners))

    times_by_column = {i: [note[1] for note in column] 
                       for i, column in enumerate(note_seq_by_column)}

    # Release Index
    I_list = []
    for i in range(len(tail_seq)):
        k, h_i, t_i = tail_seq[i]
        _, h_j, _ = find_next_note_in_column((k, h_i, t_i), times_by_column[k], note_seq_by_column)
        I_h = 0.001 * abs(t_i - h_i - 80) / x
        I_t = 0.001 * abs(h_j - t_i - 80) / x
        I_list.append(2 / (2 + math.exp(-5*(I_h-0.75)) + math.exp(-5*(I_t-0.75))))
    
    # For each interval between successive tail times, assign I and R.
    for i in range(len(tail_seq)-1):
        t_start = tail_seq[i][2]
        t_end = tail_seq[i+1][2]
        left_idx = np.searchsorted(base_corners, t_start, side='left')
        right_idx = np.searchsorted(base_corners, t_end, side='left')
        idx = np.arange(left_idx, right_idx)
        if len(idx) == 0:
            continue
        I_arr[idx] = 1 + I_list[i]
        delta_r = 0.001 * (tail_seq[i+1][2] - tail_seq[i][2])
        R_step[idx] = 0.08 * (delta_r)**(-0.5) * x**(-1) * (1 + 0.8*(I_list[i] + I_list[i+1]))
    Rbar = smooth_on_corners(base_corners, R_step, window=500, scale=0.001, mode='sum')
    return Rbar

def compute_C_and_Ks(K, T, note_seq, key_usage, base_corners):
    # C(s): count of notes within 500 ms
    note_hit_times = sorted(n[1] for n in note_seq)
    C_step = np.zeros(len(base_corners))
    for i, s in enumerate(base_corners):
        low = s - 500
        high = s + 500
        # Use binary search on note_hit_times:
        cnt = bisect.bisect_left(note_hit_times, high) - bisect.bisect_left(note_hit_times, low)
        C_step[i] = cnt

    # Ks: local key usage count (minimum 1)
    Ks_step = np.array([max(sum(1 for k in range(K) if key_usage[k][i]), 1) for i in range(len(base_corners))])
    return C_step, Ks_step

def calculate(file_path, mod):
    # === Basic Setup and Parsing ===
    x, K, T, note_seq, note_seq_by_column, LN_seq, tail_seq, LN_seq_by_column = preprocess_file(file_path, mod)

    all_corners, base_corners, A_corners = get_corners(T, note_seq)

    # For each column, store a boolean of its usage (whether non-empty within 150 ms) over time. Example: key_usage[k][idx].
    key_usage = get_key_usage(K, T, note_seq, base_corners)
    # At each time in base_corners, build a list of columns that are active:
    active_columns = [ [k for k in range(K) if key_usage[k][i]] for i in range(len(base_corners)) ]

    key_usage_400 = get_key_usage_400(K, T, note_seq, base_corners)

    anchor = compute_anchor(K, key_usage_400, base_corners)
    
    delta_ks, Jbar = compute_Jbar(K, T, x, note_seq_by_column, base_corners)
    Jbar = interp_values(all_corners, base_corners, Jbar)
    
    Xbar = compute_Xbar(K, T, x, note_seq_by_column, active_columns, base_corners)
    Xbar = interp_values(all_corners, base_corners, Xbar)

    # Build the sparse representation of cumulative LN bodies.
    LN_rep = LN_bodies_count_sparse_representation(LN_seq, T)

    Pbar = compute_Pbar(K, T, x, note_seq, LN_rep, anchor, base_corners)
    Pbar = interp_values(all_corners, base_corners, Pbar)

    Abar = compute_Abar(K, T, x, note_seq_by_column, active_columns, delta_ks, A_corners, base_corners)
    Abar = interp_values(all_corners, A_corners, Abar)

    Rbar = compute_Rbar(K, T, x, note_seq_by_column, tail_seq, base_corners)
    Rbar = interp_values(all_corners, base_corners, Rbar)

    C_step, Ks_step = compute_C_and_Ks(K, T, note_seq, key_usage, base_corners)
    C_arr = step_interp(all_corners, base_corners, C_step)
    Ks_arr = step_interp(all_corners, base_corners, Ks_step)
    
    # === Final Computations ===
    # Compute Difficulty D on all_corners:
    S_all = ((0.4 * (Abar**(3/ Ks_arr) * np.minimum(Jbar, 8+0.85*Jbar))**1.5) + 
             ((1-0.4) * (Abar**(2/3) * (0.8*Pbar + Rbar*35/(C_arr+8)))**1.5))**(2/3)
    T_all = (Abar**(3/ Ks_arr) * Xbar) / (Xbar + S_all + 1)
    D_all = 2.7 * (S_all**0.5) * (T_all**1.5) + S_all * 0.27
    
    df_corners = pd.DataFrame({
        'time': all_corners,
        'Jbar': Jbar,
        'Xbar': Xbar,
        'Pbar': Pbar,
        'Abar': Abar,
        'Rbar': Rbar,
        'C': C_arr,
        'Ks': Ks_arr,
        'D': D_all
    })
    
    # Compute the gaps between consecutive times in a vectorised way.
    # For interior points, the effective gap is the average of the left and right gap.
    gaps = np.empty_like(all_corners, dtype=float)
    gaps[0] = (all_corners[1] - all_corners[0]) / 2.0
    gaps[-1] = (all_corners[-1] - all_corners[-2]) / 2.0
    gaps[1:-1] = (all_corners[2:] - all_corners[:-2]) / 2.0
    
    # The effective weight for each corner is the product of its density and its gap.
    effective_weights = C_arr * gaps
    df_sorted = df_corners.sort_values('D')
    D_sorted = df_sorted['D'].values
    sorted_indices = df_sorted.index.to_numpy()
    w_sorted = effective_weights[sorted_indices]
    
    # Compute the cumulative sum of the effective weights.
    cum_weights = np.cumsum(w_sorted)
    total_weight = cum_weights[-1]
    norm_cum_weights = cum_weights / total_weight
    
    target_percentiles = np.array([0.945, 0.935, 0.925, 0.915, 0.845, 0.835, 0.825, 0.815])
    
    indices = np.searchsorted(norm_cum_weights, target_percentiles, side='left')
    
    percentile_93 = np.mean(D_sorted[indices[:4]])
    percentile_83 = np.mean(D_sorted[indices[4:8]])
    
    weighted_mean = (np.sum(D_sorted**5 * w_sorted) / np.sum(w_sorted))**(1 / 5)
    
    # Final SR calculation
    SR = (0.88 * percentile_93) * 0.25 + (0.94 * percentile_83) * 0.2 + weighted_mean * 0.55
    SR = SR**(1.0) / (8**1.0) * 8
    
    total_notes = len(note_seq) + 0.5*sum(np.minimum((t-h), 1000)/200 for (k, h, t) in LN_seq)
    SR *= total_notes / (total_notes + 60)

    SR = rescale_high(SR)
    SR *= 0.975
    
    return SR
