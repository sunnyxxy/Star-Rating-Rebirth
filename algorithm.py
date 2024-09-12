import osu_file_parser as osu_parser
from collections import defaultdict
import numpy as np
import heapq
import pandas as pd
import matplotlib.pyplot as plt
import bisect
import math

def calculate(file_path, mod, lambda_2, lambda_4, w_0, w_1, p_1, w_2, p_0):
    lambda_n = 5
    lambda_1 = 0.11
    lambda_3 = 24
    p = osu_parser.parser(file_path)
    p.process()
    p = p.get_parsed_data()
    x = -1
    note_seq = []
    for i in range(len(p[1])):
        k = p[1][i]
        h = p[2][i]
        if p[4][i]==128:
            t=p[3][i]
        else:
            t=-1
        if mod=="DT":
            h = int(math.floor(h*2/3))
            t = int(math.floor(t*2/3))           
        elif mod=="HT":
            h = int(math.floor(h*4/3))
            t = int(math.floor(t*4/3))
        note_seq.append((k, h, t))


    x = 0.3 * ((64.5 - math.ceil(p[5] * 3))/500)**0.5
    note_seq = sorted(note_seq, key=lambda x: (x[1], x[0]))

    note_dict = defaultdict(list)
    for value in note_seq:
        note_dict[value[0]].append(value)

    # Preprocessing
    note_seq_by_column = list(note_dict.values())
    note_seq_by_column.sort(key=lambda x: x[0][0])
    LN_seq = [t for t in note_seq if t[2] >= 0]
    tail_seq = sorted(LN_seq, key=lambda x: x[2])

    LN_dict = defaultdict(list)
    for value in LN_seq:
        LN_dict[value[0]].append(value)

    LN_seq_by_column = list(LN_dict.values())
    LN_seq_by_column.sort(key=lambda x: x[0][0])

    K=p[0]
    T = max(max([t[1] for t in note_seq]), max([t[2] for t in note_seq]))+1

    # Hyperparameters and parameters



    # helper functions
    def smooth(lst):
        lstbar = [0 for _ in range(T)]
        window_sum = sum(lst[0:min(500, T)])
        
        for s in range(T):
            lstbar[s] = 0.001*window_sum
            if s + 500 < T:
                window_sum += lst[s + 500]
            if s - 500 >= 0:
                window_sum -= lst[s - 500]
        return lstbar

    def smooth2(lst):
        lstbar = [0 for _ in range(T)]
        window_sum = sum(lst[0:min(500, T)])
        window_len = min(500, T)
        for s in range(T):
            lstbar[s] = window_sum/window_len
            if s + 500 < T:
                window_sum += lst[s + 500]
                window_len += 1
            if s - 500 >= 0:
                window_sum -= lst[s - 500]
                window_len -= 1
        return lstbar


    # Section 2.3
    def jackNerfer(delta):
        return 1 - 7 * 10**(-5) * (0.15+abs(delta - 0.08))**(-4)

    J_ks=[[0 for _ in range(T)] for _ in range(K)]
    delta_ks=[[10**9 for _ in range(T)] for _ in range(K)]
    for k in range(K):
        for i in range(len(note_seq_by_column[k])-1):
            delta = 0.001*(note_seq_by_column[k][i+1][1]-note_seq_by_column[k][i][1])
            val = delta**(-1)*(delta+lambda_1*x**(1/4))**(-1)
            for s in range(note_seq_by_column[k][i][1], note_seq_by_column[k][i+1][1]):
                delta_ks[k][s]=delta
                J_ks[k][s]=val * jackNerfer(delta)

    Jbar_ks = [[0 for _ in range(T)] for _ in range(K)]

    for k in range(K):
        Jbar_ks[k] = smooth(J_ks[k])

    Jbar = []
    for s in range(T):
        Jbar_ks_vals = []
        weights = []
        for i in range(K):
            Jbar_ks_vals.append(Jbar_ks[i][s])
            weights.append(1/delta_ks[i][s])
        weighted_avg = (sum((max(val, 0) ** lambda_n) * weight for val, weight in zip(Jbar_ks_vals, weights))/max(10**(-9),sum(weights)))**(1/lambda_n)
        Jbar.append(weighted_avg)


    # Section 2.4
    X_ks=[[0 for _ in range(T)] for _ in range(K+1)]
    for k in range(K+1):
        if k==0:
            notes_in_pair = note_seq_by_column[0]
        elif k==K:
            notes_in_pair = note_seq_by_column[K-1]
        else:
            notes_in_pair = list(heapq.merge(note_seq_by_column[k-1], note_seq_by_column[k], key=lambda x: x[1]))
        for i in range(len(notes_in_pair)-1):
            delta = 0.001*(notes_in_pair[i][1] - notes_in_pair[i-1][1])
            val = 0.16*max(x, delta)**(-2)
            for s in range(notes_in_pair[i-1][1], notes_in_pair[i][1]):
                X_ks[k][s] = val
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

    X = []
    for s in range(T):
        X.append(sum(a * b for a, b in zip([x[s] for x in X_ks], cross_matrix[K])))

    Xbar = smooth(X)


    # Section 2.5
    P = [0 for _ in range(T)]
    LN_bodies = np.zeros(T)
    for (k, h, t) in LN_seq:
        t1 = min(h + 80, t)
        LN_bodies[h:t1] += 0.5
        LN_bodies[t1:t] += 1

    def b(delta):
        val = 7.5 / delta

        if 160<val<360:
            return 1 + 1.4 * 10**(-7) * (val - 160) * (val - 360)**2

        return 1

    for i in range(len(note_seq)-1):
        delta = 0.001*(note_seq[i+1][1] - note_seq[i][1])
        if delta<10**(-9):
            P[note_seq[i][1]]+=1000*(0.02*(4/x-lambda_3))**(1/4)
        else:
            h_l = note_seq[i][1]
            h_r = note_seq[i+1][1]
            v = 1 + lambda_2 * 0.001*sum(LN_bodies[h_l:h_r])
            if delta<2*x/3:
                for s in range(h_l, h_r):
                    P[s]+= delta**(-1) * (0.08*x**(-1) * (1 - lambda_3 * x**(-1)*(delta - x/2)**2))**(1/4)*b(delta)*v
            else:
                for s in range(h_l, h_r):
                    P[s]+= delta**(-1) * (0.08*x**(-1) * (1 - lambda_3*x**(-1)*(x/6)**2))**(1/4)*b(delta)*v

    Pbar = smooth(P)
            

    # Section 2.6
    # Local Key Usage by Column/Time
    KU_ks = [[False for _ in range(T)] for _ in range(K)]
    for (k, h, t) in note_seq:
        startTime = max(0, h - 500)
        endTime = 0
        if (t < 0):
            endTime = min(h+500, T-1)
        else:
            endTime = min(t+500, T-1)

        for s in range(startTime, endTime):
            KU_ks[k][s] = True;

    # Local Key Usage by Time but as a list of column numbers for each point s in T
    KU_s_cols = [[k for k in range(K) if KU_ks[k][s]] for s in range(T)]

    # The outer loop is no longer needed here as the columns come from KU_s_cols
    dks = [[0 for _ in range(T)] for _ in range(K-1)]
    for s in range(T):
        cols = KU_s_cols[s]

        for i in range(len(cols) - 1):
            if (cols[i+1] > K - 1):
                continue

            dks[cols[i]][s] = abs(delta_ks[cols[i]][s] - delta_ks[cols[i+1]][s]) + max(0, max(delta_ks[cols[i+1]][s], delta_ks[cols[i]][s]) - 0.3)

    A = [1 for _ in range(T)]
    for s in range(T):
        cols = KU_s_cols[s]

        for i in range(len(cols) - 1):
            if (cols[i+1] > K-1):
                continue

            if dks[cols[i]][s]<0.02:
                A[s]*=min(0.75 + 0.5*max(delta_ks[cols[i+1]][s], delta_ks[cols[i]][s]), 1)
            elif dks[cols[i]][s]<0.07:
                A[s]*=min(0.65 + 5*dks[cols[i]][s] + 0.5*max(delta_ks[cols[i+1]][s], delta_ks[cols[i]][s]), 1)
            else:
                pass

    Abar = smooth2(A)


    # Section 2.7
    def find_next_note_in_column(note, note_seq_by_column):
        k, h, t = note

        second_values = [x[1] for x in note_seq_by_column[k]]
        index = bisect.bisect_left(second_values, h)
        return note_seq_by_column[k][index + 1] if index + 1 < len(note_seq_by_column[k]) else (0, 10**9, 10**9)


    I = [0 for _ in range(len(LN_seq))]
    for i in range(len(tail_seq)):
        (k, h_i, t_i) = tail_seq[i]
        (k, h_j, t_j) = find_next_note_in_column((k, h_i, t_i), note_seq_by_column)
        I_h = 0.001*abs(t_i-h_i-80)/x
        I_t = 0.001*abs(h_j-t_i-80)/x
        I[i] = 2 / (2 + np.exp(-5*(I_h-0.75)) + np.exp(-5*(I_t-0.75)))

    Is = [0 for _ in range(T)]
    R = [0 for _ in range(T)]
    for i in range(len(tail_seq)-1):
        delta_r = 0.001 * (tail_seq[i+1][2] - tail_seq[i][2])
        for s in range(tail_seq[i][2], tail_seq[i+1][2]):
            Is[s] = 1+I[i]
            R[s] = 0.08*(delta_r)**(-1/2)*x**(-1)*(1+lambda_4*(I[i]+I[i+1]))

    Rbar = smooth(R)


    # Section 3
    C = [0 for _ in range(T)]
    start = 0
    end = 0
    for t in range(T):
        while start < len(note_seq) and note_seq[start][1] < t - 500:
            start += 1
        while end < len(note_seq) and note_seq[end][1] < t + 500:
            end += 1
        C[t] = end - start

    # Local Key Usage as an integer for each point s in T (the number of columns used, minimum 1)
    K_s = [max(len([KU_ks[k][s] for k in range(K) if KU_ks[k][s]]), 1) for s in range(T)]

    df = pd.DataFrame({'Jbar': Jbar, 'Xbar': Xbar, 'Pbar': Pbar, 'Abar': Abar, 'Rbar': Rbar, 'C': C, 'Ks': K_s})
    df = df.clip(lower=0)

    df['S'] = ((w_0 * (df['Abar']**(3/df['Ks']) * df['Jbar'])**1.5) + (1-w_0) * (df['Abar']**(2/3) * (0.8*df['Pbar'] + df['Rbar']))**1.5)**(2/3)
    df['T'] = (df['Abar']**(3/df['Ks'])*df['Xbar'])/(df['Xbar']+df['S']+1)
    df['D'] = w_1*df['S']**(1/2)*df['T']**p_1+df['S']*(w_2)

    SR = (sum(df['D']**lambda_n*df['C'])/sum(df['C']))**(1/lambda_n)
    SR = SR**(p_0)/8**p_0*8
    SR *= (len(note_seq)+0.5*len(LN_seq))/(len(note_seq)+0.5*len(LN_seq)+60)
    if SR<=2:
        SR=(SR*2)**0.5
    SR *= 0.96+0.01*K

    return SR
    # Visualisation
    plt.figure(figsize=(10, 6))
    # plt.plot(df['Jbar'], label='Jbar', marker='o', linewidth=0.5, markersize=1, color='orange')
    # plt.plot(df['Xbar'], label='Xbar', marker='o', linewidth=0.5, markersize=1, color='#3333FF')
    # plt.plot(df['Pbar'], label='Pbar', marker='o', linewidth=0.5, markersize=1, color='green')
    # plt.plot(df['Rbar'], label='Rbar', marker='o', linewidth=0.5, markersize=1, color='purple')
    # plt.plot(df['S'], label='S', marker='o', linewidth=0.5, markersize=4)
    plt.plot(df['D'], label='D', marker='o', linewidth=0.5, markersize=2, color='orange')
    plt.plot(df['Ks'], label='Ks', marker='o', linewidth=0.5, markersize=2, color='red')


    ax1 = plt.gca()
    ax2 = ax1.twinx()
    ax2.plot(df['Abar'], label='Abar (Right Axis)', color='r', marker='o', linewidth=0.5, markersize=1)
    # Set the limits for the secondary y-axis
    ax2.set_ylim(0, 1)

    # Adding titles and labels
    plt.title('Difficulty Plot')
    ax1.set_xlabel('Index')
    ax1.set_ylabel('Values')
    # ax2.set_ylabel('Values')

    # Combine legends and place outside the plot
    lines1, labels1 = ax1.get_legend_handles_labels()
    # lines2, labels2 = ax2.get_legend_handles_labels()
    # ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=5)

    # Adjust layout to make room for the legend
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    # Display the plot
    plt.show()
    return SR
