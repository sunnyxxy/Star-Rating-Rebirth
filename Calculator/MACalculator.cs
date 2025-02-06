using System;
using System.Collections.Generic;

namespace Malody.Chart
{
    public struct Note
    {
        public int Column;  // key/column index
        public int Head;    // head (hit) time
        public int Tail;    // tail time (or -1 if not LN)

        public Note(int column, int head, int tail)
        {
            Column = column;
            Head = head;
            Tail = tail;
        }
    }

    /// <summary>
    /// Used to store various computed values at a corner (time point).
    /// </summary>
    public class CornerData
    {
        public double Time;
        public double Jbar;
        public double Xbar;
        public double Pbar;
        public double Abar;
        public double Rbar;
        public double C;
        public double Ks;
        public double D;
        public double Weight;
    }

    /// <summary>
    /// MACalculator computes the difficulty rating from noteSeq (a sequence of notes).
    /// All methods are static.
    /// </summary>
    public static class MACalculator
    {
        /// <param name="noteSeq">List of Note objects.</param>
        /// <param name="keyCount">Number of keys (columns).</param>
        /// <returns>The computed difficulty (Level) as an int.</returns>
        public static int Calculate(List<Note> noteSeq, int keyCount)
        {
            // Fixed tuning constants.
            const double lambda_n = 5;
            const double lambda_1 = 0.11;
            const double lambda_3 = 24.0;
            const double lambda2 = 6.8; // For Malody
            // Malody has no lambda4 for releases
            const double w0 = 0.4;
            const double w1 = 2.7;
            const double p1 = 1.5;
            const double w2 = 0.27;
            const double p0 = 1.0;
            const double x = 0.085; // for Malody

            // --- Sort notes by head time, then by column ---
            noteSeq.Sort((a, b) =>
            {
                int cmp = a.Head.CompareTo(b.Head);
                return cmp == 0 ? a.Column.CompareTo(b.Column) : cmp;
            });

            // --- Group notes by column ---
            Dictionary<int, List<Note>> noteDict = new Dictionary<int, List<Note>>();
            foreach (var note in noteSeq)
            {
                if (!noteDict.ContainsKey(note.Column))
                    noteDict[note.Column] = new List<Note>();
                noteDict[note.Column].Add(note);
            }
            List<List<Note>> noteSeqByColumn = noteDict
                .OrderBy(kvp => kvp.Key)
                .Select(kvp => kvp.Value)
                .ToList();

            // --- Long notes ---
            List<Note> LNSeq = noteSeq.Where(n => n.Tail >= 0).ToList();
            List<Note> tailSeq = LNSeq.OrderBy(n => n.Tail).ToList();

            Dictionary<int, List<Note>> LNDict = new Dictionary<int, List<Note>>();
            foreach (var note in LNSeq)
            {
                if (!LNDict.ContainsKey(note.Column))
                    LNDict[note.Column] = new List<Note>();
                LNDict[note.Column].Add(note);
            }
            List<List<Note>> LNSeqByColumn = LNDict
                .OrderBy(kvp => kvp.Key)
                .Select(kvp => kvp.Value)
                .ToList();

            int maxHead = noteSeq.Max(n => n.Head);
            int maxTail = noteSeq.Max(n => n.Tail);
            int T = Math.Max(maxHead, maxTail) + 1;

            // --- Determine Corner Times for base variables and for A ---
            HashSet<int> cornersBase = new HashSet<int>();
            foreach (var note in noteSeq)
            {
                cornersBase.Add(note.Head);
                if (note.Tail >= 0)
                    cornersBase.Add(note.Tail);
            }
            foreach (var s in cornersBase.ToList())
            {
                cornersBase.Add(s + 501);
                cornersBase.Add(s - 499);
                cornersBase.Add(s + 1);
            }
            cornersBase.Add(0);
            cornersBase.Add(T);
            List<int> cornersBaseList = cornersBase.Where(s => s >= 0 && s <= T).ToList();
            cornersBaseList.Sort();

            HashSet<int> cornersA = new HashSet<int>();
            foreach (var note in noteSeq)
            {
                cornersA.Add(note.Head);
                if (note.Tail >= 0)
                    cornersA.Add(note.Tail);
            }
            foreach (var s in cornersA.ToList())
            {
                cornersA.Add(s + 1000);
                cornersA.Add(s - 1000);
            }
            cornersA.Add(0);
            cornersA.Add(T);
            List<int> cornersAList = cornersA.Where(s => s >= 0 && s <= T).ToList();
            cornersAList.Sort();

            HashSet<int> allCornersSet = new HashSet<int>(cornersBaseList);
            allCornersSet.UnionWith(cornersAList);
            List<int> allCornersList = allCornersSet.ToList();
            allCornersList.Sort();

            double[] allCorners = allCornersList.Select(val => (double)val).ToArray();
            double[] baseCorners = cornersBaseList.Select(val => (double)val).ToArray();
            double[] ACorners = cornersAList.Select(val => (double)val).ToArray();

            Func<double, int> StarToLevel = (x) => {
                if (x < 5.78) {
                    return (int)Math.Floor(x / 0.17);
                }

                if (x < 6) {
                    return 34;
                }

                return 35 + (int)Math.Floor((x - 6) / 0.25);
            };

            // --- Section 2.3: Compute Jbar ---
            // Console.WriteLine("2.3");
            Func<double, double> jackNerfer = delta =>
            {
                return 1 - 7e-5 * Math.Pow(0.15 + Math.Abs(delta - 0.08), -4);
            };

            // Allocate arrays for each column. For each column k,
            // J_ks[k] will store the unsmoothed J values on baseCorners,
            // and delta_ks[k] will store the corresponding delta values.
            Dictionary<int, double[]> J_ks = new Dictionary<int, double[]>();
            Dictionary<int, double[]> delta_ks = new Dictionary<int, double[]>();
            for (int k = 0; k < keyCount; k++)
            {
                J_ks[k] = new double[baseCorners.Length];
                delta_ks[k] = new double[baseCorners.Length];
                // Initialize delta_ks to a large value.
                for (int j = 0; j < baseCorners.Length; j++)
                {
                    delta_ks[k][j] = 1e9;
                }
            }

            // For each column, compute unsmoothed J using a linear sweep over baseCorners.
            for (int k = 0; k < keyCount; k++)
            {
                List<Note> notes = noteSeqByColumn[k];
                int pointer = 0;  // pointer over the baseCorners array

                // For each adjacent note pair in the column.
                for (int i = 0; i < notes.Count - 1; i++)
                {
                    int start = notes[i].Head;
                    int end = notes[i + 1].Head;
                    double delta = 0.001 * (end - start);
                    double val = (1.0 / delta) * (1.0 / (delta + lambda_1 * Math.Pow(x, 0.25)));
                    double J_val = val * jackNerfer(delta);

                    // Advance pointer until we reach the first base corner >= start.
                    while (pointer < baseCorners.Length && baseCorners[pointer] < start)
                    {
                        pointer++;
                    }
                    // For all base corners in [start, end), assign J_val and delta.
                    while (pointer < baseCorners.Length && baseCorners[pointer] < end)
                    {
                        J_ks[k][pointer] = J_val;
                        delta_ks[k][pointer] = delta;
                        pointer++;
                    }
                }
            }

            // Smooth each column’s J using a sliding ±500 window.
            Dictionary<int, double[]> Jbar_ks = new Dictionary<int, double[]>();
            for (int k = 0; k < keyCount; k++)
            {
                Jbar_ks[k] = SmoothOnCorners(baseCorners, J_ks[k], 500, 0.001, "sum");
            }

            // Now, for each base corner, aggregate across columns using the lambda_n–power average.
            double[] Jbar_base = new double[baseCorners.Length];
            for (int j = 0; j < baseCorners.Length; j++)
            {
                double num = 0.0, den = 0.0;
                for (int k = 0; k < keyCount; k++)
                {
                    double v = Math.Max(Jbar_ks[k][j], 0);
                    double weight = 1.0 / delta_ks[k][j];
                    num += Math.Pow(v, lambda_n) * weight;
                    den += weight;
                }
                double avg = num / Math.Max(1e-9, den);
                Jbar_base[j] = Math.Pow(avg, 1.0 / lambda_n);
            }

            // Interpolate Jbar from baseCorners to allCorners.
            double[] Jbar = InterpValues(allCorners, baseCorners, Jbar_base);


            // --- Section 2.4: Compute Xbar ---
            // Console.WriteLine("2.4");
            List<List<double>> crossMatrix = new List<List<double>>()
            {
                new List<double> { -1 },
                new List<double> { 0.075, 0.075 },
                new List<double> { 0.125, 0.05, 0.125 },
                new List<double> { 0.125, 0.125, 0.125, 0.125 },
                new List<double> { 0.175, 0.25, 0.05, 0.25, 0.175 },
                new List<double> { 0.175, 0.25, 0.175, 0.175, 0.25, 0.175 },
                new List<double> { 0.225, 0.35, 0.25, 0.05, 0.25, 0.35, 0.225 },
                new List<double> { 0.225, 0.35, 0.25, 0.225, 0.225, 0.25, 0.35, 0.225 },
                new List<double> { 0.275, 0.45, 0.35, 0.25, 0.05, 0.25, 0.35, 0.45, 0.275 },
                new List<double> { 0.275, 0.45, 0.35, 0.25, 0.275, 0.275, 0.25, 0.35, 0.45, 0.275 },
                new List<double> { 0.325, 0.55, 0.45, 0.35, 0.25, 0.05, 0.25, 0.35, 0.45, 0.55, 0.325 }
            };

            // Allocate an array for each key pair (for k=0..keyCount, total keyCount+1 arrays).
            Dictionary<int, double[]> X_ks = new Dictionary<int, double[]>();
            for (int k = 0; k <= keyCount; k++)
            {
                X_ks[k] = new double[baseCorners.Length]; // All values default to 0.
            }

            // For each k, compute a step function over the baseCorners that is constant over each interval.
            // Instead of checking every baseCorner for each interval, we “sweep” through baseCorners with a pointer.
            for (int k = 0; k <= keyCount; k++)
            {
                // Determine the merged note sequence for this key–pair:
                List<Note> notesInPair;
                if (k == 0)
                {
                    notesInPair = noteSeqByColumn[0];
                }
                else if (k == keyCount)
                {
                    notesInPair = noteSeqByColumn[keyCount - 1];
                }
                else
                {
                    // Merge the two sorted lists from columns (k–1) and k.
                    notesInPair = MergeSorted(noteSeqByColumn[k - 1], noteSeqByColumn[k]);
                }

                // pointer scans through baseCorners once.
                int pointer = 0;
                for (int i = 1; i < notesInPair.Count; i++)
                {
                    // For this note pair, define the interval [start, end)
                    int start = notesInPair[i - 1].Head;
                    int end = notesInPair[i].Head;
                    double delta = 0.001 * (end - start);
                    double val = 0.16 * Math.Pow(Math.Max(x, delta), -2);

                    // Advance the pointer until the current base corner is within [start, end)
                    while (pointer < baseCorners.Length && baseCorners[pointer] < start)
                    {
                        pointer++;
                    }
                    // Now assign the value for all baseCorners in [start, end)
                    while (pointer < baseCorners.Length && baseCorners[pointer] < end)
                    {
                        X_ks[k][pointer] = val;
                        pointer++;
                    }
                    // Note: Because note intervals are in increasing order, pointer only moves forward.
                }
            }

            // Combine the X_ks values across k using the cross–matrix coefficients.
            // (The cross–matrix for keyCount returns a list of keyCount+1 coefficients.)
            List<double> crossCoeff = crossMatrix[keyCount];
            double[] X_base = new double[baseCorners.Length];
            for (int i = 0; i < baseCorners.Length; i++)
            {
                double sum = 0.0;
                for (int k = 0; k <= keyCount; k++)
                {
                    sum += X_ks[k][i] * crossCoeff[k];
                }
                X_base[i] = sum;
            }

            // Smooth and interpolate as in the rest of the algorithm.
            double[] Xbar_base = SmoothOnCorners(baseCorners, X_base, 500, 0.001, "sum");
            double[] Xbar = InterpValues(allCorners, baseCorners, Xbar_base);

            // --- Section 2.5: Compute Pbar ---
            // Console.WriteLine("2.5");
                
            // Build LN_bodies array over time [0, T)
            double[] LN_bodies = new double[T];
            for (int i = 0; i < T; i++)
                LN_bodies[i] = 0.0;

            // For each long note, add contributions in three segments:
            //   from h to t0, add nothing;
            //   from t0 to t1, add 1.3;
            //   from t1 to t, add 1.0.
            foreach (var note in LNSeq)
            {
                int h = note.Head;
                int t = note.Tail;
                int t0 = Math.Min(h + 60, t);
                int t1 = Math.Min(h + 120, t);
                for (int i = t0; i < t1; i++)
                    LN_bodies[i] += 1.3;
                for (int i = t1; i < t; i++)
                    LN_bodies[i] += 1.0;
            }

            // Compute cumulative sum over LN_bodies
            double[] cumsum_LN = new double[T + 1];
            cumsum_LN[0] = 0.0;
            for (int i = 1; i <= T; i++)
            {
                cumsum_LN[i] = cumsum_LN[i - 1] + LN_bodies[i - 1];
            }

            // LN_sum returns the exact sum over LN_bodies in the interval [a, b)
            Func<int, int, double> LN_sum = (a, b) => cumsum_LN[b] - cumsum_LN[a];

            // Stream Booster
            Func<double, double> streamBooster = (delta) =>
            {
                double val = 7.5 / delta;
                if (val > 160 && val < 360)
                    return 1 + 1.7e-7 * (val - 160) * Math.Pow(val - 360, 2);
                return 1.0;
            };

            // Allocate P_step on the base grid.
            double[] P_step = new double[baseCorners.Length];
            for (int i = 0; i < baseCorners.Length; i++)
                P_step[i] = 0.0;

            // Process each adjacent pair of notes in noteSeq. Since noteSeq is sorted by head time,
            // the interval [h_l, h_r) for each pair will also be in increasing order.
            // We maintain a pointer into baseCorners that advances monotonically.
            int pointerP = 0;
            for (int i = 0; i < noteSeq.Count - 1; i++)
            {
                int h_l = noteSeq[i].Head;
                int h_r = noteSeq[i + 1].Head;
                double delta_time = h_r - h_l;

                if (delta_time < 1e-9)
                {
                    // Handle Dirac delta spikes when consecutive notes have identical times.
                    // Find the base corner exactly equal to h_l (using binary search is acceptable here)
                    int idx = Array.BinarySearch(baseCorners, (double)h_l);
                    if (idx < 0)
                        idx = ~idx;
                    if (idx < baseCorners.Length && Math.Abs(baseCorners[idx] - h_l) < 1e-9)
                    {
                        double spike = 1000 * Math.Pow(0.02 * (4 / x - lambda_3), 0.25);
                        P_step[idx] += spike;
                    }
                    continue;
                }

                // Compute constant values for this interval.
                double delta = 0.001 * delta_time;
                double v = 1 + lambda2 * 0.001 * LN_sum(h_l, h_r);
                double bVal = streamBooster(delta);
                double inc;
                if (delta < 2 * x / 3)
                {
                    inc = (1.0 / delta) * Math.Pow(0.08 * (1.0 / x) *
                        (1 - lambda_3 * (1.0 / x) * Math.Pow(delta - x / 2, 2)), 0.25) * bVal * v;
                }
                else
                {
                    inc = (1.0 / delta) * Math.Pow(0.08 * (1.0 / x) *
                        (1 - lambda_3 * (1.0 / x) * Math.Pow(x / 6, 2)), 0.25) * bVal * v;
                }

                // Advance pointerP until the current base corner is at least h_l.
                while (pointerP < baseCorners.Length && baseCorners[pointerP] < h_l)
                    pointerP++;

                // For every base corner in the interval [h_l, h_r), add the increment.
                while (pointerP < baseCorners.Length && baseCorners[pointerP] < h_r)
                {
                    P_step[pointerP] += inc;
                    pointerP++;
                }
                // Since noteSeq is sorted, pointerP never resets backwards.
            }

            // Smooth and interpolate P_step as before.
            double[] Pbar_base = SmoothOnCorners(baseCorners, P_step, 500, 0.001, "sum");
            double[] Pbar = InterpValues(allCorners, baseCorners, Pbar_base);


            // --- Section 2.6: Compute Abar ---
            // Console.WriteLine("2.6");

            // Allocate a boolean active–flag array per key over baseCorners.
            // (New bool arrays are false by default; no need to loop and set to false.)
            Dictionary<int, bool[]> KU_ks = new Dictionary<int, bool[]>();
            for (int k = 0; k < keyCount; k++)
            {
                KU_ks[k] = new bool[baseCorners.Length];
            }

            // For each key, mark baseCorners that lie within the “active” interval for each note.
            // The active interval for a note is [max(note.Head - 500, 0), (note.Tail < 0 ? note.Head + 500 : min(note.Tail + 500, T - 1))).
            for (int k = 0; k < keyCount; k++)
            {
                // Get the note sequence for key k.
                List<Note> notes = noteSeqByColumn[k];
                foreach (var note in notes)
                {
                    int activeStart = Math.Max(note.Head - 500, 0);
                    int activeEnd = (note.Tail < 0) ? (note.Head + 500) : Math.Min(note.Tail + 500, T - 1);
                    // Use binary search to find the first baseCorner >= activeStart.
                    int startIdx = Array.BinarySearch(baseCorners, (double)activeStart);
                    if (startIdx < 0)
                        startIdx = ~startIdx;
                    // Advance pointer until the base corner is no longer less than activeEnd.
                    int idx = startIdx;
                    while (idx < baseCorners.Length && baseCorners[idx] < activeEnd)
                    {
                        KU_ks[k][idx] = true;
                        idx++;
                    }
                }
            }

            // For each baseCorner, build a list of active keys.
            List<List<int>> KU_s_cols = new List<List<int>>(baseCorners.Length);
            for (int i = 0; i < baseCorners.Length; i++)
            {
                List<int> activeCols = new List<int>();
                for (int k = 0; k < keyCount; k++)
                {
                    if (KU_ks[k][i])
                        activeCols.Add(k);
                }
                KU_s_cols.Add(activeCols);
            }

            // Compute dks: For each baseCorner, for each adjacent pair of active keys,
            // compute the difference measure: |delta_ks[k0] - delta_ks[k1]| + max(0, max(delta_ks[k0], delta_ks[k1]) - 0.3).
            Dictionary<int, double[]> dks = new Dictionary<int, double[]>();
            for (int k = 0; k < keyCount - 1; k++)
            {
                dks[k] = new double[baseCorners.Length];
            }
            for (int i = 0; i < baseCorners.Length; i++)
            {
                List<int> cols = KU_s_cols[i];
                // Only if there are at least two active keys.
                for (int j = 0; j < cols.Count - 1; j++)
                {
                    int k0 = cols[j];
                    int k1 = cols[j + 1];
                    dks[k0][i] = Math.Abs(delta_ks[k0][i] - delta_ks[k1][i]) +
                                0.4 * Math.Max(0, Math.Max(delta_ks[k0][i], delta_ks[k1][i]) - 0.11);
                }
            }

            // Compute A_step on the A–grid (ACorners). Start with a default value of 1.0.
            double[] A_step = new double[ACorners.Length];
            for (int i = 0; i < ACorners.Length; i++)
                A_step[i] = 1.0;

            // For each A–corner, determine the corresponding value from the base grid.
            // We do this by finding the nearest baseCorner using binary search, then using the active key list there.
            for (int i = 0; i < ACorners.Length; i++)
            {
                double s = ACorners[i];
                int idx = Array.BinarySearch(baseCorners, s);
                if (idx < 0)
                    idx = ~idx;
                if (idx >= baseCorners.Length)
                    idx = baseCorners.Length - 1;
                // Get the list of active keys at this base corner.
                List<int> cols = KU_s_cols[idx];
                // For each adjacent pair of active keys in that list, adjust A_step.
                for (int j = 0; j < cols.Count - 1; j++)
                {
                    int k0 = cols[j];
                    int k1 = cols[j + 1];
                    double d_val = dks[k0][idx];
                    if (d_val < 0.02)
                        A_step[i] *= Math.Min(0.75 + 0.5 * Math.Max(delta_ks[k0][idx], delta_ks[k1][idx]), 1);
                    else if (d_val < 0.07)
                        A_step[i] *= Math.Min(0.65 + 5 * d_val + 0.5 * Math.Max(delta_ks[k0][idx], delta_ks[k1][idx]), 1);
                }
            }

            // Finally, smooth A_step on the ACorners with a ±500 window (using average smoothing),
            // then interpolate Abar from the ACorners to the overall grid.
            double[] Abar_A = SmoothOnCorners(ACorners, A_step, 500, 1.0, "avg");
            double[] Abar = InterpValues(allCorners, ACorners, Abar_A);


            // --- Section 2.7: Compute Rbar ---
            // Console.WriteLine("2.7");

            double[] Rbar = new double[allCorners.Length];
            for (int i = 0; i < Rbar.Length; i++)
            {
                Rbar[i] = 0;
            }


            // --- Section 3: Compute C and Ks ---
            // Console.WriteLine("3");
            List<int> noteHitTimes = noteSeq.Select(n => n.Head).ToList();
            noteHitTimes.Sort();
            double[] C_step = new double[baseCorners.Length];
            for (int i = 0; i < baseCorners.Length; i++)
            {
                double s = baseCorners[i];
                double low = s - 500;
                double high = s + 500;
                int cntLow = LowerBound(noteHitTimes, (int)low);
                int cntHigh = LowerBound(noteHitTimes, (int)high);
                int cnt = cntHigh - cntLow;
                C_step[i] = cnt;
            }
            double[] C_base = C_step;
            double[] C_arr = StepInterp(allCorners, baseCorners, C_base);

            double[] Ks_step = new double[baseCorners.Length];
            for (int i = 0; i < baseCorners.Length; i++)
            {
                int cntActive = 0;
                for (int k = 0; k < keyCount; k++)
                {
                    if (KU_ks[k][i])
                        cntActive++;
                }
                Ks_step[i] = Math.Max(cntActive, 1);
            }
            double[] Ks_base = Ks_step;
            double[] Ks_arr = StepInterp(allCorners, baseCorners, Ks_base);

            // --- Final Computations: Compute S, T, D ---
            int N = allCorners.Length;
            double[] S_all = new double[N];
            double[] T_all = new double[N];
            double[] D_all = new double[N];
            for (int i = 0; i < N; i++)
            {
                double A_val = Abar[i];
                double J_val = Jbar[i];
                double X_val = Xbar[i];
                double P_val = Pbar[i];
                double R_val = Rbar[i];
                double Ks_val = Ks_arr[i];

                double term1 = Math.Pow(Math.Pow(A_val, 3.0 / Ks_val) * J_val, 1.5);
                double term2 = Math.Pow(Math.Pow(A_val, 2.0 / 3.0) * (0.8 * P_val + R_val), 1.5);
                double S_val = Math.Pow(w0 * term1 + (1 - w0) * term2, 2.0 / 3.0);
                S_all[i] = S_val;
                double T_val = (Math.Pow(A_val, 3.0 / Ks_val) * X_val) / (X_val + S_val + 1);
                T_all[i] = T_val;
                D_all[i] = w1 * Math.Pow(S_val, 0.5) * Math.Pow(T_val, p1) + S_val * w2;
            }

            // --- Weighted–Percentile Calculation ---
            double[] gaps = new double[N];
            if (N == 1)
                gaps[0] = 0;
            else
            {
                gaps[0] = (allCorners[1] - allCorners[0]) / 2.0;
                gaps[N - 1] = (allCorners[N - 1] - allCorners[N - 2]) / 2.0;
                for (int i = 1; i < N - 1; i++)
                    gaps[i] = (allCorners[i + 1] - allCorners[i - 1]) / 2.0;
            }
            double[] effectiveWeights = new double[N];
            for (int i = 0; i < N; i++)
                effectiveWeights[i] = C_arr[i] * gaps[i];

            List<CornerData> cornerDataList = new List<CornerData>();
            for (int i = 0; i < N; i++)
            {
                cornerDataList.Add(new CornerData
                {
                    Time = allCorners[i],
                    Jbar = Jbar[i],
                    Xbar = Xbar[i],
                    Pbar = Pbar[i],
                    Abar = Abar[i],
                    Rbar = Rbar[i],
                    C = C_arr[i],
                    Ks = Ks_arr[i],
                    D = D_all[i],
                    Weight = effectiveWeights[i]
                });
            }
            var sortedList = cornerDataList.OrderBy(cd => cd.D).ToList();
            double[] D_sorted = sortedList.Select(cd => cd.D).ToArray();
            double[] cumWeights = new double[sortedList.Count];
            double sumW = 0.0;
            for (int i = 0; i < sortedList.Count; i++)
            {
                sumW += sortedList[i].Weight;
                cumWeights[i] = sumW;
            }
            double totalWeight = sumW;
            double[] normCumWeights = cumWeights.Select(cw => cw / totalWeight).ToArray();

            double[] targetPercentiles = new double[] { 0.945, 0.935, 0.925, 0.915, 0.845, 0.835, 0.825, 0.815 };
            List<int> indices = new List<int>();
            foreach (double tp in targetPercentiles)
            {
                int idx = Array.FindIndex(normCumWeights, cw => cw >= tp);
                if (idx < 0)
                    idx = sortedList.Count - 1;
                indices.Add(idx);
            }
            double percentile93, percentile83;
            if (indices.Count >= 8)
            {
                double sum93 = 0.0;
                for (int i = 0; i < 4; i++)
                    sum93 += sortedList[indices[i]].D;
                percentile93 = sum93 / 4.0;
                double sum83 = 0.0;
                for (int i = 4; i < 8; i++)
                    sum83 += sortedList[indices[i]].D;
                percentile83 = sum83 / 4.0;
            }
            else
            {
                percentile93 = sortedList.Average(cd => cd.D);
                percentile83 = percentile93;
            }
            double numWeighted = 0.0;
            double denWeighted = 0.0;
            for (int i = 0; i < sortedList.Count; i++)
            {
                numWeighted += Math.Pow(sortedList[i].D, lambda_n) * sortedList[i].Weight;
                denWeighted += sortedList[i].Weight;
            }
            double weightedMean = Math.Pow(numWeighted / denWeighted, 1.0 / lambda_n);
            double SR = (0.88 * percentile93) * 0.25 + (0.94 * percentile83) * 0.2 + weightedMean * 0.55;
            SR = Math.Pow(SR, p0) / Math.Pow(8, p0) * 8;
            double totalNotes = noteSeq.Count + 0.5 * LNSeq.Count;
            SR *= totalNotes / (totalNotes + 60);
            if (SR <= 2)
                SR = Math.Sqrt(SR * 2);
            
            SR *= 1 - 0.0075*keyCount;
            return StarToLevel(SR);
        }

        #region Helper Methods

        // Returns the cumulative sum array for f evaluated on x.
        private static double[] CumulativeSum(double[] x, double[] f)
        {
            int n = x.Length;
            double[] F = new double[n];
            F[0] = 0.0;
            for (int i = 1; i < n; i++)
            {
                F[i] = F[i - 1] + f[i - 1] * (x[i] - x[i - 1]);
            }
            return F;
        }

        // Query cumulative sum at q.
        private static double QueryCumsum(double q, double[] x, double[] F, double[] f)
        {
            if (q <= x[0])
                return 0.0;
            if (q >= x[x.Length - 1])
                return F[x.Length - 1];
            int idx = Array.BinarySearch(x, q);
            if (idx < 0)
                idx = ~idx;
            int i = idx - 1;
            return F[i] + f[i] * (q - x[i]);
        }

        // Smooth values f (defined on x) over a symmetric window.
        // If mode is "avg", returns the average; otherwise multiplies the integral by scale.
        private static double[] SmoothOnCorners(double[] x, double[] f, double window, double scale, string mode)
        {
            int n = f.Length;
            double[] F = CumulativeSum(x, f);
            double[] g = new double[n];
            for (int i = 0; i < n; i++)
            {
                double s = x[i];
                double a = Math.Max(s - window, x[0]);
                double b = Math.Min(s + window, x[x.Length - 1]);
                double val = QueryCumsum(b, x, F, f) - QueryCumsum(a, x, F, f);
                if (mode == "avg")
                    g[i] = (b - a) > 0 ? val / (b - a) : 0.0;
                else
                    g[i] = scale * val;
            }
            return g;
        }

        // Linear interpolation from old_x, old_vals to new_x.
        private static double[] InterpValues(double[] new_x, double[] old_x, double[] old_vals)
        {
            int n = new_x.Length;
            double[] new_vals = new double[n];
            for (int i = 0; i < n; i++)
            {
                double x_val = new_x[i];
                if (x_val <= old_x[0])
                    new_vals[i] = old_vals[0];
                else if (x_val >= old_x[old_x.Length - 1])
                    new_vals[i] = old_vals[old_x.Length - 1];
                else
                {
                    int idx = Array.BinarySearch(old_x, x_val);
                    if (idx < 0)
                        idx = ~idx;
                    int j = idx - 1;
                    double t = (x_val - old_x[j]) / (old_x[j + 1] - old_x[j]);
                    new_vals[i] = old_vals[j] + t * (old_vals[j + 1] - old_vals[j]);
                }
            }
            return new_vals;
        }

        // Step–function interpolation (zero–order hold).
        private static double[] StepInterp(double[] new_x, double[] old_x, double[] old_vals)
        {
            int n = new_x.Length;
            double[] new_vals = new double[n];
            for (int i = 0; i < n; i++)
            {
                double x_val = new_x[i];
                int idx = Array.BinarySearch(old_x, x_val);
                if (idx < 0)
                    idx = ~idx;
                idx = idx - 1;
                if (idx < 0)
                    idx = 0;
                if (idx >= old_vals.Length)
                    idx = old_vals.Length - 1;
                new_vals[i] = old_vals[idx];
            }
            return new_vals;
        }

        // Merges two sorted lists of Note (sorted by Head) into one sorted list.
        private static List<Note> MergeSorted(List<Note> list1, List<Note> list2)
        {
            List<Note> merged = new List<Note>();
            int i = 0, j = 0;
            while (i < list1.Count && j < list2.Count)
            {
                if (list1[i].Head <= list2[j].Head)
                {
                    merged.Add(list1[i]);
                    i++;
                }
                else
                {
                    merged.Add(list2[j]);
                    j++;
                }
            }
            while (i < list1.Count)
            {
                merged.Add(list1[i]);
                i++;
            }
            while (j < list2.Count)
            {
                merged.Add(list2[j]);
                j++;
            }
            return merged;
        }

        // Implements lower_bound: first index at which list[index] >= value.
        private static int LowerBound(List<int> list, int value)
        {
            int low = 0;
            int high = list.Count;
            while (low < high)
            {
                int mid = (low + high) / 2;
                if (list[mid] < value)
                    low = mid + 1;
                else
                    high = mid;
            }
            return low;
        }

        #endregion
    }
}
