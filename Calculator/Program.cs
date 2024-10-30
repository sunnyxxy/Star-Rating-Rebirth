using System;
using System.Collections.Generic;
using System.Linq;
using System.Diagnostics;

// The MACalculator that contains the entire algorithm on its own
public class MACalculator
{
    private class Note
    {
        public int K { get; set; }
        public int H { get; set; }
        public int T { get; set; }

        public Note(int k, int h, int t)
        {
            K = k;
            H = h;
            T = t;
        }
    }

    private readonly double lambda_n = 5;
    private readonly double lambda_1 = 0.11;
    private readonly double lambda_3 = 24;
    private readonly double lambda_2 = 5.5; // for Malody
    private readonly double lambda_4 = 0.05; // for Malody 
    private readonly double w_0 = 0.4;
    private readonly double w_1 = 2.7;
    private readonly double p_1 = 1.5;
    private readonly double w_2 = 0.27;
    private readonly double p_0 = 1.0;
    private readonly double x = 0.085; // for Malody

    private readonly int granularity = 20;

    private readonly double[][] crossMatrix = new double[][]
    {
        new double[] { -1 },
        new double[] { 0.075, 0.075 },
        new double[] { 0.125, 0.05, 0.125 },
        new double[] { 0.125, 0.125, 0.125, 0.125 },
        new double[] { 0.175, 0.25, 0.05, 0.25, 0.175 },
        new double[] { 0.175, 0.25, 0.175, 0.175, 0.25, 0.175 },
        new double[] { 0.225, 0.35, 0.25, 0.05, 0.25, 0.35, 0.225 },
        new double[] { 0.225, 0.35, 0.25, 0.225, 0.225, 0.25, 0.35, 0.225 },
        new double[] { 0.275, 0.45, 0.35, 0.25, 0.05, 0.25, 0.35, 0.45, 0.275 },
        new double[] { 0.275, 0.45, 0.35, 0.25, 0.275, 0.275, 0.25, 0.35, 0.45, 0.275 },
        new double[] { 0.325, 0.55, 0.45, 0.35, 0.25, 0.05, 0.25, 0.35, 0.45, 0.55, 0.325 }
    };

    private int T;
    private int K;
    private List<Note> noteSeq = new List<Note>();
    private Dictionary<int, List<Note>> noteDict = new Dictionary<int, List<Note>>();
    private List<List<Note>> noteSeqByColumn = new List<List<Note>>();
    private List<Note> LNSeq = new List<Note>();
    private List<Note> tailSeq= new List<Note>();
    private Dictionary<int, List<Note>> LNDict = new Dictionary<int, List<Note>>();
    private List<List<Note>> LNSeqByColumn = new List<List<Note>>();

    public int Calculate(List<(int k, int h, int t)> noteSequence, int keyCount)
    {
        // Initialize data structures
        K = keyCount;
        noteSeq = noteSequence.Select(n => new Note(n.k, n.h, n.t)).OrderBy(n => n.H).ThenBy(n => n.K).ToList();
        
        // Create note dictionary
        noteDict = new Dictionary<int, List<Note>>();
        foreach (var note in noteSeq)
        {
            if (!noteDict.ContainsKey(note.K))
                noteDict[note.K] = new List<Note>();
            noteDict[note.K].Add(note);
        }

        // Create column-based sequences
        noteSeqByColumn = noteDict.Values.OrderBy(list => list[0].K).ToList();
        
        // Create LN sequences
        LNSeq = noteSeq.Where(n => n.T >= 0).ToList();
        tailSeq = LNSeq.OrderBy(n => n.T).ToList();

        LNDict = new Dictionary<int, List<Note>>();
        foreach (var note in LNSeq)
        {
            if (!LNDict.ContainsKey(note.K))
                LNDict[note.K] = new List<Note>();
            LNDict[note.K].Add(note);
        }

        LNSeqByColumn = LNDict.Values.OrderBy(list => list[0].K).ToList();

        // Calculate T
        T = Math.Max(
            noteSeq.Max(n => n.H),
            noteSeq.Max(n => n.T)
        ) + 1;

        try
        {
            var stopwatch = new Stopwatch();

            // Start all sections in parallel
            stopwatch.Start();

            // Define tasks for each section
            var task23 = Task.Run(() =>
            {
                var sectionStopwatch = Stopwatch.StartNew();
                var (Jbar, deltaKs) = CalculateSection23();
                sectionStopwatch.Stop();
                return (Jbar, deltaKs);
            });

            var task24 = Task.Run(() =>
            {
                var sectionStopwatch = Stopwatch.StartNew();
                var Xbar = CalculateSection24();
                sectionStopwatch.Stop();
                return Xbar;
            });

            var task25 = Task.Run(() =>
            {
                var sectionStopwatch = Stopwatch.StartNew();
                var Pbar = CalculateSection25();
                sectionStopwatch.Stop();
                return Pbar;
            });

            // Wait for all tasks to complete
            Task.WaitAll(task23, task24, task25);

            // Retrieve results
            var (Jbar, deltaKs) = task23.Result;
            var Xbar = task24.Result;
            var Pbar = task25.Result;

            stopwatch.Stop();
            Console.WriteLine($"Section 23/24/25 Time: {stopwatch.ElapsedMilliseconds}ms");

            stopwatch.Restart();
            var task26 = Task.Run(() => CalculateSection26(deltaKs));
            var task27 = Task.Run(() => CalculateSection27());

            // Wait for both tasks to complete
            Task.WaitAll(task26, task27);

            // Retrieve results
            var (Abar, KS) = task26.Result;
            var (Rbar, Is) = task27.Result;

            stopwatch.Stop();
            Console.WriteLine($"Section 26/27 Time: {stopwatch.ElapsedMilliseconds}ms");

            // Final calculation
            stopwatch.Restart();
            var result = StarToLevel(CalculateSection3(Jbar, Xbar, Pbar, Abar, Rbar, KS));
            stopwatch.Stop();
            Console.WriteLine($"Section 3 Time: {stopwatch.ElapsedMilliseconds}ms");

            return result;
        }
        catch (Exception)
        {
            return -1;
        }
    }

    private int StarToLevel(double x)
    {
        if (x < 5.78)
        {
            return (int)Math.Floor(x / 0.17);
        }
        if (x < 6)
        {
                return 34;
        }
        return 35 + (int)Math.Floor((x - 6) / 0.25);
    }

    private double[] Smooth(double[] lst)
    {
        var lstbar = new double[T];
        var windowSum = 0.0;
        
        for (int i = 0; i < Math.Min(500, T); i+=granularity)
            windowSum += lst[i];

        for (int s = 0; s < T; s+=granularity)
        {
            lstbar[s] = 0.001 * windowSum * granularity;
            if (s + 500 < T)
                windowSum += lst[s + 500];
            if (s - 500 >= 0)
                windowSum -= lst[s - 500];
        }
        return lstbar;
    }

    private double[] Smooth2(double[] lst)
    {
        var lstbar = new double[T];
        var windowSum = 0.0;
        var windowLen = Math.Min(500, T);

        for (int i = 0; i < windowLen; i+=granularity)
            windowSum += lst[i];

        for (int s = 0; s < T; s+=granularity)
        {
            lstbar[s] = windowSum / windowLen * granularity;

            if (s + 500 < T)
            {
                windowSum += lst[s + 500];
                windowLen+=granularity;
            }
            if (s - 500 >= 0)
            {
                windowSum -= lst[s - 500];
                windowLen-=granularity;
            }
        }
        return lstbar;
    }

    private double JackNerfer(double delta)
    {   
        return 1 - 7 * Math.Pow(10, -5) * Math.Pow(0.15 + Math.Abs(delta - 0.08), -4);
    }

    private (double[] Jbar, double[][] deltaKs) CalculateSection23()
    {
        var JKs = new double[K][];
        var deltaKs = new double[K][];

        // Fill JKs and deltaKs with values
        for (int k = 0; k < K; k++)
        {
            JKs[k] = new double[T];
            deltaKs[k] = new double[T];
            Array.Fill(deltaKs[k], 1e9);

            for (int i = 0; i < noteSeqByColumn[k].Count - 1; i++)
            {
                var delta = 0.001 * (noteSeqByColumn[k][i + 1].H - noteSeqByColumn[k][i].H);
                var val = Math.Pow(delta * (delta + lambda_1 * Math.Pow(x, 0.25)), -1) * JackNerfer(delta);

                // Define start and end for filling the range in deltaKs and JKs
                int start = noteSeqByColumn[k][i].H;
                int end = noteSeqByColumn[k][i + 1].H;
                int length = end - start;

                // Use Span to fill subarrays
                var deltaSpan = new Span<double>(deltaKs[k], start, length);
                deltaSpan.Fill(delta);

                var JKsSpan = new Span<double>(JKs[k], start, length);
                JKsSpan.Fill(val);
            }
        }

        // Smooth the JKs array
        var JbarKs = new double[K][];
        for (int k = 0; k < K; k++)
            JbarKs[k] = Smooth(JKs[k]);

        // Calculate Jbar
        var Jbar = new double[T];
        for (int s = 0; s < T; s+=granularity)
        {
            double weightedSum = 0;
            double weightSum = 0;

            // Replace list allocation with direct accumulation
            for (int i = 0; i < K; i++)
            {
                double val = JbarKs[i][s];
                double weight = 1.0 / deltaKs[i][s];

                weightSum += weight;
                weightedSum += Math.Pow(Math.Max(val, 0), lambda_n) * weight;
            }

            weightSum = Math.Max(1e-9, weightSum);
            Jbar[s] = Math.Pow(weightedSum / weightSum, 1.0 / lambda_n);
        }

        return (Jbar, deltaKs);
    }

    private double[] CalculateSection24()
    {
        var XKs = new double[K + 1][];
        for (int k = 0; k <= K; k++)
        {
            XKs[k] = new double[T];
            var notesInPair = k == 0 ? noteSeqByColumn[0] :
                             k == K ? noteSeqByColumn[K - 1] :
                             noteSeqByColumn[k - 1].Concat(noteSeqByColumn[k])
                                 .OrderBy(n => n.H).ToList();
            for (int i = 1; i < notesInPair.Count; i++)
            {
                var delta = 0.001 * (notesInPair[i].H - notesInPair[i - 1].H);
                var val = 0.16 * Math.Pow(Math.Max(x, delta), -2);

                for (int s = notesInPair[i - 1].H; s < notesInPair[i].H; s++)
                {
                    XKs[k][s] = val;
                }
            }
        }

        var X = new double[T];
        for (int s = 0; s < T; s+=granularity)
        {
            X[s] = Enumerable.Range(0, K + 1)
                .Sum(k => XKs[k][s] * crossMatrix[K][k]);
        }

        return Smooth(X);
    }

    private double[] CalculateSection25()
    {
        var P = new double[T];
        var LNBodies = new double[T];

        foreach (var note in LNSeq)
        {
            var t1 = Math.Min(note.H + 80, note.T);
            for (int t = note.H; t < t1; t++)
                LNBodies[t] += 0.5;
            for (int t = t1; t < note.T; t++)
                LNBodies[t] += 1;
        }

        double B(double delta)
        {
            var val = 7.5 / delta;
            if (160 < val && val < 360)
                return 1 + 1.4 * Math.Pow(10, -7) * (val - 160) * Math.Pow(val - 360, 2);
            return 1;
        }

        for (int i = 0; i < noteSeq.Count - 1; i++)
        {
            var delta = 0.001 * (noteSeq[i + 1].H - noteSeq[i].H);
            if (delta < Math.Pow(10, -9))
            {
                P[noteSeq[i].H] += 1000 * Math.Pow(0.02 * (4/x - lambda_3), 1.0/4);
            }
            else
            {
                var h_l = noteSeq[i].H;
                var h_r = noteSeq[i + 1].H;
                var v = 1 + lambda_2 * 0.001 * LNBodies.Skip(h_l).Take(h_r - h_l).Sum();

                if (delta < 2 * x / 3)
                {
                    var baseVal = Math.Pow(0.08 * Math.Pow(x, -1) * 
                        (1 - lambda_3 * Math.Pow(x, -1) * Math.Pow(delta - x/2, 2)), 1.0/4) * 
                        B(delta) * v / delta;
                    
                    for (int s = h_l; s < h_r; s++)
                        P[s] += baseVal;
                }
                else
                {
                    var baseVal = Math.Pow(0.08 * Math.Pow(x, -1) * 
                        (1 - lambda_3 * Math.Pow(x, -1) * Math.Pow(x/6, 2)), 1.0/4) * 
                        B(delta) * v / delta;
                    
                    for (int s = h_l; s < h_r; s++)
                        P[s] += baseVal;
                }
            }
        }

        return Smooth(P);
    }

    private (double[] Abar, int[] KS) CalculateSection26(double[][] deltaKs)
    {
        var KUKs = new bool[K][];
        for (int k = 0; k < K; k++)
        {
            KUKs[k] = new bool[T];
        }

        foreach (var note in noteSeq)
        {
            int startTime = Math.Max(0, note.H - 500);
            int endTime = note.T < 0 ? Math.Min(note.H + 500, T - 1) : Math.Min(note.T + 500, T - 1);

            for (int s = startTime; s < endTime; s++)
            {
                KUKs[note.K][s] = true;
            }
        }

        var KS = new int[T];
        var A = new double[T];
        Array.Fill(A, 1);

        var dks = new double[K - 1][];
        for (int k = 0; k < K - 1; k++)
        {
            dks[k] = new double[T];
        }

        for (int s = 0; s < T; s+=granularity)
        {
            var cols = new List<int>(K);
            for (int k = 0; k < K; k++)
            {
                if (KUKs[k][s])
                    cols.Add(k);
            }
            KS[s] = Math.Max(cols.Count, 1);

            for (int i = 0; i < cols.Count - 1; i++)
            {
                int col1 = cols[i];
                int col2 = cols[i + 1];

                dks[col1][s] = Math.Abs(deltaKs[col1][s] - deltaKs[col2][s]) +
                            Math.Max(0, Math.Max(deltaKs[col1][s], deltaKs[col2][s]) - 0.3);

                double maxDelta = Math.Max(deltaKs[col1][s], deltaKs[col2][s]);
                if (dks[col1][s] < 0.02)
                {
                    A[s] *= Math.Min(0.75 + 0.5 * maxDelta, 1);
                }
                else if (dks[col1][s] < 0.07)
                {
                    A[s] *= Math.Min(0.65 + 5 * dks[col1][s] + 0.5 * maxDelta, 1);
                }
            }
        }
        
        return (Smooth2(A), KS);
    }

    private Note FindNextNoteInColumn(Note note, List<Note> columnNotes)
    {
        int index = columnNotes.BinarySearch(note, Comparer<Note>.Create((a, b) => a.H.CompareTo(b.H)));

        // If the exact element is not found, BinarySearch returns a bitwise complement of the index.
        // Convert it to the nearest index of an element >= note.H
        if (index < 0) index = ~index;

        return index + 1 < columnNotes.Count ? 
            columnNotes[index + 1] : 
            new Note(0, (int)Math.Pow(10, 9), (int)Math.Pow(10, 9));
    }

    private (double[] Rbar, double[] Is) CalculateSection27()
    {
        var I = new double[LNSeq.Count];
        for (int i = 0; i < tailSeq.Count; i++)
        {
            var (k, h_i, t_i) = (tailSeq[i].K, tailSeq[i].H, tailSeq[i].T);
            var nextNote = FindNextNoteInColumn(tailSeq[i], noteSeqByColumn[k]);
            var (_, h_j, t_j) = (nextNote.K, nextNote.H, nextNote.T);

            var I_h = 0.001 * Math.Abs(t_i - h_i - 80) / x;
            var I_t = 0.001 * Math.Abs(h_j - t_i - 80) / x;
            I[i] = 2 / (2 + Math.Exp(-5 * (I_h - 0.75)) + Math.Exp(-5 * (I_t - 0.75)));
        }

        var Is = new double[T];
        var R = new double[T];

        for (int i = 0; i < tailSeq.Count - 1; i++)
        {
            var delta_r = 0.001 * (tailSeq[i + 1].T - tailSeq[i].T);
            for (int s = tailSeq[i].T; s < tailSeq[i + 1].T; s++)
            {
                Is[s] = 1 + I[i];
                R[s] = 0.08 * Math.Pow(delta_r, -1.0/2) * Math.Pow(x, -1) * (1 + lambda_4 * (I[i] + I[i + 1]));
            }
        }

        return (Smooth(R), Is);
    }

    public void ForwardFill(double[] array)
    {
        double lastValidValue = 0;  // Use initialValue for leading NaNs and 0s

        for (int i = 0; i < array.Length; i++)
        {
            if (!double.IsNaN(array[i]) && array[i] != 0)  // Check if the current value is valid (not NaN or 0)
            {
                lastValidValue = array[i];
            }
            else
            {
                array[i] = lastValidValue;  // Replace NaN or 0 with last valid value or initial value
            }
        }
    }

    private double CalculateSection3(double[] Jbar, double[] Xbar, double[] Pbar, 
        double[] Abar, double[] Rbar, int[] KS)
    {
        var C = new double[T];
        int start = 0, end = 0;

        for (int t = 0; t < T; t++)
        {
            while (start < noteSeq.Count && noteSeq[start].H < t - 500)
                start++;
            while (end < noteSeq.Count && noteSeq[end].H < t + 500)
                end++;
            C[t] = end - start;
        }

        var S = new double[T];
        var D = new double[T];

        for (int t = 0; t < T; t++)
        {
            // Ensure all values are non-negative
            Jbar[t] = Math.Max(0, Jbar[t]);
            Xbar[t] = Math.Max(0, Xbar[t]);
            Pbar[t] = Math.Max(0, Pbar[t]);
            Abar[t] = Math.Max(0, Abar[t]);
            Rbar[t] = Math.Max(0, Rbar[t]);

            var term1 = Math.Pow(w_0 * Math.Pow(Math.Pow(Abar[t], 3.0/KS[t]) * Jbar[t], 1.5), 1);
            var term2 = Math.Pow((1 - w_0) * Math.Pow(Math.Pow(Abar[t], 2.0/3) * 
                (0.8 * Pbar[t] + Rbar[t]), 1.5), 1);
            S[t] = Math.Pow(term1 + term2, 2.0/3);

            var T_t = (Math.Pow(Abar[t], 3.0/KS[t]) * Xbar[t]) / (Xbar[t] + S[t] + 1);
            D[t] = w_1 * Math.Pow(S[t], 1.0/2) * Math.Pow(T_t, p_1) + S[t] * w_2;
        }

        ForwardFill(D);
        ForwardFill(C);

        var weightedSum = 0.0;
        var weightSum = C.Sum();
        for (int t = 0; t < T; t++)
        {
            weightedSum += Math.Pow(D[t], lambda_n) * C[t];
        }

        var SR = Math.Pow(weightedSum / weightSum, 1.0/lambda_n);

        SR = Math.Pow(SR, p_0) / Math.Pow(8, p_0) * 8;
        SR *= (noteSeq.Count + 0.5 * LNSeq.Count) / (noteSeq.Count + 0.5 * LNSeq.Count + 60);

        if (SR <= 2)
            SR = Math.Sqrt(SR * 2);

        return SR;
    }
}

// Using the MACalculator
public class Program
{
    public static void Main(string[] args)
    {
        // Input specifications:
        // 1. For each note (int key, int hit, int release), int release must be set to -1 for a rice note. It MUST NOT simply be int hit.
        // 2. Each column MUST contain at least one note.
        // 3. No overlapping notes allowed.
        string filePath = "note_seq.txt";
        var notes = LoadNoteSequenceFromFile(filePath);
        int keyCount = 4;
        var calculator = new MACalculator();
        var result = calculator.Calculate(notes, keyCount);

        Console.WriteLine($"\nCalculated MA Score: {result:F4}");
    }

    private static List<(int k, int h, int t)> LoadNoteSequenceFromFile(string filePath)
    {
        var notes = new List<(int k, int h, int t)>();

        try
        {
            foreach (var line in File.ReadLines(filePath))
            {
                var parts = line.Split(new[] { ' ', ',', ':' }, StringSplitOptions.RemoveEmptyEntries);

                // Parse "Key: value", "Start: value", "End: value"
                if (parts.Length >= 6 &&
                    int.TryParse(parts[1], out int k) &&
                    int.TryParse(parts[3], out int h) &&
                    int.TryParse(parts[5], out int t))
                {
                    notes.Add((k, h, t));
                }
            }
        }
        catch (Exception ex)
        {
            Console.WriteLine($"Error reading file: {ex.Message}");
        }

        return notes;
    }
}

