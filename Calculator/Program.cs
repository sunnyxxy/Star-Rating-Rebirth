using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using Malody.Chart;

namespace NoteSeqRating
{
    class Program
    {
        static void Main(string[] args)
        {
            string filePath = "note_seq.txt";
            if (!File.Exists(filePath))
            {
                Console.WriteLine($"Error: File '{filePath}' not found.");
                return;
            }

            List<Note> noteSeq = new List<Note>();

            // Each line is expected to be in the format:
            // Key: 0, Start: 3417, End: 3773
            foreach (string line in File.ReadLines(filePath))
            {
                if (string.IsNullOrWhiteSpace(line))
                    continue;

                try
                {
                    // Split on commas.
                    string[] parts = line.Split(',');
                    // parts[0] should be "Key: <num>"
                    // parts[1] should be " Start: <num>"
                    // parts[2] should be " End: <num>"
                    int key = ParsePart(parts[0], "Key:");
                    int start = ParsePart(parts[1], "Start:");
                    int end = ParsePart(parts[2], "End:");

                    // Create a Note. Here we treat "Start" as the head (hit) time
                    // and "End" as the tail. (If End should be -1 for non–long notes,
                    // modify as needed.)
                    noteSeq.Add(new Note(key, start, end));
                }
                catch (Exception ex)
                {
                    Console.WriteLine($"Error parsing line '{line}': {ex.Message}");
                }
            }

            if (noteSeq.Count == 0)
            {
                Console.WriteLine("No valid notes found in the file.");
                return;
            }

            // Determine the number of keys (K) as the maximum key + 1.
            int K = noteSeq.Max(n => n.Column) + 1;

            // Set the mod and other parameters.
            // For example, "NM" for normal (no DT/HT adjustment).

            // Compute the rating using MACalculator.
            double rating = MACalculator.Calculate(noteSeq, K);

            Console.WriteLine($"Rating: {rating}");
        }

        /// <summary>
        /// Parses a part of a line given a prefix label.
        /// For example, given part "Key: 0" and label "Key:" it returns 0.
        /// </summary>
        private static int ParsePart(string part, string label)
        {
            part = part.Trim();
            if (!part.StartsWith(label))
                throw new FormatException($"Expected part to start with '{label}'");
            string numberStr = part.Substring(label.Length).Trim();
            // Use InvariantCulture to ensure dot/decimal formatting
            return int.Parse(numberStr, CultureInfo.InvariantCulture);
        }
    }
}
