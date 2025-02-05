**The algorithm paper (.pdf file) is slightly outdated. These are the modifications:**

1. Change the norm from L4 to L5;
2. Change the Stream Booster such that it boosts actual streams rather than chordstreams;
3. Add a JackNerfer;
4. Use local key-count (which calculates the number of non-empty columns in the window) in certain calculations;
5. Ignore locally empty columns for Unevenness.

**Updates on Feb 2-3, 2025:**

1. Increase StreamBooster to 1.7e-7;
2. Change (lambda_2, lambda_4) to (6, 0.8);
3. Incorporate difficulty at percentile points in final SR calculation;
4. Improve the dks computation to avoid unwanted suppression of Abar;
5. As a result of 4, adjust the final SR formula to maintain scaling.

---

## Steps to use this program:

### **Method 1: Download and run the executable file (`.exe`)**

1. Go to the [Releases](https://github.com/sunnyxxy/Star-Rating-Rebirth/releases) section of the repository;
2. Download the latest version of the `srcalc.exe` file;
3. Place the `srcalc.exe` file and your `.osu` files in the same folder;
4. Run `srcalc.exe`.


### **Method 2: Run the source code**

1. Make sure you have a python environment;
2. Create a Test folder next to the python files;
3. Throw .osu files in the Test folder;
4. Run test.py.
