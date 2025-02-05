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

Steps to use this program:

1. Create a Test folder next to the python files;
2. Throw .osu files in the Test folder;
3. Run test.py.

---

## Steps to use this program:

### **Method 1: Use the pre-built command-line executable (`.exe`)**

1. Go to the [Releases](https://github.com/sunnyxxy/Star-Rating-Rebirth/releases) section of the repository;
2. Download the latest version of the `srcalc.exe` file;
3. Place the `srcalc.exe` file and your `.osu` files in the same folder;
4. Run `srcalc.exe`.

#### Command-line Options:

The program accepts the following options when running from the command line:

- `folder_path`:  
  The path to the folder containing `.osu` files.  
  Example:
    ```sh
    srcalc.exe "Test"
    ```

- `--mod -M`:  
  Apply specific mods to the SR calculation. Valid values are:
    - `NM` (No Mod)
    - `DT` (Double Time)
    - `HT` (Half Time)
    - Default: `NM`

  Example:
    ```sh
    srcalc.exe --mod DT
    ```

- `--version -V`:  
  Show the build version (including build time) and exit. This will display the algorithm version and exit without processing .osu files.  
  Example:
    ```sh
    srcalc.exe --version
    ```

  Example Usage:  
  To calculate SR for .osu files in the `D:\Test` folder using the DT mod:
  ```sh
    srcalc.exe --mod DT "D:\Test"
  ```

### **Method 2: Run the source code**

1. Make sure you have a python environment;
2. Create a Test folder next to the python files;
3. Throw .osu files in the Test folder;
4. Run test.py.