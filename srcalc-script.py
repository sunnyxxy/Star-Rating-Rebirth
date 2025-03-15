import argparse
import sys
from enum import Enum
from pathlib import Path

import algorithm


class Mod(Enum):
    NM = "NM"
    DT = "DT"
    HT = "HT"


def main():
    parser = argparse.ArgumentParser(description="Calculate SR for osu! beatmaps.")
    parser.add_argument("folder_path", nargs='?', default=Path.cwd(), type=Path, help='Path to the folder containing .osu files.')
    parser.add_argument("--mod", "-M", type=str, choices=[mod.value for mod in Mod], default=Mod.NM.value, help='Mod to apply (NM, DT, HT).')
    parser.add_argument("--version", "-V", action="store_true", help="Show build version (build time) and exit.")
    args = parser.parse_args()

    def resource_path(relative_path: str) -> Path:
        base_path = Path(getattr(sys, '_MEIPASS', Path(__file__).parent))
        return base_path / relative_path

    build_time_file = resource_path("build_time")
    if build_time_file.exists():
        version_str = f" (algorithm version: {build_time_file.read_text(encoding="utf-8").strip()})"
    else:
        version_str = ""
    credit_str = f"Star-Rating-Rebirth by [Crz]sunnyxxy{version_str}"

    if args.version:
        print(credit_str)
        sys.exit(0)

    folder_path = args.folder_path
    if not folder_path.is_dir():
        print(f"Error: {folder_path} is not a valid directory.")
        sys.exit(1)

    print(credit_str)

    mod = args.mod
    print(f"Dir: {folder_path}, Mod: {mod}\n")

    while True:
        for file in Path(folder_path).iterdir():
            if file.suffix == ".osu":
                result = algorithm.calculate(file, mod)
                print(f"({mod}) {file.stem} | {result:.4f}")
        try:
            input("SR calculation completed. Press Enter to run again or 'Ctrl+C' to exit.")
            print()
        except KeyboardInterrupt:
            sys.exit(0)


if __name__ == "__main__":
    main()
