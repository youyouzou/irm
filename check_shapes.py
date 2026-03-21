import numpy as np
from pathlib import Path


def main() -> None:
    base = Path("data/processed")
    files = ["x_train.npy", "y_train.npy", "x_test.npy", "y_test.npy"]
    for name in files:
        path = base / name
        if not path.is_file():
            print(f"{name}: NOT FOUND")
            continue
        arr = np.load(path)
        print(f"{name}: shape={arr.shape}, dtype={arr.dtype}")


if __name__ == "__main__":
    main()

