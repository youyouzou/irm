import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple


# 动作类别映射，对应你说的 7 个 classes
ACTION_TO_LABEL: Dict[str, int] = {
    "empty": 0,
    "jump": 1,
    "pick": 2,
    "run": 3,
    "sit": 4,
    "walk": 5,
    "wave": 6,
}


def load_env_dir(env_dir: Path, env_id: int) -> Tuple[List[np.ndarray], List[int], List[int]]:
    """
    从单个环境文件夹读取所有 .npy 文件。
    文件命名格式类似： empty_jianfei_0.npy
    - 动作类别: 由文件名第一个字段决定 (empty/jump/pick/run/sit/walk/wave)
    - 人员信息: 第二个字段（目前不作为标签使用）
    - env_id: 由外部传入（0,1,2 对应 5300-1/2/3）
    """
    xs: List[np.ndarray] = []
    ys: List[int] = []
    envs: List[int] = []

    for npy_path in sorted(env_dir.glob("*.npy")):
        name = npy_path.stem  # e.g. "empty_jianfei_0"
        parts = name.split("_")
        if len(parts) < 1:
            continue
        action = parts[0]
        if action not in ACTION_TO_LABEL:
            print(f"Warning: unknown action '{action}' in file {npy_path.name}, skipped.")
            continue
        label = ACTION_TO_LABEL[action]

        x = np.load(npy_path)  # 形状应为 (2000, 30)
        xs.append(x)
        ys.append(label)
        envs.append(env_id)

    return xs, ys, envs


def main() -> None:
    base = Path("data/processed")

    # 三个环境目录及其 env_id
    env_specs = [
        ("5300-1_npy", 0),
        ("5300-2_npy", 1),
        ("5300-3_npy", 2),
    ]

    all_x: List[np.ndarray] = []
    all_y: List[int] = []
    all_env: List[int] = []
    env_to_indices: Dict[int, List[int]] = {0: [], 1: [], 2: []}

    print("Loading data from 3 environments...")

    for env_name, env_id in env_specs:
        env_dir = base / env_name
        if not env_dir.is_dir():
            raise FileNotFoundError(f"Environment directory not found: {env_dir}")

        xs, ys, envs = load_env_dir(env_dir, env_id)
        start_idx = len(all_x)
        all_x.extend(xs)
        all_y.extend(ys)
        all_env.extend(envs)

        env_to_indices[env_id].extend(range(start_idx, start_idx + len(xs)))
        print(f"  Env {env_id} ({env_name}): {len(xs)} samples")

    x_all = np.stack(all_x, axis=0).astype(np.float32)  # [N, 2000, 30]
    y_all = np.array(all_y, dtype=np.int64)
    env_all = np.array(all_env, dtype=np.int64)

    print("Total samples:", x_all.shape[0])
    print("x_all shape:", x_all.shape)
    print("y_all shape:", y_all.shape)
    print("env_all shape:", env_all.shape)

    # 划分策略（与你任务匹配）:
    # - 环境 0 和 1（5300-1, 5300-2）：作为训练 + 验证环境
    #   - 各自内部按 80% 训练 / 20% 验证划分
    # - 环境 2（5300-3）：全部作为测试环境（新环境，用来做 DG 评估）

    rng = np.random.default_rng(seed=42)

    train_indices: List[int] = []
    val_indices: List[int] = []
    test_indices: List[int] = []

    for env_id in [0, 1]:
        idxs = np.array(env_to_indices[env_id], dtype=np.int64)
        perm = rng.permutation(len(idxs))
        split = int(len(idxs) * 0.8)
        train_indices.extend(idxs[perm[:split]].tolist())
        val_indices.extend(idxs[perm[split:]].tolist())

    # 环境 2 全部用于测试
    test_indices.extend(env_to_indices[2])

    train_indices = np.array(train_indices, dtype=np.int64)
    val_indices = np.array(val_indices, dtype=np.int64)
    test_indices = np.array(test_indices, dtype=np.int64)

    print(f"Train samples: {len(train_indices)}")
    print(f"Val samples:   {len(val_indices)}")
    print(f"Test samples:  {len(test_indices)} (env 2 only)")

    x_train = x_all[train_indices]
    y_train = y_all[train_indices]
    env_train = env_all[train_indices]

    x_val = x_all[val_indices]
    y_val = y_all[val_indices]
    env_val = env_all[val_indices]

    x_test = x_all[test_indices]
    y_test = y_all[test_indices]
    env_test = env_all[test_indices]

    base.mkdir(parents=True, exist_ok=True)
    np.savez(base / "train.npz", x=x_train, y=y_train, env=env_train)
    np.savez(base / "val.npz", x=x_val, y=y_val, env=env_val)
    np.savez(base / "test.npz", x=x_test, y=y_test, env=env_test)

    num_classes = len(ACTION_TO_LABEL)
    print(f"Saved train/val/test npz to {base.resolve()}")
    print(f"Detected num_classes = {num_classes}")
    print("Env distribution in train/val/test:")
    for split_name, env_arr in [
        ("train", env_train),
        ("val", env_val),
        ("test", env_test),
    ]:
        unique, counts = np.unique(env_arr, return_counts=True)
        stats = ", ".join(f"env{int(e)}:{int(c)}" for e, c in zip(unique, counts))
        print(f"  {split_name}: {stats}")


if __name__ == "__main__":
    main()

'''
python scripts\make_npz_from_npy.py
'''