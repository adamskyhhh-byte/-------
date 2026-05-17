from __future__ import annotations
"""
少样本数据划分脚本。

这个文件负责从原始 Drebin CSV 中生成少样本实验需要的数据：
1. 先固定抽出一个共享测试集 test.csv，每类默认 100 条。
2. 再从测试集之外抽训练集 train.csv，每类抽 K 条，所以训练集总数是 2K。
3. 不同 K 共用同一个 test.csv，保证 K=1/3/5/10 的结果可以公平对比。
4. 额外保存 metadata.json，记录随机种子、样本数量、源索引、test 文件哈希等信息。

常用命令：
python prepare_fewshot_split.py --k 5 --seed 42 --test-per-class 100
"""

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd

from fewshot_utils import normalize_label


DEFAULT_DATA_PATH = "data/drebin-215-dataset-5560malware-9476-benign.csv"
DEFAULT_OUTPUT_ROOT = "data/processed"


def parse_args() -> argparse.Namespace:
    """定义命令行参数。运行脚本时传入的 --k、--seed 等都在这里解析。"""
    parser = argparse.ArgumentParser(description="Prepare fixed-test few-shot Drebin splits.")
    parser.add_argument("--k", type=int, required=True, help="Number of training samples per class.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--test-per-class", type=int, default=100)
    parser.add_argument("--data-path", default=DEFAULT_DATA_PATH)
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--label-col", default="class")
    return parser.parse_args()


def count_by_label(df: pd.DataFrame, label_col: str) -> dict[str, int]:
    """统计 B/S 两类各有多少条，并顺手给出总数。"""
    counts = df[label_col].value_counts().to_dict()
    return {
        "B": int(counts.get("B", 0)),
        "S": int(counts.get("S", 0)),
        "total": int(len(df)),
    }


def sha256_file(path: Path) -> str:
    """计算文件 SHA256，用来证明不同 K 共用的是同一个 test.csv。"""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        # 分块读取，避免文件很大时一次性读入内存。
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_sample_size(df: pd.DataFrame, label_col: str, per_class: int, purpose: str) -> None:
    """抽样前检查每个类别的样本数是否足够。"""
    for label in ("B", "S"):
        available = int((df[label_col] == label).sum())
        if available < per_class:
            raise ValueError(
                f"Not enough {label} rows for {purpose}: need {per_class}, available {available}"
            )


def sample_fixed_test(df: pd.DataFrame, label_col: str, test_per_class: int, seed: int) -> pd.DataFrame:
    """先从全量数据中每类抽固定数量样本，形成共享测试集。"""
    validate_sample_size(df, label_col, test_per_class, "test split")
    parts = []
    for offset, label in enumerate(("B", "S")):
        # B 和 S 使用 seed 的不同偏移，避免两类抽样使用完全相同的随机流。
        part = df[df[label_col] == label].sample(n=test_per_class, random_state=seed + offset)
        parts.append(part)
    # 最后把 B/S 混在一起打乱，避免 CSV 里先全是 B 后全是 S。
    return pd.concat(parts).sample(frac=1, random_state=seed).copy()


def sample_nested_train(pool: pd.DataFrame, label_col: str, k: int, seed: int) -> pd.DataFrame:
    """从测试集之外抽训练集，并保证小 K 的样本包含在大 K 中。"""
    validate_sample_size(pool, label_col, k, "train split")
    parts = []
    for offset, label in enumerate(("B", "S")):
        # 关键点：先给每一类生成固定随机顺序，再取前 K 个。
        # 因此 K=1 的样本天然包含在 K=3 中，K=3 又包含在 K=5 中。
        ordered = pool[pool[label_col] == label].sample(frac=1, random_state=seed + 1000 + offset)
        parts.append(ordered.head(k))
    return pd.concat(parts).sample(frac=1, random_state=seed + 2000).copy()


def main() -> None:
    args = parse_args()
    if args.k < 1:
        raise ValueError("--k must be at least 1")
    if args.test_per_class < 1:
        raise ValueError("--test-per-class must be at least 1")

    # 生成方案要求的目录：data/processed/fewshot_seed42_test100/k5/。
    data_path = Path(args.data_path)
    output_root = Path(args.output_root)
    split_root = output_root / f"fewshot_seed{args.seed}_test{args.test_per_class}"
    k_root = split_root / f"k{args.k}"
    split_root.mkdir(parents=True, exist_ok=True)
    k_root.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(data_path, low_memory=False)
    if args.label_col not in df.columns:
        raise ValueError(f"Missing label column: {args.label_col}")
    df = df.copy()
    # 保存到 CSV 前就把标签规范成 B/S，确保后续所有脚本看到的标签一致。
    df[args.label_col] = df[args.label_col].map(normalize_label)

    # 先抽 test，再从 test 之外抽 train，避免训练集和测试集重叠。
    test_df = sample_fixed_test(df, args.label_col, args.test_per_class, args.seed)
    train_pool = df.drop(index=test_df.index)
    train_df = sample_nested_train(train_pool, args.label_col, args.k, args.seed)

    # 这是安全检查：只要出现重叠就直接报错，不生成有数据泄漏风险的结果。
    overlap = set(test_df.index).intersection(set(train_df.index))
    if overlap:
        raise RuntimeError(f"Train/test overlap detected: {sorted(overlap)[:10]}")

    # 源索引只写到 JSON 元数据，不写进 CSV，避免污染特征列。
    test_source_indices = [int(i) for i in test_df.index.tolist()]
    train_source_indices = [int(i) for i in train_df.index.tolist()]

    test_path = split_root / "test.csv"
    train_path = k_root / "train.csv"
    test_df.to_csv(test_path, index=False)
    train_df.to_csv(train_path, index=False)
    test_sha = sha256_file(test_path)

    # test_metadata 描述共享测试集本身。
    test_metadata = {
        "seed": args.seed,
        "test_per_class": args.test_per_class,
        "label_col": args.label_col,
        "label_format": "B/S",
        "test_size": count_by_label(test_df, args.label_col),
        "test_sha256": test_sha,
        "test_source_indices": test_source_indices,
    }
    # k 目录下的 metadata 描述本次 K 值对应的训练集和测试集。
    run_metadata = {
        "k": args.k,
        "k_semantics": "per_class",
        "seed": args.seed,
        "test_per_class": args.test_per_class,
        "train_size": count_by_label(train_df, args.label_col),
        "test_size": count_by_label(test_df, args.label_col),
        "test_sha256": test_sha,
        "train_source_indices": train_source_indices,
        "test_source_indices": test_source_indices,
    }

    (split_root / "test_metadata.json").write_text(
        json.dumps(test_metadata, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (k_root / "metadata.json").write_text(
        json.dumps(run_metadata, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    # 控制台输出简短摘要，方便你肉眼检查数量是否正确。
    train_size = count_by_label(train_df, args.label_col)
    test_size = count_by_label(test_df, args.label_col)
    print(f"train.csv: B={train_size['B']}, S={train_size['S']}, total={train_size['total']}")
    print(f"test.csv:  B={test_size['B']}, S={test_size['S']}, total={test_size['total']}")
    print(f"split root: {split_root}")
    print(f"test sha256: {test_sha}")


if __name__ == "__main__":
    main()
