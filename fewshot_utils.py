from __future__ import annotations
"""
少样本实验公共工具文件。

这个文件不直接运行实验，而是给其他脚本复用一些基础能力：
1. 统一把标签规范成 B/S。
2. 统一读取 few-shot 划分后的 train.csv / test.csv。
3. 统一把 DataFrame 转成机器学习模型需要的 X 特征和 y 标签。
4. 统一把一行样本转换成给大模型看的“活跃特征文本”。

这样做的好处是：baseline_fewshot.py 和 llm_only_fewshot.py 不用各写一套
标签处理逻辑，后续如果标签规则变化，只需要改这里。
"""

from pathlib import Path
from typing import Any

import pandas as pd


# 项目约定：B 表示 Benign 良性，S 表示 Malware 恶意。
# 传统机器学习模型只能吃数字标签，所以在训练/评估时临时映射成 0/1。
LABEL_TO_ID = {"B": 0, "S": 1}
ID_TO_LABEL = {0: "B", 1: "S"}


def normalize_label(value: Any) -> str:
    """把各种可能出现的标签写法统一成项目规定的 B 或 S。"""
    if pd.isna(value):
        raise ValueError("Label is missing")

    # 先转字符串并去掉首尾空格，避免 " B " 这种值造成误判。
    text = str(value).strip()
    upper = text.upper()
    lowered = text.lower()

    # 原始数据里正常情况下就是 B/S，这里直接返回。
    if upper in LABEL_TO_ID:
        return upper
    # 下面兼容一些常见写法，增强脚本鲁棒性。
    if lowered in {"benign", "0"}:
        return "B"
    if lowered in {"malware", "malicious", "s", "1"}:
        return "S"

    raise ValueError(f"Unsupported label value: {value!r}")


def load_fewshot_split(split_root: str | Path, k: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """读取同一个 split_root 下指定 K 的训练集和共享测试集。"""
    split_root = Path(split_root)
    # 方案规定：训练集放在 k5/train.csv 这种目录里，测试集在根目录 test.csv。
    train_path = split_root / f"k{k}" / "train.csv"
    test_path = split_root / "test.csv"

    if not train_path.exists():
        raise FileNotFoundError(f"Missing train split: {train_path}")
    if not test_path.exists():
        raise FileNotFoundError(f"Missing test split: {test_path}")

    return pd.read_csv(train_path, low_memory=False), pd.read_csv(test_path, low_memory=False)


def dataframe_to_xy(df: pd.DataFrame, label_col: str = "class"):
    """把带 class 列的数据表转换成 X 特征矩阵和 y 数字标签。"""
    if label_col not in df.columns:
        raise ValueError(f"Missing label column: {label_col}")

    # CSV 中仍然保留 B/S 语义标签，只在进入模型前临时转为 0/1。
    y = df[label_col].map(normalize_label).map(LABEL_TO_ID).astype(int)

    # 特征列里可能有 "?"，先当作缺失值，再转数字，最后用 0 填充。
    # Drebin 特征本质是 0/1 二值特征，所以缺失按 0 处理比较稳妥。
    X = df.drop(columns=[label_col]).replace("?", pd.NA)
    X = X.apply(pd.to_numeric, errors="coerce").fillna(0)
    return X, y


def load_feature_categories(path: str | Path | None) -> dict[str, str]:
    """读取特征类别文件，返回 {特征名: 类别名} 的字典。"""
    if path is None:
        return {}

    path = Path(path)
    if not path.exists():
        return {}

    categories: dict[str, str] = {}
    # dataset-features-categories.csv 没有表头，格式是：特征名,类别名。
    raw = pd.read_csv(path, header=None)
    if raw.shape[1] < 2:
        return categories

    for feature, category in raw.iloc[:, :2].itertuples(index=False):
        if pd.isna(feature) or pd.isna(category):
            continue
        categories[str(feature)] = str(category)
    return categories


def row_to_feature_text(
    row: pd.Series,
    feature_categories: dict[str, str] | None = None,
    label_col: str = "class",
) -> str:
    """把一条样本中值为 1 的特征整理成可读文本，供 LLM prompt 使用。"""
    feature_categories = feature_categories or {}
    active_by_category: dict[str, list[str]] = {}
    active_features: list[str] = []

    for name, value in row.items():
        # class 是标签，不属于输入特征，不能泄漏给模型看。
        if name == label_col:
            continue

        # 只输出活跃特征，也就是值为 1 的特征；值为 0 或缺失的特征不写进 prompt。
        numeric = pd.to_numeric(pd.Series([value]).replace("?", pd.NA), errors="coerce").fillna(0).iloc[0]
        if numeric != 1:
            continue

        feature = str(name)
        if feature_categories:
            # 如果有类别信息，就按类别分组，LLM 读起来更清楚。
            category = feature_categories.get(feature, "Uncategorized")
            active_by_category.setdefault(category, []).append(feature)
        else:
            active_features.append(feature)

    if feature_categories:
        if not active_by_category:
            return "Active features: none"
        lines = []
        # 排序是为了让同一条样本每次生成的文本顺序稳定，方便复现实验。
        for category in sorted(active_by_category):
            features = ", ".join(sorted(active_by_category[category]))
            lines.append(f"{category}: {features}")
        return "\n".join(lines)

    if not active_features:
        return "Active features: none"
    return "Active features: " + ", ".join(sorted(active_features))
