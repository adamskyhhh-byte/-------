"""
论文方法 LLM 推理实验的 prompt 特征文本工具。

这个文件只负责把一条 Drebin 样本转换成可以放进 LLM prompt 的特征文本。
它提供 raw 和 semantic 两种表达方式，保证两组实验除了“特征怎么写给 LLM 看”
之外，训练样本、测试样本、标签和评估逻辑都保持一致。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from fewshot_utils import row_to_feature_text


def load_feature_semantics(path: str | Path) -> dict[str, dict[str, Any]]:
    """读取离线生成的特征语义 JSON，返回 {特征名: 语义信息}。"""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Missing feature semantics file: {path}")

    with path.open(encoding="utf-8") as handle:
        data = json.load(handle)

    if not isinstance(data, dict):
        raise ValueError("Feature semantics JSON must be an object")
    return data


def is_active_feature(value: Any) -> bool:
    """判断一个 Drebin 特征值是否表示“激活”，也就是值为 1。"""
    numeric = pd.to_numeric(pd.Series([value]).replace("?", pd.NA), errors="coerce").fillna(0).iloc[0]
    return numeric == 1


def row_to_raw_feature_text(
    row: pd.Series,
    feature_categories: dict[str, str] | None = None,
    label_col: str = "class",
) -> str:
    """把样本转换为 raw 表达：按类别列出激活的原始特征名。"""
    return row_to_feature_text(row, feature_categories=feature_categories, label_col=label_col)


def row_to_semantic_feature_text(
    row: pd.Series,
    feature_categories: dict[str, str] | None,
    feature_semantics: dict[str, dict[str, Any]],
    label_col: str = "class",
) -> str:
    """把样本转换为 semantic 表达：按类别列出“特征名: 中文语义描述”。"""
    feature_categories = feature_categories or {}
    active_by_category: dict[str, list[tuple[str, str]]] = {}

    for name, value in row.items():
        if name == label_col:
            continue
        if not is_active_feature(value):
            continue

        feature = str(name)
        category = feature_categories.get(feature, "Uncategorized")
        semantic_record = feature_semantics.get(feature, {})
        description = str(semantic_record.get("description", "")).strip()

        # 文档要求：缺少语义描述时退回原始特征名，不中断实验。
        if not description:
            description = feature

        active_by_category.setdefault(category, []).append((feature, description))

    if not active_by_category:
        return "Active features: none"

    lines: list[str] = []
    for category in sorted(active_by_category):
        lines.append(f"{category}:")
        for feature, description in sorted(active_by_category[category]):
            lines.append(f"- {feature}: {description}")
        lines.append("")

    return "\n".join(lines).strip()


def row_to_feature_expr_text(
    row: pd.Series,
    feature_expr: str,
    feature_categories: dict[str, str] | None,
    feature_semantics: dict[str, dict[str, Any]] | None = None,
    label_col: str = "class",
) -> str:
    """根据 feature_expr 统一分发到 raw 或 semantic 文本构造函数。"""
    if feature_expr == "raw":
        return row_to_raw_feature_text(row, feature_categories, label_col=label_col)
    if feature_expr == "semantic":
        if feature_semantics is None:
            raise ValueError("feature_semantics is required when feature_expr='semantic'")
        return row_to_semantic_feature_text(
            row,
            feature_categories,
            feature_semantics,
            label_col=label_col,
        )
    raise ValueError(f"Unsupported feature expression: {feature_expr}")
