"""
论文方法 LLM 推理实验的 prompt 特征文本工具。

这个文件只负责把一条 Drebin 样本转换成可以放进 LLM prompt 的特征文本。
它提供三种表达方式，保证除了"特征怎么写给 LLM 看"之外，
训练样本、测试样本、标签和评估逻辑都保持一致：

- raw：按类别列出激活的原始特征名。
- semantic-risky-old：旧 `feature_semantics_gemma.json` 的风险词描述（失败对照）。
- semantic-neutral-fixed：4 字段中性描述 + 训练池统计（修复版）。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from fewshot_utils import row_to_feature_text


SEMANTIC_RISKY_OLD = "semantic-risky-old"
SEMANTIC_NEUTRAL_FIXED = "semantic-neutral-fixed"
SEMANTIC_ALIAS = "semantic"  # 兼容旧 CLI，等价于 semantic-risky-old


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


def load_feature_stats(path: str | Path | None) -> dict[str, dict[str, Any]]:
    """读取 feature_stats.csv 并以 feature 名为键返回。"""
    if path is None:
        return {}
    path = Path(path)
    if not path.exists():
        return {}
    df = pd.read_csv(path)
    if "feature" not in df.columns:
        return {}
    return {str(row["feature"]): row.to_dict() for _, row in df.iterrows()}


def is_active_feature(value: Any) -> bool:
    """判断一个 Drebin 特征值是否表示"激活"，也就是值为 1。"""
    numeric = pd.to_numeric(pd.Series([value]).replace("?", pd.NA), errors="coerce").fillna(0).iloc[0]
    return numeric == 1


def row_to_raw_feature_text(
    row: pd.Series,
    feature_categories: dict[str, str] | None = None,
    label_col: str = "class",
) -> str:
    """按类别列出激活的原始特征名（raw 表达）。"""
    return row_to_feature_text(row, feature_categories=feature_categories, label_col=label_col)


def row_to_semantic_risky_old_text(
    row: pd.Series,
    feature_categories: dict[str, str] | None,
    feature_semantics: dict[str, dict[str, Any]],
    label_col: str = "class",
) -> str:
    """按类别列出"特征名: 旧风险词描述"（semantic-risky-old）。

    这里不附带训练池统计；保持旧 `feature_semantics_gemma.json` 的原始风险词
    描述，用作偏 S 的失败对照。
    """
    feature_categories = feature_categories or {}
    active_by_category: dict[str, list[tuple[str, str]]] = {}

    for name, value in row.items():
        if name == label_col:
            continue
        if not is_active_feature(value):
            continue

        feature = str(name)
        category = feature_categories.get(feature, "Uncategorized")
        semantic_record = feature_semantics.get(feature, {}) if isinstance(feature_semantics, dict) else {}
        description = str(
            semantic_record.get("description")
            or semantic_record.get("meaning")
            or semantic_record.get("neutral_description")
            or ""
        ).strip()
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


def _format_train_stat(stats_record: dict[str, Any]) -> str:
    """渲染单条 feature 的训练池统计句。"""
    direction = str(stats_record.get("stat_direction", "")).strip() or "weak_or_mixed"
    p_b = stats_record.get("p_feature_given_B")
    p_s = stats_record.get("p_feature_given_S")

    def _fmt(value: Any) -> str:
        try:
            return f"{float(value):.4f}"
        except (TypeError, ValueError):
            return "NA"

    return f"{direction}, P(feature|B)={_fmt(p_b)}, P(feature|S)={_fmt(p_s)}"


def row_to_semantic_neutral_text(
    row: pd.Series,
    feature_categories: dict[str, str] | None,
    semantics_neutral: dict[str, dict[str, Any]],
    feature_stats: dict[str, dict[str, Any]] | None = None,
    label_col: str = "class",
) -> str:
    """semantic-neutral-fixed 渲染：每个 active feature 一个 4 字段 block。

    输出格式（每个 active feature）：

        - feature: <name>
          category: <category>
          meaning: <neutral meaning, no risk-loaded words>
          train_stat: <leans_B/leans_S/weak_or_mixed, P(feature|B)=..., P(feature|S)=...>

    `meaning` 来自 `feature_semantics_neutral_stats.json` 的 `meaning`
    （兼容 `description`/`neutral_description`），不允许重新拼接风险词；
    `train_stat` 来自 `feature_stats.csv`，缺失时输出 `weak_or_mixed, ...=NA`。
    """
    feature_categories = feature_categories or {}
    feature_stats = feature_stats or {}
    semantics_neutral = semantics_neutral or {}

    blocks: list[str] = []
    for name, value in row.items():
        if name == label_col:
            continue
        if not is_active_feature(value):
            continue

        feature = str(name)
        category = feature_categories.get(feature, "unknown")
        semantic_record = semantics_neutral.get(feature, {}) if isinstance(semantics_neutral, dict) else {}
        meaning = str(
            semantic_record.get("meaning")
            or semantic_record.get("description")
            or semantic_record.get("neutral_description")
            or ""
        ).strip() or "unknown"
        stats_record = feature_stats.get(feature, {}) if isinstance(feature_stats, dict) else {}
        train_stat = _format_train_stat(stats_record)

        blocks.append(
            f"- feature: {feature}\n"
            f"  category: {category}\n"
            f"  meaning: {meaning}\n"
            f"  train_stat: {train_stat}"
        )

    if not blocks:
        return "Active features: none"
    header = (
        "Active features (each block is structural; train_stat is training-pool "
        "context, not a direct label):"
    )
    return header + "\n" + "\n".join(blocks)


# 旧函数别名：保持外部脚本仍可以引用 `row_to_semantic_feature_text` 这个名字。
# 它指向 risky-old 渲染。
row_to_semantic_feature_text = row_to_semantic_risky_old_text


def row_to_feature_expr_text(
    row: pd.Series,
    feature_expr: str,
    feature_categories: dict[str, str] | None,
    feature_semantics: dict[str, dict[str, Any]] | None = None,
    feature_stats: dict[str, dict[str, Any]] | None = None,
    label_col: str = "class",
) -> str:
    """根据 feature_expr 统一分发到 raw / risky-old / neutral-fixed 三种渲染。

    `feature_semantics` 的语义会随 feature_expr 变化：
    - feature_expr == "semantic" / "semantic-risky-old"：是旧 risky-old JSON。
    - feature_expr == "semantic-neutral-fixed"：是 neutral-fixed JSON。
    """
    if feature_expr == "raw":
        return row_to_raw_feature_text(row, feature_categories, label_col=label_col)
    if feature_expr in (SEMANTIC_ALIAS, SEMANTIC_RISKY_OLD):
        if feature_semantics is None:
            raise ValueError(
                f"feature_semantics is required when feature_expr='{feature_expr}'"
            )
        return row_to_semantic_risky_old_text(
            row,
            feature_categories,
            feature_semantics,
            label_col=label_col,
        )
    if feature_expr == SEMANTIC_NEUTRAL_FIXED:
        if feature_semantics is None:
            raise ValueError(
                "feature_semantics is required when feature_expr='semantic-neutral-fixed'"
            )
        return row_to_semantic_neutral_text(
            row,
            feature_categories,
            feature_semantics,
            feature_stats=feature_stats,
            label_col=label_col,
        )
    raise ValueError(f"Unsupported feature expression: {feature_expr}")
