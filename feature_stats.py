from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import pandas as pd

from fewshot_utils import load_feature_categories, normalize_label


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute Drebin feature statistics from the training pool with test indices excluded."
    )
    parser.add_argument("--data-path", required=True)
    parser.add_argument("--test-metadata", required=True)
    parser.add_argument("--feature-category-path", default="data/dataset-features-categories.csv")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--label-col", default="class")
    return parser.parse_args()


def load_test_indices(path: str | Path) -> list[int]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    indices = data.get("test_source_indices")
    if not isinstance(indices, list):
        raise ValueError("test_metadata.json must contain list field test_source_indices")
    return [int(index) for index in indices]


def stat_direction(log_odds: float) -> str:
    if log_odds >= 0.5:
        return "leans_S"
    if log_odds <= -0.5:
        return "leans_B"
    return "weak_or_mixed"


def compute_stats(
    df: pd.DataFrame,
    categories: dict[str, str],
    *,
    label_col: str,
) -> list[dict[str, Any]]:
    labels = df[label_col].map(normalize_label)
    pool_b = df.loc[labels == "B"]
    pool_s = df.loc[labels == "S"]
    pool_b_total = int(len(pool_b))
    pool_s_total = int(len(pool_s))
    rows: list[dict[str, Any]] = []

    for feature in [column for column in df.columns if column != label_col]:
        values_b = pd.to_numeric(pool_b[feature].replace("?", pd.NA), errors="coerce").fillna(0)
        values_s = pd.to_numeric(pool_s[feature].replace("?", pd.NA), errors="coerce").fillna(0)
        support_b = int((values_b == 1).sum())
        support_s = int((values_s == 1).sum())
        p_b = (support_b + 1) / (pool_b_total + 2)
        p_s = (support_s + 1) / (pool_s_total + 2)
        log_odds = math.log(p_s / (1 - p_s)) - math.log(p_b / (1 - p_b))
        rows.append(
            {
                "feature": feature,
                "category": categories.get(feature, "Uncategorized"),
                "support_B": support_b,
                "support_S": support_s,
                "pool_B_total": pool_b_total,
                "pool_S_total": pool_s_total,
                "p_feature_given_B": p_b,
                "p_feature_given_S": p_s,
                "log_odds_S_vs_B": log_odds,
                "abs_log_odds": abs(log_odds),
                "stat_direction": stat_direction(log_odds),
            }
        )
    return rows


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    full_df = pd.read_csv(args.data_path, low_memory=False)
    test_indices = load_test_indices(args.test_metadata)
    test_index_set = set(test_indices)
    pool_df = full_df.loc[[idx for idx in full_df.index if int(idx) not in test_index_set]].copy()
    categories = load_feature_categories(args.feature_category_path)
    rows = compute_stats(pool_df, categories, label_col=args.label_col)

    csv_path = output_dir / "feature_stats.csv"
    json_path = output_dir / "feature_stats.json"
    metadata_path = output_dir / "feature_stats_metadata.json"
    pd.DataFrame(rows).to_csv(csv_path, index=False, encoding="utf-8-sig")
    json_path.write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
    metadata = {
        "data_path": str(args.data_path),
        "test_metadata": str(args.test_metadata),
        "feature_category_path": str(args.feature_category_path),
        "row_count_full": int(len(full_df)),
        "row_count_training_pool": int(len(pool_df)),
        "test_indices_excluded": test_indices,
        "test_indices_excluded_count": len(test_indices),
        "feature_count": len(rows),
        "label_col": args.label_col,
    }
    metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Saved: {csv_path}")
    print(f"Saved: {json_path}")
    print(f"Saved: {metadata_path}")


if __name__ == "__main__":
    main()
