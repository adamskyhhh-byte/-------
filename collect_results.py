from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd


METRIC_COLUMNS = [
    "method",
    "k",
    "test_size",
    "accuracy_strict",
    "accuracy",
    "precision",
    "recall_benign",
    "recall_malware",
    "f1",
    "macro_f1",
    "pred_S_ratio",
    "TP",
    "TN",
    "FP",
    "FN",
    "parse_ok_rate",
    "strict_parse_ok_rate",
    "invalid_evidence_vocab_rate",
    "invalid_evidence_inactive_rate",
    "behavior_rule_hit_rate",
    "num_ctx",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect final experiment metrics into Excel and figures.")
    parser.add_argument("--results-root", default="results/final")
    parser.add_argument("--feature-stats", default=None)
    parser.add_argument("--output-xlsx", default="results/final/results.xlsx")
    parser.add_argument("--figures-dir", default="results/final/figures")
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return data if isinstance(data, dict) else {}


def rows_from_baseline(results_root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in results_root.glob("baseline/**/baseline_metrics.csv"):
        df = pd.read_csv(path)
        metadata = load_json(path.with_name("run_metadata.json"))
        k = metadata.get("k")
        test_size = (metadata.get("test_size") or {}).get("total")
        for _, row in df.iterrows():
            rows.append(
                {
                    "method": str(row.get("model")),
                    "k": k,
                    "test_size": test_size,
                    "accuracy": row.get("accuracy"),
                    "precision": row.get("precision"),
                    "recall_benign": row.get("recall_benign"),
                    "recall_malware": row.get("recall_malware"),
                    "f1": row.get("f1"),
                    "macro_f1": row.get("macro_f1"),
                    "pred_S_ratio": row.get("pred_S_ratio"),
                    "TP": row.get("TP"),
                    "TN": row.get("TN"),
                    "FP": row.get("FP"),
                    "FN": row.get("FN"),
                    "parse_ok_rate": "N/A",
                    "strict_parse_ok_rate": "N/A",
                    "invalid_evidence_vocab_rate": "N/A",
                    "invalid_evidence_inactive_rate": "N/A",
                    "behavior_rule_hit_rate": "N/A",
                    "num_ctx": "N/A",
                }
            )
    return rows


def rows_from_metric_json(results_root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in results_root.glob("**/*_metrics.json"):
        if "baseline" in path.parts:
            continue
        metrics = load_json(path)
        method = metrics.get("experiment_id") or path.stem.removesuffix("_metrics")
        rows.append(
            {
                "method": method,
                "k": metrics.get("k"),
                "test_size": metrics.get("total") or metrics.get("test_size"),
                "accuracy_strict": metrics.get("accuracy_strict"),
                "accuracy": metrics.get("accuracy"),
                "precision": metrics.get("precision"),
                "recall_benign": metrics.get("recall_benign"),
                "recall_malware": metrics.get("recall_malware"),
                "f1": metrics.get("f1"),
                "macro_f1": metrics.get("macro_f1"),
                "pred_S_ratio": metrics.get("pred_S_ratio"),
                "TP": metrics.get("TP"),
                "TN": metrics.get("TN"),
                "FP": metrics.get("FP"),
                "FN": metrics.get("FN"),
                "parse_ok_rate": metrics.get("parse_ok_rate"),
                "strict_parse_ok_rate": metrics.get("strict_parse_ok_rate"),
                "invalid_evidence_vocab_rate": metrics.get("invalid_evidence_vocab_rate"),
                "invalid_evidence_inactive_rate": metrics.get("invalid_evidence_inactive_rate"),
                "behavior_rule_hit_rate": metrics.get("behavior_rule_hit_rate", "N/A"),
                "num_ctx": metrics.get("num_ctx"),
            }
        )
    return rows


def save_bar(df: pd.DataFrame, figures_dir: Path, column: str, filename: str) -> None:
    numeric = pd.to_numeric(df[column], errors="coerce")
    plot_df = df.loc[numeric.notna()].copy()
    if plot_df.empty:
        return
    plot_df[column] = pd.to_numeric(plot_df[column], errors="coerce")
    fig, ax = plt.subplots(figsize=(max(7, len(plot_df) * 0.8), 4.5))
    ax.bar(plot_df["method"].astype(str), plot_df[column])
    ax.set_ylabel(column)
    ax.tick_params(axis="x", rotation=35)
    fig.tight_layout()
    fig.savefig(figures_dir / filename, dpi=160)
    plt.close(fig)


def save_feature_log_odds(feature_stats: str | None, figures_dir: Path) -> None:
    if not feature_stats or not Path(feature_stats).exists():
        return
    df = pd.read_csv(feature_stats)
    if "abs_log_odds" not in df.columns:
        return
    plot_df = df.sort_values("abs_log_odds", ascending=False).head(20).copy()
    if plot_df.empty:
        return
    fig, ax = plt.subplots(figsize=(8, 7))
    ax.barh(plot_df["feature"].astype(str), plot_df["log_odds_S_vs_B"])
    ax.invert_yaxis()
    ax.set_xlabel("log_odds_S_vs_B")
    fig.tight_layout()
    fig.savefig(figures_dir / "feature_log_odds_top20.png", dpi=160)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    results_root = Path(args.results_root)
    output_xlsx = Path(args.output_xlsx)
    figures_dir = Path(args.figures_dir)
    output_xlsx.parent.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    rows = rows_from_baseline(results_root) + rows_from_metric_json(results_root)
    df = pd.DataFrame(rows)
    if df.empty:
        df = pd.DataFrame(columns=METRIC_COLUMNS)
    for column in METRIC_COLUMNS:
        if column not in df.columns:
            df[column] = "N/A"
    df = df[METRIC_COLUMNS]

    with pd.ExcelWriter(output_xlsx) as writer:
        df.to_excel(writer, sheet_name="metrics", index=False)

    save_bar(df, figures_dir, "macro_f1", "accuracy_macro_f1.png")
    save_bar(df, figures_dir, "pred_S_ratio", "pred_s_ratio.png")
    save_bar(df, figures_dir, "behavior_rule_hit_rate", "rag_doc_type_distribution.png")
    save_feature_log_odds(args.feature_stats, figures_dir)
    print(f"Saved: {output_xlsx}")
    print(f"Saved figures: {figures_dir}")


if __name__ == "__main__":
    main()
