"""
论文方法中的 raw/semantic full LLM few-shot 推理脚本。

这个文件实现实验方案里新增的对照 runner：它读取同一个 split-root 和同一个 k，
可以运行 raw full、semantic full 或 both。两种实验共用训练/测试样本、标签、
Ollama 调用、JSON 解析和指标计算逻辑，只替换样本特征在 prompt 中的表达方式。
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score

from fewshot_utils import LABEL_TO_ID, load_feature_categories, load_fewshot_split, normalize_label
from llm_only_fewshot import (
    add_prompt_and_evidence_diagnostics,
    build_record_from_raw,
    call_ollama,
)
from paper_prompt_utils import (
    load_feature_stats,
    load_feature_semantics,
    row_to_feature_expr_text,
)


SYSTEM_PROMPT = (
    "You are a strict JSON API for binary Android app classification. "
    "Return one JSON object only. Do not use Markdown. Do not explain outside JSON."
)

SUMMARY_COLUMNS = [
    "experiment_id",
    "feature_expr",
    "feature_subset",
    "k",
    "accuracy_strict",
    "precision",
    "recall_benign",
    "recall_malware",
    "f1",
    "macro_f1",
    "pred_S_ratio",
    "parse_ok_rate",
]


def parse_args() -> argparse.Namespace:
    """读取命令行参数，覆盖实验输入、特征表达、模型和输出目录。"""
    parser = argparse.ArgumentParser(description="Run raw/semantic full LLM few-shot experiments.")
    parser.add_argument("--split-root", required=True)
    parser.add_argument("--k", type=int, required=True)
    parser.add_argument(
        "--feature-expr",
        choices=["raw", "semantic", "both", "all"],
        default="semantic",
    )
    parser.add_argument("--feature-subset", choices=["full"], default="full")
    parser.add_argument("--feature-semantics", default=None)
    parser.add_argument("--feature-semantics-risky-old", default=None)
    parser.add_argument("--feature-semantics-neutral-fixed", default=None)
    parser.add_argument("--feature-stats", default=None)
    parser.add_argument("--feature-category-path", default="data/dataset-features-categories.csv")
    parser.add_argument("--output-root", default="results/feature_expr_llm")
    parser.add_argument("--model", default="gemma4:e4b")
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--request-timeout", type=float, default=60.0)
    parser.add_argument("--num-ctx", type=int, default=12288)
    parser.add_argument("--max-test-samples", type=int, default=None)
    parser.add_argument(
        "--reparse-jsonl",
        default=None,
        help="只重新解析已有 JSONL 的 raw 字段，不重新调用 LLM；常用于修复解析规则后重算指标。",
    )
    return parser.parse_args()


def selected_feature_exprs(feature_expr: str) -> list[str]:
    """把 CLI 中的 both 展开成实际要运行的两个实验。"""
    if feature_expr == "both":
        return ["raw", "semantic"]
    if feature_expr == "all":
        return ["raw", "semantic-risky-old", "semantic-neutral-fixed"]
    return [feature_expr]


def split_tag(split_root: Path) -> str:
    """把 fewshot_seed42_test100 转成 seed42_test100，作为输出目录名。"""
    name = split_root.name
    return name.removeprefix("fewshot_")


def build_output_dir(output_root: str | Path, split_root: Path, k: int) -> Path:
    """按文档约定构造 results/feature_expr_llm/seed42_test100/k5 目录。"""
    return Path(output_root) / split_tag(split_root) / f"k{k}"


def build_prompt(
    train_df,
    test_row,
    feature_expr: str,
    render_feature_expr: str,
    feature_subset: str,
    feature_categories: dict[str, str],
    feature_semantics: dict[str, dict[str, Any]] | None,
    feature_stats: dict[str, dict[str, Any]] | None,
) -> str:
    """把 few-shot 示例和当前测试样本拼成一次 LLM 分类请求。"""
    examples: list[str] = []
    for idx, row in train_df.iterrows():
        label = normalize_label(row["class"])
        features = row_to_feature_expr_text(
            row,
            feature_expr=render_feature_expr,
            feature_categories=feature_categories,
            feature_semantics=feature_semantics,
            feature_stats=feature_stats,
        )
        examples.append(f"Example {idx}\nLabel: {label}\n{features}")

    test_features = row_to_feature_expr_text(
        test_row,
        feature_expr=render_feature_expr,
        feature_categories=feature_categories,
        feature_semantics=feature_semantics,
        feature_stats=feature_stats,
    )

    return (
        "Classify the CURRENT TEST SAMPLE only.\n\n"
        "Task:\n"
        "- Binary classification for an Android app.\n"
        "- Use only the active Drebin features shown in this prompt.\n\n"
        "Allowed labels:\n"
        "- B means benign app.\n"
        "- S means malware app.\n\n"
        "Important mapping rule:\n"
        "- If your answer is benign app, output \"pred_label\":\"B\".\n"
        "- If your answer is malware app, output \"pred_label\":\"S\".\n\n"
        
        "Experiment settings:\n"
        f"- feature_expr: {feature_expr}\n"
        f"- feature_subset: {feature_subset}\n"
        "- k_semantics: per_class\n\n"
        "Few-shot examples:\n"
        + "\n\n".join(examples)
        + "\n\nCURRENT TEST SAMPLE FEATURES:\n"
        + test_features
        + "\n\nReturn EXACTLY ONE JSON object and nothing else.\n"
        "The first character must be { and the last character must be }.\n"
        "Do not use Markdown code fences.\n"
        "Use exactly these keys: pred_label, evidence, explanation, confidence.\n"
        "pred_label must be exactly one of: \"B\", \"S\".\n"
        "evidence must be a JSON array of active feature names from the current test sample.\n"
        "confidence must be a number from 0 to 1.\n\n"
        "Valid output example for benign:\n"
        "{\"pred_label\":\"B\",\"evidence\":[\"FEATURE_NAME\"],\"explanation\":\"short reason\",\"confidence\":0.50}\n"
        "Valid output example for malware:\n"
        "{\"pred_label\":\"S\",\"evidence\":[\"FEATURE_NAME\"],\"explanation\":\"short reason\",\"confidence\":0.80}"
    )


def metrics_from_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    """根据 JSONL 记录计算文档要求的指标，并补充预测分布统计。"""
    total = len(records)
    y_true = [LABEL_TO_ID[record["true_label"]] for record in records]
    strict_pred: list[int] = []
    valid_true: list[int] = []
    valid_pred: list[int] = []
    parse_source_counts: dict[str, int] = {}

    for record in records:
        true_id = LABEL_TO_ID[record["true_label"]]
        pred_label = record.get("pred_label")
        parse_source = str(record.get("parse_source", "failed"))
        parse_source_counts[parse_source] = parse_source_counts.get(parse_source, 0) + 1

        if pred_label in LABEL_TO_ID:
            pred_id = LABEL_TO_ID[pred_label]
            strict_pred.append(pred_id)
            valid_true.append(true_id)
            valid_pred.append(pred_id)
        else:
            # 解析失败计为 strict 错误，避免忽略失败样本造成指标虚高。
            strict_pred.append(1 - true_id)

    cm = confusion_matrix(y_true, strict_pred, labels=[0, 1]) if total else [[0, 0], [0, 0]]
    tn, fp, fn, tp = [int(value) for value in cm.ravel()]
    parse_ok = [record for record in records if record.get("parse_ok")]
    strict_parse_ok = [record for record in records if record.get("strict_parse_ok")]
    pred_b_count = sum(1 for record in records if record.get("pred_label") == "B")
    pred_s_count = sum(1 for record in records if record.get("pred_label") == "S")
    invalid_vocab_total = sum(int(record.get("invalid_evidence_vocab_count", 0)) for record in records)
    invalid_inactive_total = sum(
        int(record.get("invalid_evidence_inactive_count", 0)) for record in records
    )
    evidence_total = sum(len(record.get("evidence", []) or []) for record in records)

    return {
        "total": total,
        "accuracy_strict": float(accuracy_score(y_true, strict_pred)) if total else 0.0,
        "accuracy_valid_only": float(accuracy_score(valid_true, valid_pred)) if valid_true else None,
        "precision": float(precision_score(y_true, strict_pred, pos_label=1, zero_division=0)) if total else 0.0,
        "recall_benign": float(recall_score(y_true, strict_pred, pos_label=0, zero_division=0)) if total else 0.0,
        "recall_malware": float(recall_score(y_true, strict_pred, pos_label=1, zero_division=0)) if total else 0.0,
        "f1": float(f1_score(y_true, strict_pred, pos_label=1, zero_division=0)) if total else 0.0,
        "macro_f1": float(f1_score(y_true, strict_pred, average="macro", zero_division=0)) if total else 0.0,
        "pred_B_count": int(pred_b_count),
        "pred_S_count": int(pred_s_count),
        "pred_S_ratio": float(pred_s_count / total) if total else 0.0,
        "parse_ok_rate": float(len(parse_ok) / total) if total else 0.0,
        "strict_parse_ok_rate": float(len(strict_parse_ok) / total) if total else 0.0,
        "invalid_evidence_vocab_total": invalid_vocab_total,
        "invalid_evidence_vocab_rate": (
            float(invalid_vocab_total / evidence_total) if evidence_total else 0.0
        ),
        "invalid_evidence_inactive_total": invalid_inactive_total,
        "invalid_evidence_inactive_rate": (
            float(invalid_inactive_total / evidence_total) if evidence_total else 0.0
        ),
        "TP": tp,
        "TN": tn,
        "FP": fp,
        "FN": fn,
        "parse_ok": len(parse_ok),
        "strict_parse_ok": len(strict_parse_ok),
        "parse_source_counts": parse_source_counts,
    }


def enrich_record(
    record: dict[str, Any],
    experiment_id: str,
    feature_expr: str,
    feature_subset: str,
    k: int,
    prompt_path: Path | None,
) -> dict[str, Any]:
    """把通用解析记录补充成文档约定的 JSONL 单条结构。"""
    pred_label = record.get("pred_label")
    true_label = record.get("true_label")
    record.update(
        {
            "experiment_id": experiment_id,
            "feature_expr": feature_expr,
            "feature_subset": feature_subset,
            "k": k,
            "k_semantics": "per_class",
            "fewshot_total": 2 * k,
            "correct": bool(pred_label == true_label) if pred_label in LABEL_TO_ID else False,
        }
    )
    if prompt_path is not None:
        record["prompt_path"] = str(prompt_path)
    return record


def run_reparse(
    args: argparse.Namespace,
    output_dir: Path,
    feature_expr: str,
    feature_subset: str,
) -> dict[str, Any]:
    """不调用 LLM，只重解析旧 JSONL 的 raw 字段并重新生成指标。"""
    experiment_id = f"{feature_expr}_{feature_subset}_k{args.k}"
    jsonl_path = output_dir / f"{feature_expr}_{feature_subset}.jsonl"
    records: list[dict[str, Any]] = []

    with open(args.reparse_jsonl, encoding="utf-8") as source, jsonl_path.open("w", encoding="utf-8") as target:
        for line_no, line in enumerate(source):
            old_record = json.loads(line)
            true_label = normalize_label(old_record["true_label"])
            record = build_record_from_raw(
                idx=int(old_record.get("idx", line_no)),
                true_label=true_label,
                raw=str(old_record.get("raw", "")),
            )
            prompt_path_value = old_record.get("prompt_path")
            prompt_path = Path(prompt_path_value) if prompt_path_value else None
            record = enrich_record(record, experiment_id, feature_expr, feature_subset, args.k, prompt_path)
            records.append(record)
            target.write(json.dumps(record, ensure_ascii=False) + "\n")

    return save_metrics(output_dir, records, args, feature_expr, feature_subset)


def run_experiment(
    args: argparse.Namespace,
    output_dir: Path,
    feature_expr: str,
    render_feature_expr: str,
    feature_subset: str,
    feature_categories: dict[str, str],
    feature_semantics: dict[str, dict[str, Any]] | None,
    feature_stats: dict[str, dict[str, Any]] | None,
) -> dict[str, Any]:
    """运行单个 raw_full 或 semantic_full 实验。"""
    train_df, test_df = load_fewshot_split(args.split_root, args.k)
    if args.max_test_samples is not None:
        test_df = test_df.head(args.max_test_samples).copy()
    drebin_vocab = {str(column) for column in test_df.columns if str(column) != "class"}

    experiment_id = f"{feature_expr}_{feature_subset}_k{args.k}"
    jsonl_path = output_dir / f"{feature_expr}_{feature_subset}.jsonl"
    prompt_dir = output_dir / "prompts" / f"{feature_expr}_{feature_subset}"
    prompt_dir.mkdir(parents=True, exist_ok=True)

    records: list[dict[str, Any]] = []
    with jsonl_path.open("w", encoding="utf-8") as handle:
        for out_idx, (row_idx, row) in enumerate(test_df.iterrows(), start=1):
            true_label = normalize_label(row["class"])
            prompt = build_prompt(
                train_df,
                row,
                feature_expr,
                render_feature_expr,
                feature_subset,
                feature_categories,
                feature_semantics,
                feature_stats,
            )
            prompt_path = prompt_dir / f"{int(row_idx)}.txt"
            prompt_path.write_text(prompt, encoding="utf-8")

            raw = ""
            try:
                raw = call_ollama(
                    args.model,
                    prompt,
                    args.temperature,
                    args.request_timeout,
                    args.num_ctx,
                )
                record = build_record_from_raw(int(row_idx), true_label, raw)
            except Exception as exc:
                record = {
                    "idx": int(row_idx),
                    "true_label": true_label,
                    "pred_label": None,
                    "parse_ok": False,
                    "strict_parse_ok": False,
                    "error": str(exc),
                    "raw": raw or str(exc),
                }

            record = add_prompt_and_evidence_diagnostics(record, prompt, row, drebin_vocab)
            record = enrich_record(record, experiment_id, feature_expr, feature_subset, args.k, prompt_path)
            records.append(record)
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            print(f"[{experiment_id} {out_idx}/{len(test_df)}] true={true_label} pred={record['pred_label']}")

    return save_metrics(output_dir, records, args, feature_expr, feature_subset)


def save_metrics(
    output_dir: Path,
    records: list[dict[str, Any]],
    args: argparse.Namespace,
    feature_expr: str,
    feature_subset: str,
) -> dict[str, Any]:
    """保存单个实验的 metrics JSON，并返回指标字典供汇总 CSV 使用。"""
    experiment_id = f"{feature_expr}_{feature_subset}_k{args.k}"
    metrics = metrics_from_records(records)
    metrics.update(
        {
            "experiment_id": experiment_id,
            "feature_expr": feature_expr,
            "feature_subset": feature_subset,
            "k": args.k,
            "k_semantics": "per_class",
            "fewshot_total": 2 * args.k,
            "split_root": str(args.split_root),
            "model": args.model,
            "temperature": args.temperature,
            "request_timeout": args.request_timeout,
            "num_ctx": args.num_ctx,
            "max_test_samples": args.max_test_samples,
        }
    )
    metrics_path = output_dir / f"{feature_expr}_{feature_subset}_metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Saved: {metrics_path}")
    return metrics


def save_summary_csv(output_dir: Path, metrics_rows: list[dict[str, Any]]) -> None:
    """保存 raw/semantic 可横向比较的汇总 CSV。"""
    summary_path = output_dir / "feature_expr_metrics.csv"
    with summary_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=SUMMARY_COLUMNS)
        writer.writeheader()
        for row in metrics_rows:
            writer.writerow({column: row.get(column) for column in SUMMARY_COLUMNS})
    print(f"Saved: {summary_path}")


def main() -> None:
    args = parse_args()
    split_root = Path(args.split_root)
    output_dir = build_output_dir(args.output_root, split_root, args.k)
    output_dir.mkdir(parents=True, exist_ok=True)

    feature_categories = load_feature_categories(args.feature_category_path)
    feature_stats = load_feature_stats(args.feature_stats)
    exprs = selected_feature_exprs(args.feature_expr)
    semantics_by_expr: dict[str, dict[str, dict[str, Any]] | None] = {"raw": None}
    if "semantic" in exprs:
        if not args.feature_semantics:
            raise ValueError("--feature-semantics is required for semantic experiments")
        semantics_by_expr["semantic"] = load_feature_semantics(args.feature_semantics)
    if "semantic-risky-old" in exprs:
        if not args.feature_semantics_risky_old:
            raise ValueError("--feature-semantics-risky-old is required")
        semantics_by_expr["semantic-risky-old"] = load_feature_semantics(
            args.feature_semantics_risky_old
        )
    if "semantic-neutral-fixed" in exprs:
        if not args.feature_semantics_neutral_fixed:
            raise ValueError("--feature-semantics-neutral-fixed is required")
        semantics_by_expr["semantic-neutral-fixed"] = load_feature_semantics(
            args.feature_semantics_neutral_fixed
        )

    metrics_rows: list[dict[str, Any]] = []
    for feature_expr in exprs:
        if args.reparse_jsonl:
            metrics = run_reparse(args, output_dir, feature_expr, args.feature_subset)
        else:
            render_expr = "raw" if feature_expr == "raw" else "semantic"
            metrics = run_experiment(
                args,
                output_dir,
                feature_expr,
                render_expr,
                args.feature_subset,
                feature_categories,
                semantics_by_expr.get(feature_expr),
                feature_stats,
            )
        metrics_rows.append(metrics)

    save_summary_csv(output_dir, metrics_rows)


if __name__ == "__main__":
    main()
