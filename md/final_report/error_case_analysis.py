"""
错误案例分析脚本

从 results/final_50/feature_expr_llm/*/*.jsonl 和 results/final_50/rag_raw_llm/**/*.jsonl
中抽取错误样本（pred != true），按方法、K、错误方向（FP/FN）分组，
导出 markdown 表格便于报告引用。

用法：
    python md/final_report/error_case_analysis.py
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path


ROOT = Path("results/final_50")


def iter_jsonl(path: Path):
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            yield json.loads(line)
        except json.JSONDecodeError:
            continue


def list_runs() -> list[tuple[str, int, Path]]:
    """(method_tag, K, jsonl_path)"""
    runs: list[tuple[str, int, Path]] = []
    for k_dir in (ROOT / "feature_expr_llm" / "seed42_test100").glob("k*"):
        try:
            K = int(k_dir.name[1:])
        except ValueError:
            continue
        for expr in ("raw", "semantic-risky-old", "semantic-neutral-fixed"):
            path = k_dir / f"{expr}_full.jsonl"
            if path.exists():
                runs.append((expr, K, path))
    for k_dir in (ROOT / "rag_raw_llm" / "seed42_test100").glob("k*"):
        try:
            K = int(k_dir.name[1:])
        except ValueError:
            continue
        path = k_dir / "rag_raw_bucketed_f2_r3.jsonl"
        if path.exists():
            runs.append(("rag-bucketed", K, path))
    return runs


def summarize_errors(method: str, K: int, jsonl_path: Path) -> dict:
    errors: list[dict] = []
    total = 0
    for record in iter_jsonl(jsonl_path):
        total += 1
        true = record.get("true_label")
        pred = record.get("pred_label")
        if pred is None or pred not in ("B", "S"):
            errors.append({
                "idx": record.get("idx"),
                "true": true,
                "pred": pred,
                "kind": "parse_fail",
                "evidence": record.get("evidence"),
                "explanation": record.get("explanation"),
                "raw_preview": str(record.get("raw", ""))[:300],
            })
            continue
        if true != pred:
            kind = "FP" if (true == "B" and pred == "S") else "FN"
            errors.append({
                "idx": record.get("idx"),
                "true": true,
                "pred": pred,
                "kind": kind,
                "evidence": record.get("evidence"),
                "explanation": record.get("explanation"),
                "active_feature_count": record.get("active_feature_count"),
                "invalid_vocab": record.get("invalid_evidence_vocab"),
                "invalid_inactive": record.get("invalid_evidence_inactive"),
            })
    return {
        "method": method,
        "K": K,
        "total": total,
        "errors": errors,
        "n_FP": sum(1 for e in errors if e["kind"] == "FP"),
        "n_FN": sum(1 for e in errors if e["kind"] == "FN"),
        "n_parse_fail": sum(1 for e in errors if e["kind"] == "parse_fail"),
    }


def build_markdown(all_summaries: list[dict]) -> str:
    out: list[str] = ["# 错误案例分析 (50 samples × {K=1,3,5})\n"]
    out.append("## 总览\n")
    out.append("| Method | K | total | n_FP | n_FN | n_parse_fail |")
    out.append("|---|---:|---:|---:|---:|---:|")
    for s in all_summaries:
        out.append(
            f"| {s['method']} | {s['K']} | {s['total']} | {s['n_FP']} | {s['n_FN']} | {s['n_parse_fail']} |"
        )
    out.append("")

    # 选 3 个有代表性的错误案例：semantic-risky-old K=5 FP；semantic-neutral-fixed K=5 FN；rag-bucketed K=5 FP
    picks: list[tuple[str, int, str]] = [
        ("semantic-risky-old", 5, "FP"),
        ("semantic-neutral-fixed", 5, "FN"),
        ("rag-bucketed", 5, "FP"),
        ("raw", 5, "FN"),
    ]
    for method, K, kind in picks:
        match = next(
            (s for s in all_summaries if s["method"] == method and s["K"] == K),
            None,
        )
        if not match:
            continue
        sample = next((e for e in match["errors"] if e["kind"] == kind), None)
        if not sample:
            continue
        out.append(f"## 代表案例：method={method}, K={K}, kind={kind}")
        out.append(f"- idx: {sample.get('idx')}")
        out.append(f"- true: {sample.get('true')}, pred: {sample.get('pred')}")
        out.append(f"- active_feature_count: {sample.get('active_feature_count')}")
        ev = sample.get("evidence")
        if isinstance(ev, list):
            out.append("- evidence: " + ", ".join(map(str, ev[:10])))
        else:
            out.append(f"- evidence: {ev}")
        out.append(f"- explanation: {sample.get('explanation')}")
        if sample.get("invalid_vocab"):
            out.append(f"- invalid_evidence_vocab: {sample['invalid_vocab']}")
        if sample.get("invalid_inactive"):
            out.append(f"- invalid_evidence_inactive: {sample['invalid_inactive']}")
        out.append("")
    return "\n".join(out)


def main() -> None:
    runs = list_runs()
    if not runs:
        print("[error] no jsonl results found in", ROOT)
        return
    summaries = [summarize_errors(method, K, path) for method, K, path in runs]
    summaries.sort(key=lambda s: (s["method"], s["K"]))
    md_path = Path("md/final_report/error_cases.md")
    md_path.write_text(build_markdown(summaries), encoding="utf-8")
    print(f"Saved: {md_path}")


if __name__ == "__main__":
    main()
