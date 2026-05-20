from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from fewshot_utils import load_feature_categories, load_fewshot_split, normalize_label, row_to_feature_text
from llm_only_fewshot import (
    add_prompt_and_evidence_diagnostics,
    build_record_from_raw,
    call_ollama,
    metrics_from_records,
)
from rag_retriever import RagRetriever, RetrievalHit


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run raw Drebin few-shot classification with RAG.")
    parser.add_argument("--split-root", default="data/processed/fewshot_seed42_test100")
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--retrieval-mode", choices=["global", "bucketed"], default="global")
    parser.add_argument("--feature-top-k", type=int, default=2)
    parser.add_argument("--rule-top-k", type=int, default=3)
    parser.add_argument("--kb-dir", default="data/processed/rag_kb_fixed")
    parser.add_argument("--feature-category-path", default="data/dataset-features-categories.csv")
    parser.add_argument("--embedding-model", default="BAAI/bge-small-en-v1.5")
    parser.add_argument("--model", default="gemma4:e4b")
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--request-timeout", type=float, default=60.0)
    parser.add_argument("--num-ctx", type=int, default=12288)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--output-root", default="results/rag_raw_llm")
    parser.add_argument("--max-test-samples", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true", help="Save prompts and retrieval logs only.")
    parser.add_argument("--rebuild-index", action="store_true")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--no-save-prompts", action="store_true")
    return parser.parse_args()


def active_feature_names(row: Any, label_col: str = "class") -> list[str]:
    names: list[str] = []
    for name, value in row.items():
        if name == label_col:
            continue
        try:
            is_active = float(value) == 1.0
        except (TypeError, ValueError):
            is_active = False
        if is_active:
            names.append(str(name))
    return sorted(names)


# 高风险组：每个组列出 Drebin 215 中真实存在的特征名。
# 用于 build_rule_query_text，把"哪些高危组激活了"显式写进 rule query，
# 引导 bucketed 检索到行为规则桶。
RISK_GROUPS: dict[str, list[str]] = {
    "sms": [
        "SEND_SMS",
        "READ_SMS",
        "RECEIVE_SMS",
        "WRITE_SMS",
        "android.telephony.SmsManager",
        "android.telephony.gsm.SmsManager",
        "sendDataMessage",
        "sendMultipartTextMessage",
    ],
    "telephony_identifier": [
        "READ_PHONE_STATE",
        "TelephonyManager.getDeviceId",
        "TelephonyManager.getSubscriberId",
        "TelephonyManager.getSimSerialNumber",
        "TelephonyManager.getLine1Number",
    ],
    "root_system_command": [
        "/system/bin",
        "/system/app",
        "mount",
        "remount",
        "chmod",
        "chown",
        "Runtime.exec",
    ],
    "package_install": [
        "INSTALL_PACKAGES",
        "DELETE_PACKAGES",
        "PackageInstaller",
    ],
    "dynamic_loading": [
        "DexClassLoader",
        "PathClassLoader",
        "ClassLoader",
        "System.loadLibrary",
        "Runtime.load",
        "Runtime.loadLibrary",
    ],
    "network": [
        "INTERNET",
        "ACCESS_NETWORK_STATE",
        "HttpGet.init",
        "HttpPost.init",
        "HttpUriRequest",
    ],
    "privacy_sensor": [
        "RECORD_AUDIO",
        "CAMERA",
        "ACCESS_FINE_LOCATION",
        "ACCESS_COARSE_LOCATION",
        "READ_CONTACTS",
        "READ_CALL_LOG",
    ],
    "persistence_overlay": [
        "RECEIVE_BOOT_COMPLETED",
        "android.intent.action.BOOT_COMPLETED",
        "WAKE_LOCK",
        "SYSTEM_ALERT_WINDOW",
    ],
}


def active_groups(active_set: set[str]) -> dict[str, list[str]]:
    """返回 {group_name: [active feature names]}（保留命中顺序）。"""
    return {
        group: [name for name in members if name in active_set]
        for group, members in RISK_GROUPS.items()
    }


def build_query_text(test_row: Any, feature_categories: dict[str, str]) -> str:
    """feature query：列 active feature 名，用于检索 feature_card 桶。"""
    feature_text = row_to_feature_text(test_row, feature_categories=feature_categories)
    active_names = active_feature_names(test_row)
    compact_names = ", ".join(active_names[:80])
    if len(active_names) > 80:
        compact_names += f", ... ({len(active_names)} active features total)"
    return (
        "Android app active Drebin features for feature card retrieval.\n"
        f"{feature_text}\n"
        f"Raw active feature names: {compact_names}\n"
        "Retrieve feature meanings (feature_card)."
    )


def build_rule_query_text(test_row: Any) -> str:
    """rule query：列 8 个高风险组的 active count + 命中清单，用于检索行为规则。"""
    active_set = set(active_feature_names(test_row))
    groups = active_groups(active_set)
    lines = ["Android malware behavior rule query.", "Active groups:"]
    for group, hits in groups.items():
        count = len(hits)
        if hits:
            lines.append(f"- {group}: {count} active ({', '.join(hits)})")
        else:
            lines.append(f"- {group}: 0 active")
    nonzero = [name for name, hits in groups.items() if hits]
    empty = [name for name, hits in groups.items() if not hits]
    summary_parts = []
    if nonzero:
        summary_parts.append("present: " + ", ".join(nonzero))
    if empty:
        summary_parts.append("absent: " + ", ".join(empty))
    if summary_parts:
        lines.append("Important combinations: " + "; ".join(summary_parts) + ".")
    lines.append(
        "Retrieve behavior rules including benign and context-dependent rules."
    )
    return "\n".join(lines)


def format_fewshot_examples(train_df: Any, feature_categories: dict[str, str]) -> str:
    blocks: list[str] = []
    for example_no, (_, row) in enumerate(train_df.iterrows(), start=1):
        label = normalize_label(row["class"])
        features = row_to_feature_text(row, feature_categories=feature_categories)
        blocks.append(f"Example {example_no}\nLabel: {label}\n{features}")
    return "\n\n".join(blocks)


def format_retrieved_docs(hits: list[RetrievalHit], max_chars_per_doc: int = 700) -> str:
    lines: list[str] = []
    for hit in hits:
        prompt_text = str(hit.doc.get("prompt_text", "")).strip()
        if len(prompt_text) > max_chars_per_doc:
            prompt_text = prompt_text[: max_chars_per_doc - 3].rstrip() + "..."
        lines.append(f"[{hit.rank}] score={hit.score:.4f} {prompt_text}")
    return "\n".join(lines) if lines else "No retrieved knowledge."


def split_hits_by_doc_type(hits: list[RetrievalHit]) -> tuple[list[RetrievalHit], list[RetrievalHit]]:
    """把检索结果拆成 feature_card / behavior_rule 两块，rank 各自重排。"""
    feature_hits: list[RetrievalHit] = []
    rule_hits: list[RetrievalHit] = []
    for hit in hits:
        doc_type = hit.doc.get("doc_type")
        if doc_type == "feature_card":
            feature_hits.append(hit)
        elif doc_type == "behavior_rule":
            rule_hits.append(hit)
        else:
            feature_hits.append(hit)
    # 重新连续编号，便于 prompt 读取
    feature_hits = [
        RetrievalHit(rank=i + 1, doc_index=h.doc_index, score=h.score, doc=h.doc)
        for i, h in enumerate(feature_hits)
    ]
    rule_hits = [
        RetrievalHit(rank=i + 1, doc_index=h.doc_index, score=h.score, doc=h.doc)
        for i, h in enumerate(rule_hits)
    ]
    return feature_hits, rule_hits


def build_rag_prompt(
    *,
    train_df: Any,
    test_row: Any,
    feature_categories: dict[str, str],
    hits: list[RetrievalHit],
) -> str:
    fewshot_text = format_fewshot_examples(train_df, feature_categories)
    test_features = row_to_feature_text(test_row, feature_categories=feature_categories)
    feature_hits, rule_hits = split_hits_by_doc_type(hits)
    retrieved_feature_block = format_retrieved_docs(feature_hits)
    retrieved_rule_block = format_retrieved_docs(rule_hits)
    return (
        "Classify the CURRENT TEST SAMPLE only.\n\n"
        "Allowed labels:\n"
        "- B means benign app.\n"
        "- S means malware app.\n\n"
        "Important mapping rule:\n"
        "- If your answer is benign app, output \"pred_label\":\"B\".\n"
        "- If your answer is malware app, output \"pred_label\":\"S\".\n\n"
        "General reasoning hint:\n"
        "- Consider the combination of active features rather than any single feature.\n"
        "- Use the current test sample features as primary evidence.\n"
        "- Retrieved knowledge is background only.\n"
        "- Do not cite or rely on a retrieved rule unless at least one related active feature appears below.\n\n"
        "Knowledge usage policy:\n"
        "- Retrieved behavior rules may include a related_label field. This is a soft prior from the rule base, "
        "not a hard rule.\n"
        "- Do not copy related_label as the final answer unless the current active features and few-shot examples "
        "support it.\n"
        "- When a retrieved rule matches current active features, cite its rule id in explanation. "
        "If no retrieved rule matches, say so briefly in explanation.\n\n"
        "Few-shot examples:\n"
        f"{fewshot_text}\n\n"
        "CURRENT TEST SAMPLE FEATURES:\n"
        f"{test_features}\n\n"
        "RETRIEVED FEATURE KNOWLEDGE:\n"
        f"{retrieved_feature_block}\n\n"
        "RETRIEVED BEHAVIOR RULES:\n"
        f"{retrieved_rule_block}\n\n"
        "Return EXACTLY ONE JSON object and nothing else.\n"
        "The first character must be { and the last character must be }.\n"
        "Do not use Markdown code fences.\n"
        "Do not use keys such as label, prediction, classification, malicious, or analysis.\n"
        "Use exactly these keys: pred_label, evidence, explanation, confidence.\n"
        "pred_label must be exactly one of: \"B\", \"S\".\n"
        "evidence must be a JSON array of active feature names from the current test sample.\n"
        "In explanation, you may mention retrieved doc ids like [RULE_SMS_ABUSE] only when they match active features.\n"
        "confidence must be a number from 0 to 1.\n\n"
        "Valid output example for benign:\n"
        "{\"pred_label\":\"B\",\"evidence\":[\"FEATURE_NAME\"],\"explanation\":\"short reason\",\"confidence\":0.50}\n"
        "Valid output example for malware:\n"
        "{\"pred_label\":\"S\",\"evidence\":[\"FEATURE_NAME\"],\"explanation\":\"short reason\",\"confidence\":0.80}"
    )


def output_dir_for(args: argparse.Namespace) -> Path:
    split_name = Path(args.split_root).name
    if split_name.startswith("fewshot_"):
        split_name = split_name[len("fewshot_") :]
    return Path(args.output_root) / split_name / f"k{args.k}"


def write_jsonl_line(handle: Any, obj: dict[str, Any]) -> None:
    handle.write(json.dumps(obj, ensure_ascii=False) + "\n")
    handle.flush()


def behavior_rule_hit_rate(retrieval_logs: list[dict[str, Any]]) -> float:
    if not retrieval_logs:
        return 0.0
    return sum(
        1
        for sample in retrieval_logs
        if any(hit.get("doc_type") == "behavior_rule" for hit in sample.get("hits", []))
    ) / len(retrieval_logs)


def avg_behavior_rule_hits(retrieval_logs: list[dict[str, Any]]) -> float:
    if not retrieval_logs:
        return 0.0
    return sum(
        sum(1 for hit in sample.get("hits", []) if hit.get("doc_type") == "behavior_rule")
        for sample in retrieval_logs
    ) / len(retrieval_logs)


def main() -> None:
    args = parse_args()
    train_df, test_df = load_fewshot_split(args.split_root, args.k)
    if args.max_test_samples is not None:
        test_df = test_df.head(args.max_test_samples).copy()

    feature_categories = load_feature_categories(args.feature_category_path)
    drebin_vocab = {str(column) for column in test_df.columns if str(column) != "class"}
    retriever = RagRetriever(
        kb_dir=args.kb_dir,
        model_name=args.embedding_model,
        batch_size=args.batch_size,
        rebuild=args.rebuild_index,
        local_files_only=args.local_files_only,
    )

    out_dir = output_dir_for(args)
    out_dir.mkdir(parents=True, exist_ok=True)
    prompts_dir = out_dir / "prompts"
    if not args.no_save_prompts:
        prompts_dir.mkdir(parents=True, exist_ok=True)

    result_tag = (
        f"rag_raw_bucketed_f{args.feature_top_k}_r{args.rule_top_k}"
        if args.retrieval_mode == "bucketed"
        else f"rag_raw_top{args.top_k}"
    )
    jsonl_path = out_dir / f"{result_tag}.jsonl"
    metrics_path = out_dir / f"{result_tag}_metrics.json"
    retrieval_log_path = out_dir / "retrieval_logs.jsonl"
    run_config_path = out_dir / "run_config.json"

    run_config = {
        "split_root": str(args.split_root),
        "k": args.k,
        "top_k": args.top_k,
        "retrieval_mode": args.retrieval_mode,
        "feature_top_k": args.feature_top_k,
        "rule_top_k": args.rule_top_k,
        "kb_dir": str(args.kb_dir),
        "embedding_model": args.embedding_model,
        "llm_model": args.model,
        "temperature": args.temperature,
        "request_timeout": args.request_timeout,
        "num_ctx": args.num_ctx,
        "max_test_samples": args.max_test_samples,
        "dry_run": args.dry_run,
        "local_files_only": args.local_files_only,
        "output_dir": str(out_dir),
    }
    run_config_path.write_text(json.dumps(run_config, indent=2, ensure_ascii=False), encoding="utf-8")

    records: list[dict[str, Any]] = []
    retrieval_logs: list[dict[str, Any]] = []
    jsonl_handle = None if args.dry_run else jsonl_path.open("w", encoding="utf-8")
    with retrieval_log_path.open("w", encoding="utf-8") as retrieval_handle:
        try:
            for out_no, (row_idx, row) in enumerate(test_df.iterrows(), start=1):
                true_label = normalize_label(row["class"])
                query_text = build_query_text(row, feature_categories)
                rule_query_text = build_rule_query_text(row)
                if args.retrieval_mode == "bucketed":
                    hits = retriever.retrieve_bucketed(
                        feature_query=query_text,
                        rule_query=rule_query_text,
                        feature_top_k=args.feature_top_k,
                        rule_top_k=args.rule_top_k,
                    )
                else:
                    hits = retriever.retrieve(query_text, top_k=args.top_k)

                prompt = build_rag_prompt(
                    train_df=train_df,
                    test_row=row,
                    feature_categories=feature_categories,
                    hits=hits,
                )
                prompt_path = None
                if not args.no_save_prompts:
                    prompt_path = prompts_dir / f"{int(row_idx)}.txt"
                    prompt_path.write_text(prompt, encoding="utf-8")

                retrieved_doc_ids = [str(hit.doc.get("doc_id")) for hit in hits]
                retrieved_scores = [hit.score for hit in hits]
                retrieval_log = {
                    "test_index": int(row_idx),
                    "idx": int(row_idx),
                    "query_text": query_text,
                    "rule_query_text": rule_query_text,
                    "active_feature_count": len(active_feature_names(row)),
                    "top_k": args.top_k,
                    "retrieval_mode": args.retrieval_mode,
                    "hits": [hit.to_log_dict() for hit in hits],
                    "retrieved": [hit.to_log_dict() for hit in hits],
                    "prompt_path": str(prompt_path) if prompt_path else None,
                }
                retrieval_logs.append(retrieval_log)
                write_jsonl_line(retrieval_handle, retrieval_log)

                if args.dry_run:
                    print(
                        f"[dry-run {out_no}/{len(test_df)}] "
                        f"idx={int(row_idx)} retrieved={','.join(retrieved_doc_ids)}"
                    )
                    continue

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

                record.update(
                    {
                        "retrieved_doc_ids": retrieved_doc_ids,
                        "retrieval_scores": retrieved_scores,
                        "prompt_path": str(prompt_path) if prompt_path else None,
                    }
                )
                record = add_prompt_and_evidence_diagnostics(record, prompt, row, drebin_vocab)
                records.append(record)

                assert jsonl_handle is not None
                write_jsonl_line(jsonl_handle, record)
                print(
                    f"[{out_no}/{len(test_df)}] true={true_label} pred={record.get('pred_label')} "
                    f"retrieved={','.join(retrieved_doc_ids)}"
                )
        finally:
            if jsonl_handle is not None:
                jsonl_handle.close()

    if args.dry_run:
        print(f"Saved prompts: {prompts_dir}")
        print(f"Saved retrieval logs: {retrieval_log_path}")
        print(f"Saved run config: {run_config_path}")
        return

    metrics = metrics_from_records(records)
    metrics.update(run_config)
    metrics["behavior_rule_hit_rate"] = behavior_rule_hit_rate(retrieval_logs)
    metrics["avg_behavior_rule_hits_per_sample"] = avg_behavior_rule_hits(retrieval_logs)
    metrics_path.write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Saved predictions: {jsonl_path}")
    print(f"Saved metrics: {metrics_path}")
    print(f"Saved retrieval logs: {retrieval_log_path}")


if __name__ == "__main__":
    main()
