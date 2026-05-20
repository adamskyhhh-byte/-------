from __future__ import annotations

import argparse
import csv
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

DEFAULT_FEATURE_CATEGORY_PATH = "data/dataset-features-categories.csv"
DEFAULT_FEATURE_SEMANTICS_PATH = "data/processed/paper_features/feature_semantics_neutral_stats.json"
DEFAULT_OUTPUT_DIR = "data/processed/rag_kb_fixed"

FORBIDDEN_FIELD_NAMES = {
    "class",
    "label",
    "true_label",
    "pred_label",
    "prediction",
    "answer",
    "result",
    "verdict",
    "decision",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the fixed Drebin RAG knowledge base.")
    parser.add_argument("--feature-category-path", default=DEFAULT_FEATURE_CATEGORY_PATH)
    parser.add_argument("--feature-semantics", default=DEFAULT_FEATURE_SEMANTICS_PATH)
    parser.add_argument("--feature-stats", default=None)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--skip-behavior-rules", action="store_true")
    return parser.parse_args()


def sanitize_doc_id(value: str, prefix: str = "FEAT") -> str:
    cleaned = re.sub(r"[^A-Za-z0-9]+", "_", value).strip("_").upper()
    return f"{prefix}_{(cleaned or 'UNKNOWN')[:90]}"


def load_feature_categories(path: str | Path) -> dict[str, str]:
    raw = pd.read_csv(path, header=None)
    categories: dict[str, str] = {}
    for feature, category in raw.iloc[:, :2].itertuples(index=False):
        if pd.isna(feature) or pd.isna(category):
            continue
        feature_text = str(feature).strip()
        if feature_text.lower() == "class":
            continue
        categories[feature_text] = str(category).strip()
    return categories


def load_json(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    if not path.exists():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return data


def load_feature_stats(path: str | Path | None) -> dict[str, dict[str, Any]]:
    if path is None:
        return {}
    path = Path(path)
    if not path.exists():
        return {}
    df = pd.read_csv(path)
    if "feature" not in df.columns:
        return {}
    return {str(row["feature"]): row.to_dict() for _, row in df.iterrows()}


def semantic_description(record: dict[str, Any], feature: str) -> str:
    description = str(
        record.get("description") or record.get("meaning") or record.get("neutral_description") or ""
    ).strip()
    return description or f"Android Drebin feature {feature}."


def format_stat(stats: dict[str, Any]) -> str:
    direction = str(stats.get("stat_direction", "weak_or_mixed"))
    p_b = stats.get("p_feature_given_B")
    p_s = stats.get("p_feature_given_S")
    if p_b is None or p_s is None:
        return f"Training-pool statistic: this feature {direction}. Use this only as dataset context, not as a direct label."
    return (
        f"Training-pool statistic: this feature {direction}; "
        f"P(feature|B)={float(p_b):.4f}, P(feature|S)={float(p_s):.4f}. "
        "Use this only as dataset context, not as a direct label."
    )


def related_label_from_stats(stats: dict[str, Any]) -> str:
    direction = str(stats.get("stat_direction", "weak_or_mixed"))
    if direction == "leans_B":
        return "B"
    if direction == "leans_S":
        return "S"
    return "context-dependent"


def build_feature_docs(
    categories: dict[str, str],
    semantics: dict[str, Any],
    stats_by_feature: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    docs: list[dict[str, Any]] = []
    for feature in sorted(categories):
        category = categories[feature]
        semantic_record = semantics.get(feature, {})
        stats = stats_by_feature.get(feature, {})
        description = semantic_description(semantic_record if isinstance(semantic_record, dict) else {}, feature)
        stat_text = format_stat(stats)
        related_label = related_label_from_stats(stats)
        doc_id = sanitize_doc_id(feature, prefix="FEAT")
        docs.append(
            {
                "doc_id": doc_id,
                "doc_type": "feature_card",
                "feature": feature,
                "category": category,
                "related_label": related_label,
                "retrieval_text": (
                    f"Android Drebin feature {feature}. Category: {category}. "
                    f"Meaning: {description}. {stat_text}"
                ),
                "prompt_text": (
                    f"[{doc_id}] Feature: {feature}. Category: {category}. "
                    f"Neutral meaning: {description}. {stat_text} "
                    f"Related label: {related_label} (soft prior only, not a direct answer)."
                ),
                "source": "dataset-features-categories + neutral semantics + training-pool stats",
                "leakage_risk": "no_sample_or_label",
                "p_feature_given_B": stats.get("p_feature_given_B"),
                "p_feature_given_S": stats.get("p_feature_given_S"),
                "stat_direction": stats.get("stat_direction"),
            }
        )
    return docs


def manual_behavior_rules() -> list[dict[str, Any]]:
    raw_rules = [
        (
            "SMS_ABUSE",
            "SMS abuse",
            "S",
            "SEND_SMS READ_SMS RECEIVE_SMS WRITE_SMS SmsManager INTERNET READ_PHONE_STATE",
            "SMS send/read/receive APIs together with network or phone-state access can indicate SMS abuse or premium-message behavior.",
        ),
        (
            "DEVICE_IDENTIFIER_COLLECTION",
            "Device identifiers",
            "S",
            "READ_PHONE_STATE getDeviceId getSubscriberId getSimSerialNumber getLine1Number GET_ACCOUNTS",
            "Telephony and account identifiers provide device or subscriber identity context when combined with network access.",
        ),
        (
            "BOOT_PERSISTENCE",
            "Persistence",
            "S",
            "BOOT_COMPLETED RECEIVE_BOOT_COMPLETED WAKE_LOCK SYSTEM_ALERT_WINDOW background service",
            "Boot receivers and wake locks can support automatic background execution after restart.",
        ),
        (
            "DYNAMIC_CODE_LOADING",
            "Dynamic loading",
            "S",
            "DexClassLoader ClassLoader System.loadLibrary Runtime.getRuntime reflection getMethods getField",
            "Dynamic loading and reflection can change runtime behavior beyond statically visible code paths.",
        ),
        (
            "NETWORK_EXFILTRATION_CONTEXT",
            "Network context",
            "S",
            "INTERNET ACCESS_NETWORK_STATE HttpGet DefaultHttpClient URLDecoder sensitive data",
            "Network APIs provide communication context and become more meaningful with identifier, contact, SMS, or location features.",
        ),
        (
            "CRYPTO_USAGE",
            "Cryptography",
            "context-dependent",
            "Ljavax.crypto.Cipher SecretKeySpec MessageDigest Base64",
            "Cryptographic APIs are common in benign and malware apps; interpret them with surrounding data access and network features.",
        ),
        (
            "STANDARD_IPC",
            "Android IPC",
            "context-dependent",
            "Binder IBinder bindService onServiceConnected ServiceConnection attachInterface",
            "Binder and service binding are standard Android IPC mechanisms and should be interpreted with the broader feature set.",
        ),
        (
            "FILE_STORAGE_ACCESS",
            "Storage",
            "context-dependent",
            "READ_EXTERNAL_STORAGE WRITE_EXTERNAL_STORAGE file path sdcard",
            "External storage access can be routine app behavior or part of data handling depending on surrounding permissions and APIs.",
        ),
        (
            "LOCATION_TRACKING",
            "Location",
            "S",
            "ACCESS_FINE_LOCATION ACCESS_COARSE_LOCATION GPS INTERNET background",
            "Location features with network and background execution provide context for tracking or reporting location data.",
        ),
        (
            "APP_ENTRY",
            "App entry",
            "B",
            "android.intent.action.MAIN android.intent.category.LAUNCHER",
            "MAIN and LAUNCHER describe ordinary app entry points and are usually background context rather than evidence by themselves.",
        ),
    ]
    docs: list[dict[str, Any]] = []
    for name, category, related_label, retrieval_text, explanation in raw_rules:
        doc_id = sanitize_doc_id(name, prefix="RULE")
        docs.append(
            {
                "doc_id": doc_id,
                "doc_type": "behavior_rule",
                "category": category,
                "related_label": related_label,
                "retrieval_text": (
                    f"Android behavior rule {name}. Related features: {retrieval_text}. "
                    f"Explanation: {explanation}"
                ),
                "prompt_text": (
                    f"[{doc_id}] Behavior rule: {category}. Related features: {retrieval_text}. "
                    f"Explanation: {explanation} Related label: {related_label} "
                    "(soft prior only, not a direct answer)."
                ),
                "source": "manual_android_security_knowledge",
                "leakage_risk": "no_sample_or_label",
            }
        )
    return docs


def assert_no_forbidden_fields(doc: dict[str, Any]) -> None:
    bad = {str(key).lower() for key in doc} & FORBIDDEN_FIELD_NAMES
    if bad:
        raise ValueError(f"{doc.get('doc_id')} has forbidden fields: {sorted(bad)}")


def validate_docs(docs: list[dict[str, Any]]) -> None:
    if not docs:
        raise ValueError("No RAG docs generated")
    seen: set[str] = set()
    required = {"doc_id", "doc_type", "retrieval_text", "prompt_text", "source", "leakage_risk"}
    for doc in docs:
        missing = required - set(doc)
        if missing:
            raise ValueError(f"{doc.get('doc_id')} missing fields: {sorted(missing)}")
        assert_no_forbidden_fields(doc)
        doc_id = str(doc["doc_id"])
        if doc_id in seen:
            raise ValueError(f"Duplicate doc_id: {doc_id}")
        seen.add(doc_id)


def write_jsonl(path: Path, docs: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for doc in docs:
            handle.write(json.dumps(doc, ensure_ascii=False) + "\n")


def write_rules_csv(path: Path, docs: list[dict[str, Any]]) -> None:
    rules = [doc for doc in docs if doc.get("doc_type") == "behavior_rule"]
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["doc_id", "category", "related_label", "explanation", "retrieval_text"],
        )
        writer.writeheader()
        for doc in rules:
            writer.writerow(
                {
                    "doc_id": doc.get("doc_id"),
                    "category": doc.get("category"),
                    "related_label": doc.get("related_label"),
                    "explanation": doc.get("prompt_text"),
                    "retrieval_text": doc.get("retrieval_text"),
                }
            )


def write_doc_index_map(path: Path, docs: list[dict[str, Any]]) -> None:
    mapping = [{"doc_index": index, "doc_id": doc.get("doc_id")} for index, doc in enumerate(docs)]
    path.write_text(json.dumps(mapping, indent=2, ensure_ascii=False), encoding="utf-8")


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    categories = load_feature_categories(args.feature_category_path)
    semantics = load_json(args.feature_semantics)
    stats_by_feature = load_feature_stats(args.feature_stats)
    docs = build_feature_docs(categories, semantics, stats_by_feature)
    if not args.skip_behavior_rules:
        docs.extend(manual_behavior_rules())
    validate_docs(docs)

    docs_path = output_dir / "kb_docs.jsonl"
    metadata_path = output_dir / "kb_metadata.json"
    write_jsonl(docs_path, docs)
    write_rules_csv(output_dir / "rules.csv", docs)
    write_doc_index_map(
        output_dir / "kb_feature_doc_index_map.json",
        [doc for doc in docs if doc.get("doc_type") == "feature_card"],
    )
    write_doc_index_map(
        output_dir / "kb_rule_doc_index_map.json",
        [doc for doc in docs if doc.get("doc_type") == "behavior_rule"],
    )

    metadata = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "stage": "fixed_text_kb",
        "feature_category_path": str(args.feature_category_path),
        "feature_semantics_path": str(args.feature_semantics),
        "feature_stats_path": str(args.feature_stats),
        "output_docs_path": str(docs_path),
        "doc_count": len(docs),
        "feature_card_count": sum(1 for doc in docs if doc.get("doc_type") == "feature_card"),
        "behavior_rule_count": sum(1 for doc in docs if doc.get("doc_type") == "behavior_rule"),
        "contains_embeddings": False,
        "contains_faiss_index": False,
        "contains_llm_outputs": False,
        "contains_test_samples": False,
        "contains_sample_labels": False,
    }
    metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Saved docs: {docs_path}")
    print(f"Saved metadata: {metadata_path}")
    print(f"Generated docs: {len(docs)}")


if __name__ == "__main__":
    main()
