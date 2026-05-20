"""Drebin LLM + RAG 实验的修复后知识库构建器。

修复版职责：
1. 写 `kb_docs.jsonl`：215 条 feature_card + ≥22 条 behavior_rule（含 S/B/Context-dependent）。
2. 写 `rules.csv`：PPT 要求的规则表。
3. 构建 3 个 FAISS 索引并落盘：
   - `kb.index`        全量索引（225+ 条）
   - `kb_feature.index` 仅 feature_card 子集
   - `kb_rule.index`    仅 behavior_rule 子集
4. 写 `kb_feature_doc_index_map.json` / `kb_rule_doc_index_map.json`：
   子索引行号 → 全量 `kb_docs.jsonl` 行号。
5. 写 `kb_metadata.json`：包含 `forbidden_field_names` / `sub_indices` 等元数据。

KB 内部禁止字段名严格按集合相等比较。`related_label` 不在禁字段内。
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from rag_retriever import (
    DEFAULT_MODEL_NAME,
    build_index,
    docs_fingerprint,
    require_faiss,
)

DEFAULT_FEATURE_CATEGORY_PATH = "data/dataset-features-categories.csv"
DEFAULT_FEATURE_SEMANTICS_PATH = (
    "data/processed/paper_features/feature_semantics_neutral_stats.json"
)
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
    parser.add_argument(
        "--no-build-faiss",
        action="store_true",
        help="Skip FAISS index construction; only write docs and metadata.",
    )
    parser.add_argument("--embedding-model", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--show-progress", action="store_true")
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
        record.get("meaning")
        or record.get("description")
        or record.get("neutral_description")
        or ""
    ).strip()
    return description or f"Android Drebin feature {feature}."


def format_stat(stats: dict[str, Any]) -> str:
    direction = str(stats.get("stat_direction", "weak_or_mixed"))
    p_b = stats.get("p_feature_given_B")
    p_s = stats.get("p_feature_given_S")
    if p_b is None or p_s is None:
        return (
            f"Training-pool statistic: this feature {direction}. "
            "Use this only as dataset context, not as a direct label."
        )
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
        description = semantic_description(
            semantic_record if isinstance(semantic_record, dict) else {}, feature
        )
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


# 共 24 条行为规则：S=10 / B=6 / Context-dependent=8。
# Related label 仅作为 soft prior，不能在 prompt 中直接当判决。
MANUAL_BEHAVIOR_RULES: list[tuple[str, str, str, str, str]] = [
    # ---- related_label = S （10 条）----
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
        "DexClassLoader PathClassLoader ClassLoader System.loadLibrary Runtime.load Runtime.loadLibrary reflection getMethods",
        "Dynamic loading and reflection can change runtime behavior beyond statically visible code paths.",
    ),
    (
        "NETWORK_EXFILTRATION_CONTEXT",
        "Network context",
        "S",
        "INTERNET ACCESS_NETWORK_STATE HttpGet HttpPost DefaultHttpClient URLDecoder",
        "Network APIs provide communication context and become more meaningful with identifier, contact, SMS, or location features.",
    ),
    (
        "LOCATION_TRACKING",
        "Location",
        "S",
        "ACCESS_FINE_LOCATION ACCESS_COARSE_LOCATION GPS INTERNET background",
        "Location features with network and background execution provide context for tracking or reporting location data.",
    ),
    (
        "PACKAGE_INSTALL_ABUSE",
        "Package install/remove",
        "S",
        "INSTALL_PACKAGES DELETE_PACKAGES PackageInstaller getInstalledPackages",
        "Install-package and delete-package capabilities together suggest payload-dropper or self-update behavior.",
    ),
    (
        "PRIVATE_API_REFLECTION",
        "Reflection on private API",
        "S",
        "Class.forName getMethod getDeclaredMethod setAccessible reflection invoke",
        "Reflective access to hidden or private API surface is common in apps that need to bypass static analysis.",
    ),
    (
        "SYSTEM_COMMAND_EXEC",
        "System command execution",
        "S",
        "Runtime.exec /system/bin /system/app mount remount chmod chown",
        "Direct execution of system commands and shell utilities expands an app's reach beyond the standard Android API surface.",
    ),
    (
        "SCREEN_OVERLAY_ABUSE",
        "Screen overlay",
        "S",
        "SYSTEM_ALERT_WINDOW TYPE_SYSTEM_ALERT_WINDOW WindowManager overlay",
        "System-alert overlays can be combined with input redirection to support phishing or tap-jacking attacks.",
    ),
    # ---- related_label = B （6 条）----
    (
        "APP_ENTRY",
        "App entry",
        "B",
        "android.intent.action.MAIN android.intent.category.LAUNCHER",
        "MAIN and LAUNCHER describe ordinary app entry points and are usually background context rather than evidence by themselves.",
    ),
    (
        "STANDARD_UI_INTENT",
        "Standard UI intent",
        "B",
        "android.intent.action.VIEW android.intent.action.SEND android.intent.action.PICK",
        "Standard UI-triggered intents describe ordinary user-initiated navigation between apps and components.",
    ),
    (
        "STANDARD_NETWORK_USAGE",
        "Basic network usage",
        "B",
        "INTERNET ACCESS_NETWORK_STATE HttpURLConnection URL",
        "Internet permission combined with network-state checking is a baseline pattern shared by most connected benign apps.",
    ),
    (
        "STANDARD_LIFECYCLE_BROADCASTS",
        "Lifecycle broadcasts",
        "B",
        "android.intent.action.PACKAGE_ADDED android.intent.action.SCREEN_ON android.intent.action.USER_PRESENT",
        "Listening to standard lifecycle broadcasts is a normal pattern for utility and accessory apps.",
    ),
    (
        "STANDARD_RESOURCE_ACCESS",
        "Standard resources",
        "B",
        "VIBRATE READ_SETTINGS WRITE_SETTINGS android.intent.action.SET_WALLPAPER",
        "Reading or writing standard system settings and resources is common in benign system utilities.",
    ),
    (
        "STANDARD_SERVICE_BIND",
        "Service bind",
        "B",
        "bindService onServiceConnected ServiceConnection",
        "Service binding via the standard Android lifecycle is part of normal component composition.",
    ),
    # ---- related_label = context-dependent （8 条）----
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
        "REFLECTION_GENERIC",
        "Generic reflection",
        "context-dependent",
        "Class.getClass getMethod getMethods getFields",
        "Reflection appears in plugins, ORM and framework code as well as obfuscated apps; combine with surrounding APIs.",
    ),
    (
        "INTENT_FILTER_RECEIVER",
        "Broadcast receiver",
        "context-dependent",
        "registerReceiver onReceive BroadcastReceiver IntentFilter",
        "Broadcast receivers are a standard Android pattern; their meaning depends on which events are observed.",
    ),
    (
        "NETWORK_HTTP_CLIENT",
        "Network HTTP client",
        "context-dependent",
        "HttpClient DefaultHttpClient HttpGet HttpPost URL",
        "HTTP client classes appear in both benign network-enabled apps and exfiltration paths; treat with other context features.",
    ),
    (
        "BACKGROUND_SERVICE",
        "Background service",
        "context-dependent",
        "startService Service onStartCommand FOREGROUND_SERVICE",
        "Background and foreground services are widely used for both legitimate sync work and silent execution.",
    ),
    (
        "CONTENT_PROVIDER_ACCESS",
        "Content provider",
        "context-dependent",
        "ContentResolver query insert delete READ_CONTACTS",
        "Content-provider access can serve normal contact / media features or sensitive data harvesting depending on what is queried.",
    ),
]


def manual_behavior_rules() -> list[dict[str, Any]]:
    docs: list[dict[str, Any]] = []
    for name, category, related_label, feature_pattern, explanation in MANUAL_BEHAVIOR_RULES:
        doc_id = sanitize_doc_id(name, prefix="RULE")
        docs.append(
            {
                "doc_id": doc_id,
                "doc_type": "behavior_rule",
                "rule_id": doc_id,
                "category": category,
                "related_label": related_label,
                "feature_pattern": feature_pattern,
                "explanation": explanation,
                "retrieval_text": (
                    f"Android behavior rule {name}. Related features: {feature_pattern}. "
                    f"Explanation: {explanation}"
                ),
                "prompt_text": (
                    f"[{doc_id}] Behavior rule: {category}. Related features: {feature_pattern}. "
                    f"Explanation: {explanation} Related label: {related_label} "
                    "(soft prior only, not a direct answer)."
                ),
                "source": "manual_android_security_knowledge",
                "leakage_risk": "no_sample_or_label",
            }
        )
    return docs


def assert_no_forbidden_fields(doc: dict[str, Any]) -> None:
    # 集合相交 == 字段名严格相等比较。`related_label` 不会被误判为 `label`。
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
    # 行为规则桶下界检查
    rules = [doc for doc in docs if doc.get("doc_type") == "behavior_rule"]
    by_label: dict[str, int] = {}
    for rule in rules:
        by_label[str(rule.get("related_label"))] = by_label.get(str(rule.get("related_label")), 0) + 1
    if len(rules) < 22:
        raise ValueError(f"behavior_rule count {len(rules)} < 22 lower bound")
    for label, lower in (("S", 10), ("B", 6), ("context-dependent", 6)):
        if by_label.get(label, 0) < lower:
            raise ValueError(
                f"behavior_rule related_label={label} count {by_label.get(label, 0)} < {lower}"
            )


def write_jsonl(path: Path, docs: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for doc in docs:
            handle.write(json.dumps(doc, ensure_ascii=False) + "\n")


def write_rules_csv(path: Path, docs: list[dict[str, Any]]) -> None:
    """按 PPT 要求导出 rules.csv，列名固定为：
    rule_id / feature_pattern / related_label / category / explanation。
    """
    rules = [doc for doc in docs if doc.get("doc_type") == "behavior_rule"]
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["rule_id", "feature_pattern", "related_label", "category", "explanation"],
        )
        writer.writeheader()
        for doc in rules:
            writer.writerow(
                {
                    "rule_id": doc.get("rule_id") or doc.get("doc_id"),
                    "feature_pattern": doc.get("feature_pattern", ""),
                    "related_label": doc.get("related_label"),
                    "category": doc.get("category"),
                    "explanation": doc.get("explanation", ""),
                }
            )


def write_doc_index_map(path: Path, items: list[tuple[int, dict[str, Any]]]) -> None:
    """items: list of (global_doc_index, doc)；只保留全局行号 → doc_id 对应。"""
    mapping = [
        {"sub_index": sub_index, "global_doc_index": global_index, "doc_id": doc.get("doc_id")}
        for sub_index, (global_index, doc) in enumerate(items)
    ]
    path.write_text(json.dumps(mapping, indent=2, ensure_ascii=False), encoding="utf-8")


def build_subset_index(
    docs: list[dict[str, Any]],
    *,
    doc_type: str,
    model_name: str,
    text_field: str,
    batch_size: int,
    local_files_only: bool,
    show_progress_bar: bool,
) -> tuple[Any, np.ndarray, list[tuple[int, dict[str, Any]]]]:
    """构建单一 doc_type 的子索引。"""
    items = [(index, doc) for index, doc in enumerate(docs) if doc.get("doc_type") == doc_type]
    if not items:
        raise ValueError(f"No docs of doc_type={doc_type}")
    subset = [doc for _, doc in items]
    index, embeddings = build_index(
        subset,
        model_name=model_name,
        text_field=text_field,
        batch_size=batch_size,
        local_files_only=local_files_only,
        show_progress_bar=show_progress_bar,
    )
    # 用全局 doc_index 重新覆盖 FAISS id：检索回来的 id 直接对应全量 kb_docs.jsonl 行号。
    faiss = require_faiss()
    base_index = faiss.IndexFlatIP(embeddings.shape[1])
    rebuilt = faiss.IndexIDMap2(base_index)
    ids = np.array([global_index for global_index, _ in items], dtype=np.int64)
    rebuilt.add_with_ids(embeddings, ids)
    return rebuilt, embeddings, items


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

    sub_indices_meta: dict[str, Any] = {}
    if not args.no_build_faiss:
        faiss = require_faiss()

        # 全量索引
        full_index, full_embeddings = build_index(
            docs,
            model_name=args.embedding_model,
            text_field="retrieval_text",
            batch_size=args.batch_size,
            local_files_only=args.local_files_only,
            show_progress_bar=args.show_progress,
        )
        np.save(output_dir / "kb_embeddings.npy", full_embeddings)
        faiss.write_index(full_index, str(output_dir / "kb.index"))

        # feature_card 子索引
        feature_index, _, feature_items = build_subset_index(
            docs,
            doc_type="feature_card",
            model_name=args.embedding_model,
            text_field="retrieval_text",
            batch_size=args.batch_size,
            local_files_only=args.local_files_only,
            show_progress_bar=args.show_progress,
        )
        faiss.write_index(feature_index, str(output_dir / "kb_feature.index"))
        write_doc_index_map(
            output_dir / "kb_feature_doc_index_map.json",
            feature_items,
        )

        # behavior_rule 子索引
        rule_index, _, rule_items = build_subset_index(
            docs,
            doc_type="behavior_rule",
            model_name=args.embedding_model,
            text_field="retrieval_text",
            batch_size=args.batch_size,
            local_files_only=args.local_files_only,
            show_progress_bar=args.show_progress,
        )
        faiss.write_index(rule_index, str(output_dir / "kb_rule.index"))
        write_doc_index_map(
            output_dir / "kb_rule_doc_index_map.json",
            rule_items,
        )

        sub_indices_meta = {
            "feature_card": {
                "index_path": "kb_feature.index",
                "doc_index_map": "kb_feature_doc_index_map.json",
                "size": len(feature_items),
            },
            "behavior_rule": {
                "index_path": "kb_rule.index",
                "doc_index_map": "kb_rule_doc_index_map.json",
                "size": len(rule_items),
            },
        }

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
        "forbidden_field_names": sorted(FORBIDDEN_FIELD_NAMES),
        "contains_embeddings": not args.no_build_faiss,
        "contains_faiss_index": not args.no_build_faiss,
        "contains_llm_outputs": False,
        "contains_test_samples": False,
        "contains_sample_labels": False,
    }

    if not args.no_build_faiss:
        metadata["stage_2_embedding_index"] = {
            "docs_path": str(docs_path),
            "embeddings_path": str(output_dir / "kb_embeddings.npy"),
            "index_path": str(output_dir / "kb.index"),
            "doc_count": len(docs),
            "embedding_model": args.embedding_model,
            "embedding_text_field": "retrieval_text",
            "embedding_normalized": True,
            "local_files_only": args.local_files_only,
            "faiss_index": "IndexIDMap2(IndexFlatIP)",
            "score": "inner_product_on_normalized_embeddings",
            "docs_fingerprint": docs_fingerprint(docs, "retrieval_text"),
        }
        metadata["sub_indices"] = sub_indices_meta

    metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Saved docs: {docs_path}")
    print(f"Saved metadata: {metadata_path}")
    print(f"Generated docs: {len(docs)}")
    if not args.no_build_faiss:
        print(f"Built sub_indices: {list(sub_indices_meta.keys())}")


if __name__ == "__main__":
    main()
