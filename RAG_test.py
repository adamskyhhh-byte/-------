from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


DEFAULT_KB_DIR = Path("data/processed/rag_kb")
DEFAULT_MODEL_NAME = "BAAI/bge-small-en-v1.5"
DEFAULT_QUERY = (
    "SEND_SMS READ_PHONE_STATE BOOT_COMPLETED INTERNET SMS fraud "
    "device identifier persistence"
)
FORBIDDEN_DOC_KEYS = {
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


@dataclass(frozen=True)
class RetrievalHit:
    rank: int
    doc_index: int
    score: float
    doc: dict[str, Any]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Stage-2 RAG demo: build/load embeddings and a FAISS index for "
            "the Drebin knowledge base, then retrieve top-k knowledge docs."
        )
    )
    parser.add_argument("--kb-dir", default=str(DEFAULT_KB_DIR))
    parser.add_argument("--docs-name", default="kb_docs.jsonl")
    parser.add_argument("--embeddings-name", default="kb_embeddings.npy")
    parser.add_argument("--index-name", default="kb.index")
    parser.add_argument("--metadata-name", default="kb_metadata.json")
    parser.add_argument("--model", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--text-field", default="retrieval_text")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--rebuild", action="store_true")
    parser.add_argument(
        "--query",
        default=None,
        help="Free-form retrieval query. If omitted, a small SMS/device demo query is used.",
    )
    parser.add_argument(
        "--features",
        nargs="*",
        default=None,
        help=(
            "Active Drebin features. Commas and spaces are both accepted, "
            "for example: --features SEND_SMS READ_PHONE_STATE INTERNET"
        ),
    )
    parser.add_argument(
        "--test-row-index",
        type=int,
        default=None,
        help="Use one row from split-root/test.csv as the retrieval query.",
    )
    parser.add_argument(
        "--split-root",
        default="data/processed/fewshot_seed42_test100",
        help="Few-shot split root used when --test-row-index is set.",
    )
    parser.add_argument(
        "--feature-category-path",
        default="data/dataset-features-categories.csv",
        help="Optional feature category file used when formatting a test-row query.",
    )
    parser.add_argument(
        "--show-prompt-text",
        action="store_true",
        help="Print prompt_text in addition to retrieval_text for each hit.",
    )
    parser.add_argument("--max-text-chars", type=int, default=600)
    return parser.parse_args()


def require_faiss():
    try:
        import faiss
    except ImportError as exc:
        raise SystemExit(
            "Missing dependency: faiss. Install it with:\n"
            "  python -m pip install faiss-cpu"
        ) from exc
    return faiss


def require_sentence_transformer():
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise SystemExit(
            "Missing dependency: sentence-transformers. Install it with:\n"
            "  python -m pip install sentence-transformers"
        ) from exc
    return SentenceTransformer


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Missing knowledge-base docs: {path}")

    docs: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                doc = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{line_no}") from exc
            if not isinstance(doc, dict):
                raise ValueError(f"JSONL row is not an object at {path}:{line_no}")
            docs.append(doc)
    if not docs:
        raise ValueError(f"No docs found in {path}")
    return docs


def validate_docs(docs: list[dict[str, Any]], text_field: str) -> None:
    seen_doc_ids: set[str] = set()
    for position, doc in enumerate(docs):
        doc_id = str(doc.get("doc_id", "")).strip()
        if not doc_id:
            raise ValueError(f"Doc at position {position} has no doc_id")
        if doc_id in seen_doc_ids:
            raise ValueError(f"Duplicate doc_id: {doc_id}")
        seen_doc_ids.add(doc_id)

        lowered_keys = {str(key).lower() for key in doc}
        leaked = lowered_keys & FORBIDDEN_DOC_KEYS
        if leaked:
            raise ValueError(f"{doc_id} has forbidden top-level keys: {sorted(leaked)}")

        text = str(doc.get(text_field, "")).strip()
        if not text:
            raise ValueError(f"{doc_id} has empty {text_field}")


def docs_fingerprint(docs: list[dict[str, Any]], text_field: str) -> str:
    digest = hashlib.sha256()
    for doc in docs:
        digest.update(str(doc.get("doc_id", "")).encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(doc.get(text_field, "")).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def load_metadata(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    return data if isinstance(data, dict) else {}


def write_stage2_metadata(
    metadata_path: Path,
    *,
    metadata: dict[str, Any],
    docs_path: Path,
    embeddings_path: Path,
    index_path: Path,
    doc_count: int,
    model_name: str,
    text_field: str,
    fingerprint: str,
) -> None:
    updated = dict(metadata)
    updated.update(
        {
            "stage_2_embedding_index": {
                "docs_path": str(docs_path),
                "embeddings_path": str(embeddings_path),
                "index_path": str(index_path),
                "doc_count": doc_count,
                "embedding_model": model_name,
                "embedding_text_field": text_field,
                "embedding_normalized": True,
                "faiss_index": "IndexIDMap2(IndexFlatIP)",
                "score": "inner_product_on_normalized_embeddings",
                "docs_fingerprint": fingerprint,
            },
            "contains_embeddings": True,
            "contains_faiss_index": True,
        }
    )
    metadata_path.write_text(json.dumps(updated, indent=2, ensure_ascii=False), encoding="utf-8")


def model_encode(model: Any, texts: list[str], batch_size: int) -> np.ndarray:
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        normalize_embeddings=True,
    )
    embeddings = np.asarray(embeddings, dtype=np.float32)
    if embeddings.ndim != 2 or embeddings.shape[0] != len(texts):
        raise ValueError(f"Unexpected embedding shape: {embeddings.shape}")
    return embeddings


def build_and_save_index(
    *,
    docs: list[dict[str, Any]],
    text_field: str,
    model_name: str,
    batch_size: int,
    embeddings_path: Path,
    index_path: Path,
) -> tuple[Any, Any]:
    faiss = require_faiss()
    SentenceTransformer = require_sentence_transformer()

    texts = [str(doc[text_field]) for doc in docs]
    print(f"Loading embedding model: {model_name}")
    model = SentenceTransformer(model_name)
    print(f"Encoding {len(texts)} knowledge docs...")
    embeddings = model_encode(model, texts, batch_size=batch_size)

    base_index = faiss.IndexFlatIP(embeddings.shape[1])
    index = faiss.IndexIDMap2(base_index)
    ids = np.arange(len(docs), dtype=np.int64)
    index.add_with_ids(embeddings, ids)

    embeddings_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(embeddings_path, embeddings)
    faiss.write_index(index, str(index_path))
    return index, model


def load_index(index_path: Path) -> Any:
    faiss = require_faiss()
    if not index_path.exists():
        raise FileNotFoundError(f"Missing FAISS index: {index_path}")
    return faiss.read_index(str(index_path))


def should_rebuild(
    *,
    rebuild: bool,
    index_path: Path,
    embeddings_path: Path,
    metadata: dict[str, Any],
    docs_count: int,
    model_name: str,
    text_field: str,
    fingerprint: str,
) -> bool:
    if rebuild:
        return True
    if not index_path.exists() or not embeddings_path.exists():
        return True

    stage2 = metadata.get("stage_2_embedding_index")
    if not isinstance(stage2, dict):
        return True
    return not (
        stage2.get("doc_count") == docs_count
        and stage2.get("embedding_model") == model_name
        and stage2.get("embedding_text_field") == text_field
        and stage2.get("docs_fingerprint") == fingerprint
    )


def get_or_build_index(
    *,
    docs: list[dict[str, Any]],
    args: argparse.Namespace,
    docs_path: Path,
    embeddings_path: Path,
    index_path: Path,
    metadata_path: Path,
) -> tuple[Any, Any | None]:
    metadata = load_metadata(metadata_path)
    fingerprint = docs_fingerprint(docs, args.text_field)
    if should_rebuild(
        rebuild=args.rebuild,
        index_path=index_path,
        embeddings_path=embeddings_path,
        metadata=metadata,
        docs_count=len(docs),
        model_name=args.model,
        text_field=args.text_field,
        fingerprint=fingerprint,
    ):
        index, model = build_and_save_index(
            docs=docs,
            text_field=args.text_field,
            model_name=args.model,
            batch_size=args.batch_size,
            embeddings_path=embeddings_path,
            index_path=index_path,
        )
        write_stage2_metadata(
            metadata_path,
            metadata=metadata,
            docs_path=docs_path,
            embeddings_path=embeddings_path,
            index_path=index_path,
            doc_count=len(docs),
            model_name=args.model,
            text_field=args.text_field,
            fingerprint=fingerprint,
        )
        return index, model

    print(f"Loading existing FAISS index: {index_path}")
    index = load_index(index_path)
    if index.ntotal != len(docs):
        raise ValueError(
            f"Index/doc mismatch: index has {index.ntotal} vectors, docs has {len(docs)} rows. "
            "Run again with --rebuild."
        )
    return index, None


def split_features(raw_features: list[str] | None) -> list[str]:
    if not raw_features:
        return []
    features: list[str] = []
    for value in raw_features:
        for item in value.replace(",", " ").split():
            item = item.strip()
            if item:
                features.append(item)
    return features


def build_query_from_features(features: list[str]) -> str:
    return (
        "Android app active Drebin features: "
        + ", ".join(features)
        + ". Retrieve feature meanings and behavior patterns for malware analysis."
    )


def build_query_from_test_row(
    *,
    split_root: Path,
    row_index: int,
    feature_category_path: Path,
) -> str:
    import pandas as pd

    from fewshot_utils import load_feature_categories, row_to_feature_text

    test_path = split_root / "test.csv"
    if not test_path.exists():
        raise FileNotFoundError(f"Missing test split: {test_path}")

    test_df = pd.read_csv(test_path, low_memory=False)
    if row_index < 0 or row_index >= len(test_df):
        raise IndexError(f"--test-row-index must be in [0, {len(test_df) - 1}]")

    feature_categories = load_feature_categories(feature_category_path)
    feature_text = row_to_feature_text(test_df.iloc[row_index], feature_categories=feature_categories)
    return (
        f"Android app active Drebin features from test.csv row {row_index}:\n"
        f"{feature_text}\n"
        "Retrieve feature meanings and behavior patterns for malware analysis."
    )


def choose_query(args: argparse.Namespace) -> str:
    features = split_features(args.features)
    query_modes = sum(
        [
            bool(args.query),
            bool(features),
            args.test_row_index is not None,
        ]
    )
    if query_modes > 1:
        raise ValueError("Use only one query source: --query, --features, or --test-row-index.")
    if args.query:
        return args.query
    if features:
        return build_query_from_features(features)
    if args.test_row_index is not None:
        return build_query_from_test_row(
            split_root=Path(args.split_root),
            row_index=args.test_row_index,
            feature_category_path=Path(args.feature_category_path),
        )
    return DEFAULT_QUERY


def retrieve(
    *,
    docs: list[dict[str, Any]],
    index: Any,
    model: Any | None,
    query: str,
    model_name: str,
    batch_size: int,
    top_k: int,
) -> list[RetrievalHit]:
    if model is None:
        SentenceTransformer = require_sentence_transformer()
        print(f"Loading embedding model for query: {model_name}")
        model = SentenceTransformer(model_name)
    query_embedding = model_encode(model, [query], batch_size=batch_size)

    k = min(top_k, len(docs))
    scores, doc_indices = index.search(query_embedding, k)

    hits: list[RetrievalHit] = []
    for rank, (doc_index, score) in enumerate(zip(doc_indices[0], scores[0]), start=1):
        if doc_index < 0:
            continue
        hits.append(
            RetrievalHit(
                rank=rank,
                doc_index=int(doc_index),
                score=float(score),
                doc=docs[int(doc_index)],
            )
        )
    return hits


def shorten(text: Any, max_chars: int) -> str:
    value = str(text).replace("\r\n", "\n").strip()
    if len(value) <= max_chars:
        return value
    return value[: max(0, max_chars - 3)].rstrip() + "..."


def print_hits(hits: list[RetrievalHit], *, show_prompt_text: bool, max_text_chars: int) -> None:
    for hit in hits:
        doc = hit.doc
        print(f"\nRank {hit.rank} | score={hit.score:.4f} | doc_index={hit.doc_index}")
        print(f"doc_id: {doc.get('doc_id')}")
        print(f"doc_type: {doc.get('doc_type')}")
        if doc.get("feature") is not None:
            print(f"feature: {doc.get('feature')}")
        if doc.get("category") is not None:
            print(f"category: {doc.get('category')}")
        print("retrieval_text:")
        print(shorten(doc.get("retrieval_text", ""), max_text_chars))
        if show_prompt_text:
            print("prompt_text:")
            print(shorten(doc.get("prompt_text", ""), max_text_chars))


def main() -> None:
    args = parse_args()
    kb_dir = Path(args.kb_dir)
    docs_path = kb_dir / args.docs_name
    embeddings_path = kb_dir / args.embeddings_name
    index_path = kb_dir / args.index_name
    metadata_path = kb_dir / args.metadata_name

    docs = read_jsonl(docs_path)
    validate_docs(docs, args.text_field)

    index, model = get_or_build_index(
        docs=docs,
        args=args,
        docs_path=docs_path,
        embeddings_path=embeddings_path,
        index_path=index_path,
        metadata_path=metadata_path,
    )
    query = choose_query(args)
    print("\nQuery:")
    print(query)

    hits = retrieve(
        docs=docs,
        index=index,
        model=model,
        query=query,
        model_name=args.model,
        batch_size=args.batch_size,
        top_k=args.top_k,
    )
    print_hits(hits, show_prompt_text=args.show_prompt_text, max_text_chars=args.max_text_chars)


if __name__ == "__main__":
    main()
