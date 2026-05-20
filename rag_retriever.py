from __future__ import annotations
"""Drebin LLM + RAG 实验的知识库检索器。

这个文件只负责“检索”这一层，不负责 few-shot 数据读取，也不负责调用 LLM。

它做的事情：
1. 读取 `data/processed/rag_kb/kb_docs.jsonl`。
2. 检查知识库文档里有没有可能造成标签泄漏的字段。
3. 构建或加载 BGE embedding 和 FAISS 索引。
4. 把当前测试样本的 query 转成向量。
5. 返回最相似的 top-k 条知识文档和相似度分数。

它刻意不做的事情：
- 不读取 few-shot split。
- 不调用 Ollama 或任何 LLM。
- 不知道测试样本的真实标签。

把检索逻辑单独放在这里，是为了方便调试：如果 LLM 结果不好，可以先单独检查
retrieval log，看看是不是检索出来的知识本身就有问题。
"""

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


DEFAULT_KB_DIR = Path("data/processed/rag_kb")
DEFAULT_MODEL_NAME = "BAAI/bge-small-en-v1.5"

# 这些字段名绝对不能出现在知识库文档的顶层。
# RAG 知识库只能包含背景知识，不能包含样本标签或预测结果。
# 这是本实验中最重要的数据泄漏防护之一。
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
    "test_source_indices",
}


@dataclass(frozen=True)
class RetrievalHit:
    """FAISS 返回的一条检索结果。

    `rank` 是它在 top-k 结果里的排名。
    `doc_index` 是它在 `kb_docs.jsonl` 里的行号。
    `score` 是归一化向量之间的内积，基本可以理解为 cosine similarity。
    `doc` 是知识库里原始的 JSON 对象。
    """

    rank: int
    doc_index: int
    score: float
    doc: dict[str, Any]

    def to_log_dict(self) -> dict[str, Any]:
        """返回一个适合写入 retrieval log 的简化版本。"""
        return {
            "rank": self.rank,
            "doc_index": self.doc_index,
            "score": self.score,
            "doc_id": self.doc.get("doc_id"),
            "doc_type": self.doc.get("doc_type"),
            "related_label": self.doc.get("related_label"),
            "feature": self.doc.get("feature"),
            "category": self.doc.get("category"),
        }


def require_faiss():
    """延迟导入 FAISS；如果缺依赖，就给出适合初学者看的安装提示。"""
    try:
        import faiss
    except ImportError as exc:
        raise SystemExit(
            "Missing dependency: faiss-cpu. Install it with:\n"
            "  python -m pip install faiss-cpu"
        ) from exc
    return faiss


def require_sentence_transformer():
    """延迟导入 SentenceTransformer；如果缺依赖，就给出安装提示。"""
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise SystemExit(
            "Missing dependency: sentence-transformers. Install it with:\n"
            "  python -m pip install sentence-transformers"
        ) from exc
    return SentenceTransformer


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    """读取 JSONL 格式的知识库文件。

    JSONL 指的是“一行一个完整 JSON 对象”。这个格式很适合 RAG，因为每一行
    都可以看作一个独立的、可被检索的知识片段。
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Missing knowledge-base docs: {path}")

    docs: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            text = line.strip()
            if not text:
                continue
            try:
                doc = json.loads(text)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{line_no}") from exc
            if not isinstance(doc, dict):
                raise ValueError(f"JSONL row is not an object at {path}:{line_no}")
            docs.append(doc)

    if not docs:
        raise ValueError(f"No docs found in {path}")
    return docs


def validate_docs(docs: list[dict[str, Any]], text_field: str = "retrieval_text") -> None:
    """在构建或查询索引之前检查知识库文档。

    这里的检查故意比较严格：
    - 每条文档必须有稳定的 `doc_id`；
    - `doc_id` 不能重复；
    - 不允许出现标签、预测结果等泄漏字段；
    - `retrieval_text` 和 `prompt_text` 都不能为空。
    """
    seen: set[str] = set()
    for position, doc in enumerate(docs):
        doc_id = str(doc.get("doc_id", "")).strip()
        if not doc_id:
            raise ValueError(f"Doc at position {position} has no doc_id")
        if doc_id in seen:
            raise ValueError(f"Duplicate doc_id: {doc_id}")
        seen.add(doc_id)

        forbidden = {str(key).lower() for key in doc} & FORBIDDEN_DOC_KEYS
        if forbidden:
            raise ValueError(f"{doc_id} has forbidden fields: {sorted(forbidden)}")

        if not str(doc.get(text_field, "")).strip():
            raise ValueError(f"{doc_id} has empty {text_field}")
        if not str(doc.get("prompt_text", "")).strip():
            raise ValueError(f"{doc_id} has empty prompt_text")


def docs_fingerprint(docs: list[dict[str, Any]], text_field: str) -> str:
    """给当前知识库内容计算一个稳定指纹。

    这个指纹会写进 metadata。下次运行时，如果文档内容、embedding 模型和文本字段
    都没有变化，就可以复用已有 FAISS 索引，而不用重新构建。
    """
    digest = hashlib.sha256()
    for doc in docs:
        digest.update(str(doc.get("doc_id", "")).encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(doc.get(text_field, "")).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def load_metadata(path: Path) -> dict[str, Any]:
    """读取 `kb_metadata.json`；如果文件不存在或损坏，就返回空字典。"""
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    return data if isinstance(data, dict) else {}


def save_metadata(path: Path, metadata: dict[str, Any]) -> None:
    """用易读的 JSON 格式写入 metadata。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")


def encode_texts(
    model: Any,
    texts: list[str],
    *,
    batch_size: int,
    show_progress_bar: bool,
) -> np.ndarray:
    """把文本编码成归一化 embedding。

    `normalize_embeddings=True` 很重要。向量归一化之后，FAISS 的 `IndexFlatIP`
    内积检索基本等价于 cosine similarity。
    """
    vectors = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=show_progress_bar,
        normalize_embeddings=True,
    )
    vectors = np.asarray(vectors, dtype=np.float32)
    if vectors.ndim != 2 or vectors.shape[0] != len(texts):
        raise ValueError(f"Unexpected embedding shape: {vectors.shape}")
    return vectors


def build_index(
    docs: list[dict[str, Any]],
    *,
    model_name: str,
    text_field: str,
    batch_size: int,
    local_files_only: bool,
    show_progress_bar: bool = True,
) -> tuple[Any, np.ndarray]:
    """根据知识库文档构建 FAISS 索引。

    索引里保存的是“向量 + 整数 id”。这里直接用文档在 `docs` 里的行号作为
    FAISS id，因此检索返回的 id 可以直接用来取回原始文档。
    """
    faiss = require_faiss()
    SentenceTransformer = require_sentence_transformer()

    model = SentenceTransformer(model_name, local_files_only=local_files_only)
    texts = [str(doc[text_field]) for doc in docs]
    embeddings = encode_texts(
        model,
        texts,
        batch_size=batch_size,
        show_progress_bar=show_progress_bar,
    )

    base_index = faiss.IndexFlatIP(embeddings.shape[1])
    index = faiss.IndexIDMap2(base_index)
    ids = np.arange(len(docs), dtype=np.int64)
    index.add_with_ids(embeddings, ids)
    return index, embeddings


class RagRetriever:
    """加载或构建 RAG 索引，并提供简单的 `retrieve()` 方法。

    典型用法：

    ```python
    retriever = RagRetriever(kb_dir="data/processed/rag_kb")
    hits = retriever.retrieve("SEND_SMS READ_PHONE_STATE INTERNET", top_k=3)
    ```
    """

    def __init__(
        self,
        *,
        kb_dir: str | Path = DEFAULT_KB_DIR,
        docs_name: str = "kb_docs.jsonl",
        index_name: str = "kb.index",
        embeddings_name: str = "kb_embeddings.npy",
        metadata_name: str = "kb_metadata.json",
        model_name: str = DEFAULT_MODEL_NAME,
        text_field: str = "retrieval_text",
        batch_size: int = 32,
        rebuild: bool = False,
        local_files_only: bool = False,
        show_progress_bar: bool = True,
    ) -> None:
        # 先把所有路径统一转成 Path 对象。这样调用方传字符串或 Path 都可以。
        self.kb_dir = Path(kb_dir)
        self.docs_path = self.kb_dir / docs_name
        self.index_path = self.kb_dir / index_name
        self.embeddings_path = self.kb_dir / embeddings_name
        self.metadata_path = self.kb_dir / metadata_name
        self.model_name = model_name
        self.text_field = text_field
        self.batch_size = batch_size
        self.local_files_only = local_files_only
        self.show_progress_bar = show_progress_bar

        # 先读取并检查知识库文档，再处理 embedding。
        # 如果知识库存在标签泄漏或格式错误，就尽早失败。
        self.docs = read_jsonl(self.docs_path)
        validate_docs(self.docs, text_field=self.text_field)
        self.metadata = load_metadata(self.metadata_path)

        # 如果 metadata 证明当前文档和已有索引匹配，就可以复用旧索引。
        self.index = self._load_or_build_index(rebuild=rebuild)

        # embedding 模型采用延迟加载。只有真正执行查询时才加载模型，避免初始化时太慢。
        self._model: Any | None = None

    def _index_is_current(self) -> bool:
        """如果已保存的 embedding/index 与当前知识库设置一致，则返回 True。"""
        if not self.index_path.exists() or not self.embeddings_path.exists():
            return False
        stage = self.metadata.get("stage_2_embedding_index")
        if not isinstance(stage, dict):
            return False
        return (
            stage.get("doc_count") == len(self.docs)
            and stage.get("embedding_model") == self.model_name
            and stage.get("embedding_text_field") == self.text_field
            and stage.get("docs_fingerprint") == docs_fingerprint(self.docs, self.text_field)
        )

    def _load_or_build_index(self, *, rebuild: bool) -> Any:
        """加载已有 FAISS 索引；如果不匹配或用户要求重建，就重新构建。"""
        faiss = require_faiss()

        # 快路径：metadata 显示文档、模型、文本字段都没有变化。
        if not rebuild and self._index_is_current():
            index = faiss.read_index(str(self.index_path))
            if index.ntotal != len(self.docs):
                raise ValueError(
                    f"Index/doc mismatch: index has {index.ntotal}, docs has {len(self.docs)}. "
                    "Run with --rebuild-index."
                )
            return index

        # 慢路径：文档变了、索引缺失，或者用户显式要求重建。
        index, embeddings = build_index(
            self.docs,
            model_name=self.model_name,
            text_field=self.text_field,
            batch_size=self.batch_size,
            local_files_only=self.local_files_only,
            show_progress_bar=self.show_progress_bar,
        )

        self.kb_dir.mkdir(parents=True, exist_ok=True)
        np.save(self.embeddings_path, embeddings)
        faiss.write_index(index, str(self.index_path))

        # 保存足够的 metadata，便于复现实验，也便于下次判断是否能复用缓存。
        updated = dict(self.metadata)
        updated.update(
            {
                "contains_embeddings": True,
                "contains_faiss_index": True,
                "stage_2_embedding_index": {
                    "docs_path": str(self.docs_path),
                    "embeddings_path": str(self.embeddings_path),
                    "index_path": str(self.index_path),
                    "doc_count": len(self.docs),
                    "embedding_model": self.model_name,
                    "embedding_text_field": self.text_field,
                    "embedding_normalized": True,
                    "local_files_only": self.local_files_only,
                    "faiss_index": "IndexIDMap2(IndexFlatIP)",
                    "score": "inner_product_on_normalized_embeddings",
                    "docs_fingerprint": docs_fingerprint(self.docs, self.text_field),
                },
            }
        )
        save_metadata(self.metadata_path, updated)
        self.metadata = updated
        return index

    @property
    def model(self) -> Any:
        """第一次真正需要查询向量时，再加载 SentenceTransformer 模型。"""
        if self._model is None:
            SentenceTransformer = require_sentence_transformer()
            self._model = SentenceTransformer(
                self.model_name,
                local_files_only=self.local_files_only,
            )
        return self._model

    def retrieve(self, query_text: str, top_k: int = 3) -> list[RetrievalHit]:
        """为一个测试样本 query 返回 top-k 条知识文档。

        `query_text` 应该只由当前样本的 active Drebin features 构成，不能包含
        当前样本的真实标签。
        """
        query_text = str(query_text).strip()
        if not query_text:
            raise ValueError("query_text is empty")

        query_embedding = encode_texts(
            self.model,
            [query_text],
            batch_size=self.batch_size,
            show_progress_bar=False,
        )

        # 把 k 限制在合法范围内，避免向 FAISS 请求超过知识库总数的结果。
        k = min(max(1, int(top_k)), len(self.docs))
        scores, doc_indices = self.index.search(query_embedding, k)

        hits: list[RetrievalHit] = []
        for rank, (doc_index, score) in enumerate(zip(doc_indices[0], scores[0]), start=1):
            if doc_index < 0:
                continue
            hits.append(
                RetrievalHit(
                    rank=rank,
                    doc_index=int(doc_index),
                    score=float(score),
                    doc=self.docs[int(doc_index)],
                )
            )
        return hits

    def retrieve_by_doc_type(
        self,
        query_text: str,
        *,
        doc_type: str,
        top_k: int,
    ) -> list[RetrievalHit]:
        """Retrieve top hits from one doc_type bucket."""
        if top_k <= 0:
            return []
        oversample = min(len(self.docs), max(top_k * 8, top_k + 20))
        candidates = self.retrieve(query_text, top_k=oversample)
        hits: list[RetrievalHit] = []
        for hit in candidates:
            if hit.doc.get("doc_type") != doc_type:
                continue
            hits.append(
                RetrievalHit(
                    rank=len(hits) + 1,
                    doc_index=hit.doc_index,
                    score=hit.score,
                    doc=hit.doc,
                )
            )
            if len(hits) >= top_k:
                break
        return hits

    def retrieve_bucketed(
        self,
        *,
        feature_query: str,
        rule_query: str,
        feature_top_k: int = 2,
        rule_top_k: int = 3,
    ) -> list[RetrievalHit]:
        """Retrieve feature cards and behavior rules in fixed buckets."""
        feature_hits = self.retrieve_by_doc_type(
            feature_query,
            doc_type="feature_card",
            top_k=feature_top_k,
        )
        rule_hits = self.retrieve_by_doc_type(
            rule_query,
            doc_type="behavior_rule",
            top_k=rule_top_k,
        )
        merged: list[RetrievalHit] = []
        for hit in [*feature_hits, *rule_hits]:
            merged.append(
                RetrievalHit(
                    rank=len(merged) + 1,
                    doc_index=hit.doc_index,
                    score=hit.score,
                    doc=hit.doc,
                )
            )
        return merged


def parse_args() -> argparse.Namespace:
    """解析命令行参数，用于手动调试检索质量。"""
    parser = argparse.ArgumentParser(description="Retrieve top-k docs from the Drebin RAG KB.")
    parser.add_argument("--kb-dir", default=str(DEFAULT_KB_DIR))
    parser.add_argument("--model", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--rebuild-index", action="store_true")
    parser.add_argument(
        "--local-files-only",
        action="store_true",
        help="Load the embedding model from the local Hugging Face cache only.",
    )
    parser.add_argument(
        "--query",
        default="SEND_SMS READ_PHONE_STATE BOOT_COMPLETED INTERNET SmsManager",
    )
    return parser.parse_args()


def main() -> None:
    """命令行入口：手动输入 query，查看检索结果。"""
    args = parse_args()
    retriever = RagRetriever(
        kb_dir=args.kb_dir,
        model_name=args.model,
        batch_size=args.batch_size,
        rebuild=args.rebuild_index,
        local_files_only=args.local_files_only,
    )
    hits = retriever.retrieve(args.query, top_k=args.top_k)

    print("Query:")
    print(args.query)
    for hit in hits:
        doc = hit.doc
        print("=" * 80)
        print(f"Rank {hit.rank} | score={hit.score:.4f} | doc_id={doc.get('doc_id')}")
        print(f"type={doc.get('doc_type')} | category={doc.get('category')}")
        print(doc.get("retrieval_text", ""))


if __name__ == "__main__":
    main()
