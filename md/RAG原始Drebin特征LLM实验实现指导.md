# RAG + 原始 Drebin 特征 LLM 推理实验实现指导

这份文档面向当前项目，而不是泛泛介绍 RAG。你的目标不是“把 RAG 框架接进来”，而是在已有三类实验基础上，新增一个可公平对比、可调试、可解释的实验：

> 当前测试样本仍然以原始 active Drebin features 作为主要输入；在 LLM 推理前，先从知识库检索相关解释性知识；再把“few-shot 示例 + 当前样本原始特征 + 检索知识”一起放进 prompt，让 LLM 输出 `B` 或 `S` 以及解释。

建议文件名就是本文件：`md/RAG原始Drebin特征LLM实验实现指导.md`。

## 1. 先把这个实验放回当前项目里

你现在已经有的实验大致是：

| 已有实验 | 主要文件 | 作用 |
|---|---|---|
| 传统 ML few-shot baseline | `baseline_fewshot.py` | 用同一个 split 做传统分类器对比 |
| 原始 Drebin 特征 LLM-only | `llm_only_fewshot.py` | few-shot 示例 + 当前样本 active raw features |
| 原始/语义化特征表达对比 | `llm_feature_expr_fewshot.py`, `paper_prompt_utils.py` | 比较 `raw` 和 `semantic` 两种输入表达 |
| 特征语义说明生成 | `generate_feature_semantics.py` | 为 215 个 Drebin 特征生成中性中文解释 |

新增 RAG 实验时，推荐继续沿用这个工程风格：

- 不改 `llm_only_fewshot.py`，让它继续作为 LLM-only baseline。
- 不改 `prepare_fewshot_split.py`，继续使用同一个 `data/processed/fewshot_seed42_test100/`。
- 复用 `fewshot_utils.py` 的数据读取、标签规范化、raw 特征文本生成。
- 复用 `llm_only_fewshot.py` 的 Ollama 调用、JSON 解析和记录构造逻辑。
- 新增 RAG 相关文件，而不是把检索逻辑塞进已有 LLM-only 脚本。

建议新增：

```text
rag_kb_builder.py              # 构建知识库文档、embedding 和 FAISS index
rag_retriever.py               # 只负责加载 index 并执行 top-k 检索
llm_rag_raw_fewshot.py         # RAG + raw Drebin features 的实验 runner

data/processed/rag_kb/
  kb_docs.jsonl                # 文本知识库，每行一个知识片段
  kb_embeddings.npy            # 可选，保存向量方便检查
  kb.index                     # FAISS 索引
  kb_metadata.json             # 构建参数、模型名、doc 数量

results/rag_raw_llm/
  seed42_test100/k5/
    rag_raw_top3.jsonl
    rag_raw_top3_metrics.json
    retrieval_logs.jsonl
    prompts/
```

这样做的好处是：RAG 实验的唯一新增变量就是“检索知识”，已有实验不会被破坏。

## 2. 先回答最容易混的几个问题

### 2.1 RAG 里面到底 embedding 什么？

Embedding 的对象有两类：

1. 知识库文档：离线 embedding 一次。
2. 当前测试样本的查询文本：每预测一个样本时临时 embedding 一次。

不要 embedding LLM 的 prompt 全文，也不要 embedding 测试样本标签。

在你的项目里，最小可运行版可以 embedding 这些知识文档：

```text
Feature: SEND_SMS
Category: Manifest Permission
Meaning: Requests permission to send SMS messages.
Security note: When combined with SMS receive/read permissions or network access, it may indicate SMS abuse or data exfiltration.
```

查询则来自当前样本的 active raw features，例如：

```text
Manifest Permission: SEND_SMS, READ_PHONE_STATE, INTERNET
API call signature: android.telephony.SmsManager, Ljava.lang.Class.getMethods
Intent: android.intent.action.BOOT_COMPLETED
```

也就是说：知识库是被检索的“书”，当前测试样本是“问题”。

### 2.2 当前测试样本是知识库，还是 query？

当前测试样本必须是 query，不是知识库。

RAG 在这个实验中的角色是“给当前样本找相关安全知识”，不是“把测试样本放进去让它被查到”。如果你把测试集样本做进知识库，尤其还带上标签，那就变成了数据泄漏，实验结果会失真。

可以放进知识库的内容：

- Drebin 特征解释：`SEND_SMS`、`READ_PHONE_STATE`、`Ljava.lang.Class.getMethods` 等是什么意思。
- Android 权限说明：权限能做什么。
- API 行为说明：API 通常代表什么行为。
- Intent / Component 说明：系统事件、组件交互含义。
- 恶意软件行为知识：例如短信滥用、设备指纹收集、开机持久化、动态加载、root 命令等。
- 手写安全规则：例如“短信权限 + 设备标识 + 联网”为什么需要关注。

不能放进知识库的内容：

- 测试集行数据。
- 测试集真实标签。
- 对测试样本的人工答案或 LLM 预测结果。
- 根据全量数据计算出的“某特征在恶意样本中出现比例”这类统计，除非只用训练侧数据并在报告里明确说明。
- `results/` 中的预测结果、错误分析、指标文件。

### 2.3 sentence-transformers、BAAI/bge-small-en-v1.5、faiss-cpu 分别做什么？

| 组件 | 在流程中的角色 | 你应该怎么理解 |
|---|---|---|
| `sentence-transformers` | Python 库 | 负责加载 embedding 模型，并把文本转成向量 |
| `BAAI/bge-small-en-v1.5` | embedding 模型 | 把知识片段和 query 映射到同一个向量空间 |
| `faiss-cpu` | 向量索引库 | 保存知识库向量，并快速找出和 query 最相似的 Top-K 文档 |

更直白一点：

```text
文本知识片段 -> sentence-transformers + BGE -> 向量
当前样本 query -> sentence-transformers + BGE -> 向量
query 向量 + FAISS index -> 最相关的知识片段 id
```

如果使用 `normalize_embeddings=True`，再配合 `faiss.IndexFlatIP`，内积相似度基本就等价于 cosine similarity。你的知识库规模很小，`IndexFlatIP` 足够，不需要复杂 ANN 索引。

一个重要提醒：`BAAI/bge-small-en-v1.5` 是英文 embedding 模型。如果知识库主体是中文，检索质量可能不稳。当前项目里更推荐这样处理：

- `retrieval_text` 用英文或中英混合，但保留原始特征名。
- `prompt_text` 可以写中文解释，给 LLM 阅读。
- 如果你想知识库全中文，可以改用 `BAAI/bge-small-zh-v1.5`，但文档和实验记录里要写清楚 embedding 模型换了。

## 3. 知识库应该怎么构建

### 3.1 推荐的第一版知识库：特征知识卡 + 少量行为规则

最小可运行版本不需要一上来爬 Android 官方文档。你已有两个非常适合做知识库的来源：

1. `data/dataset-features-categories.csv`
2. `data/processed/paper_features/feature_semantics_gemma.json`

第一版知识库可以由两部分组成：

| 文档类型 | 数量 | 来源 | 作用 |
|---|---:|---|---|
| Feature card | 约 215 条 | Drebin 特征名 + 类别 + 已生成语义 | 让检索能命中特定权限、API、Intent、Command |
| Behavior rule | 20 到 40 条 | 你手写的安全行为知识 | 让 LLM 看到组合行为解释，而不只是单个特征解释 |

知识库文档建议用 JSONL，每行一个对象：

```json
{
  "doc_id": "FEAT_SEND_SMS",
  "doc_type": "feature_card",
  "feature": "SEND_SMS",
  "category": "Manifest Permission",
  "retrieval_text": "Feature SEND_SMS. Android manifest permission. Allows an app to send SMS messages.",
  "prompt_text": "[FEAT_SEND_SMS] SEND_SMS: 请求发送短信的系统权限。若与读取短信、接收短信、联网或设备标识收集同时出现，需要关注短信滥用或信息外传行为。",
  "source": "dataset-features-categories + feature_semantics_gemma",
  "leakage_risk": "no_label"
}
```

规则类文档可以这样写：

```json
{
  "doc_id": "RULE_SMS_ABUSE",
  "doc_type": "behavior_rule",
  "category": "SMS abuse",
  "retrieval_text": "SMS abuse pattern. SEND_SMS, READ_SMS, RECEIVE_SMS, SmsManager, INTERNET, READ_PHONE_STATE.",
  "prompt_text": "[RULE_SMS_ABUSE] 短信相关权限或 SmsManager 与联网、设备标识收集同时出现时，常用于短信扣费、验证码拦截或信息上传，需要结合上下文判断。",
  "source": "manual_security_knowledge",
  "leakage_risk": "no_sample_label"
}
```

第一版不要求规则非常多。你真正要先验证的是：检索链路能工作，检索内容能进入 prompt，最终结果能和 LLM-only 对比。

### 3.2 不建议第一版直接做的知识库

| 路线 | 问题 |
|---|---|
| 把训练样本做成 labeled case base | 会变成“检索相似样本 + 标签”的 case-based 推理，不再是你现在要做的知识增强实验 |
| 把测试集也放进知识库 | 严重数据泄漏 |
| 用全量数据统计恶意比例并写入知识库 | 如果包含测试集，会泄漏标签分布信息 |
| 一次性加入大量官方文档 | 检索噪声大、调试困难、时间成本高 |

如果以后要做“相似样本 RAG”，可以作为单独实验，名字要写成 `case_retrieval_rag`，并且只允许检索训练集样本。

## 4. RAG 每一步的输入、输出和作用

### 4.1 离线构建阶段

| 步骤 | 输入 | 输出 | 作用 |
|---|---|---|---|
| 生成知识文档 | 特征类别 CSV、特征语义 JSON、手写规则 | `kb_docs.jsonl` | 把知识整理成可检索片段 |
| 选择检索文本 | 每个 doc 的 `retrieval_text` | `texts: list[str]` | 控制 embedding 看见什么 |
| 文本向量化 | `texts` | `kb_embeddings.npy` | 把知识库变成向量 |
| 建 FAISS index | embedding 矩阵 | `kb.index` | 支持快速 Top-K 检索 |
| 保存元数据 | 模型名、doc 数量、时间、来源 | `kb_metadata.json` | 保证实验可复现 |

### 4.2 在线推理阶段

| 步骤 | 输入 | 输出 | 作用 |
|---|---|---|---|
| 读取 split | `split_root`, `k` | `train_df`, `test_df` | 保持和已有实验同一数据划分 |
| 构造 few-shot 示例 | `train_df` | prompt 中的 examples | 和 LLM-only 公平对比 |
| 构造当前样本 query | 当前测试行，不含 `class` | `query_text` | 用 active raw features 找相关知识 |
| query embedding | `query_text` | `query_vector` | 进入 FAISS 检索 |
| Top-K 检索 | `query_vector`, `kb.index` | `retrieved_docs` | 找相关解释性知识 |
| 拼 prompt | examples + raw features + retrieved docs | `prompt` | 给 LLM 分类和解释所需上下文 |
| LLM 推理 | `prompt` | raw JSON 文本 | 输出分类结果 |
| 解析和评估 | raw JSON + true label | JSONL + metrics | 和已有实验横向比较 |
| 保存 retrieval log | query + doc ids + scores | `retrieval_logs.jsonl` | 后续分析 RAG 是否真的有帮助 |

## 5. 哪些代码复用，哪些代码新增

### 5.1 建议复用

从 `fewshot_utils.py` 复用：

- `load_fewshot_split(split_root, k)`
- `load_feature_categories(path)`
- `normalize_label(value)`
- `row_to_feature_text(row, feature_categories)`

从 `llm_only_fewshot.py` 复用：

- `call_ollama(model, prompt, temperature, request_timeout)`
- `build_record_from_raw(idx, true_label, raw)`
- `metrics_from_records(records)`

注意：`build_prompt()` 不建议直接复用，因为 RAG prompt 需要额外插入 `Retrieved Knowledge` 区块。你可以参考它的结构，写一个新的 `build_rag_prompt()`。

### 5.2 建议新增

`rag_kb_builder.py` 负责：

- 读取特征类别和语义说明。
- 生成 `kb_docs.jsonl`。
- 调用 BGE embedding。
- 保存 `kb.index`、`kb_embeddings.npy`、`kb_metadata.json`。

`rag_retriever.py` 负责：

- 加载 `kb_docs.jsonl` 和 `kb.index`。
- 把 query encode 成向量。
- 返回 Top-K 文档和 score。

`llm_rag_raw_fewshot.py` 负责：

- 读取同一个 split。
- 对每条测试样本构造 raw query。
- 调用 retriever。
- 拼 RAG prompt。
- 调用 Ollama。
- 保存 JSONL、metrics、prompt 文件、retrieval log。

## 6. 最小可运行版本路线

### 阶段 1：只构建知识库，不跑 LLM

目标：先确认知识库文档是干净的。

检查点：

- `kb_docs.jsonl` 行数大于等于 215。
- 每条 doc 有 `doc_id`, `retrieval_text`, `prompt_text`。
- 不出现 `true_label`, `pred_label`, `class`, `test_source_indices`。
- 抽查 `SEND_SMS`, `READ_PHONE_STATE`, `BOOT_COMPLETED`, `system/bin/su` 都有对应文档。

骨架：

```python
def build_feature_docs(categories: dict[str, str], semantics: dict) -> list[dict]:
    docs = []
    for feature, category in categories.items():
        desc = semantics.get(feature, {}).get("description", "")
        docs.append({
            "doc_id": f"FEAT_{feature}",
            "doc_type": "feature_card",
            "feature": feature,
            "category": category,
            "retrieval_text": f"Feature {feature}. Category {category}. Meaning: {desc}",
            "prompt_text": f"[FEAT_{feature}] {feature}: {desc}",
            "source": "feature_categories_and_semantics",
            "leakage_risk": "no_label",
        })
    return docs
```

TODO：

- `doc_id` 里如果包含 `/`、`.` 这类字符，可以做一个简单清洗，方便日志阅读。
- 手写 10 到 20 条 behavior rule，先覆盖短信、设备标识、联网、开机自启、动态加载、root 命令。

### 阶段 2：构建 embedding 和 FAISS index

目标：检索链路能返回合理结果。

骨架：

```python
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

def build_index(docs: list[dict], model_name: str, index_path: str, emb_path: str):
    model = SentenceTransformer(model_name)
    texts = [doc["retrieval_text"] for doc in docs]
    embeddings = model.encode(texts, normalize_embeddings=True).astype("float32")

    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)

    np.save(emb_path, embeddings)
    faiss.write_index(index, index_path)
```

检查点：

- `index.ntotal == len(docs)`。
- `embeddings.dtype == float32`。
- 查询 `SEND_SMS READ_SMS INTERNET` 时，Top-K 中应该有短信相关 feature card 或 SMS rule。
- 查询 `mount system/bin/su` 时，应该命中 root/command 相关文档。

调试片段：

```python
def debug_search(query, embedder, index, docs, top_k=5):
    q = embedder.encode([query], normalize_embeddings=True).astype("float32")
    scores, ids = index.search(q, top_k)
    for score, doc_id in zip(scores[0], ids[0]):
        doc = docs[int(doc_id)]
        print(round(float(score), 4), doc["doc_id"], doc["prompt_text"])
```

### 阶段 3：只生成 prompt，不调用 LLM

目标：确认 RAG prompt 中没有标签泄漏，且结构清楚。

RAG prompt 建议结构：

```text
Classify the CURRENT TEST SAMPLE only.

Allowed labels:
- B means benign app.
- S means malware app.

Few-shot examples:
Example 0
Label: B
...

CURRENT TEST SAMPLE FEATURES:
Manifest Permission: SEND_SMS, READ_PHONE_STATE
API call signature: android.telephony.SmsManager

RETRIEVED KNOWLEDGE:
[1] [FEAT_SEND_SMS] SEND_SMS: 请求发送短信的系统权限。
[2] [RULE_SMS_ABUSE] 短信相关权限或 SmsManager 与联网、设备标识收集同时出现时...
[3] [FEAT_android.telephony.SmsManager] android.telephony.SmsManager: 调用短信管理接口。

Rules:
- Use the current test sample features as primary evidence.
- Use retrieved knowledge only as background explanation.
- Do not assume a label only because one retrieved document mentions suspicious behavior.
- Return exactly one JSON object.
```

骨架：

```python
def format_retrieved_docs(retrieved: list[dict]) -> str:
    lines = []
    for rank, item in enumerate(retrieved, start=1):
        doc = item["doc"]
        score = item["score"]
        lines.append(f"[{rank}] score={score:.4f} {doc['prompt_text']}")
    return "\n".join(lines)

def build_rag_prompt(train_df, test_row, feature_categories, retrieved_docs) -> str:
    # few-shot examples 的写法参考 llm_only_fewshot.build_prompt
    # 当前测试样本继续使用 row_to_feature_text，也就是 raw active features
    # 只额外添加 RETRIEVED KNOWLEDGE 区块
    ...
```

检查点：

- 当前测试样本区块不能出现 `Label:`。
- `Few-shot examples` 可以出现标签，因为它们来自训练侧示例，和 LLM-only baseline 一致。
- `RETRIEVED KNOWLEDGE` 里不出现测试样本真实标签。
- prompt 文件保存到 `results/rag_raw_llm/.../prompts/{idx}.txt`，便于肉眼检查。

### 阶段 4：跑 5 条样本烟雾测试

目标：端到端跑通，不追求指标。

建议命令形态：

```bash
python llm_rag_raw_fewshot.py --split-root data/processed/fewshot_seed42_test100 --k 5 --top-k 3 --max-test-samples 5
```

检查点：

- `rag_raw_top3.jsonl` 有 5 行。
- 每行有 `idx`, `true_label`, `pred_label`, `parse_ok`, `retrieved_doc_ids`, `raw`。
- `retrieval_logs.jsonl` 也有 5 行。
- `parse_ok_rate` 不为 0。
- 控制台能打印类似：`[1/5] true=B pred=B retrieved=FEAT_SEND_SMS,RULE_SMS_ABUSE,...`。

### 阶段 5：跑完整 200 条并比较

目标：和已有 LLM-only 实验公平比较。

建议输出：

```text
results/rag_raw_llm/seed42_test100/k5/
  rag_raw_top3.jsonl
  rag_raw_top3_metrics.json
  retrieval_logs.jsonl
  prompts/
```

对比时至少看：

- `accuracy_strict`
- `macro_f1`
- `recall_malware`
- `precision`
- `parse_ok_rate`
- `pred_S_ratio`

## 7. retrieval log 应该保存什么

RAG 是否有用，不能只看最终准确率，还要看“检索是否检到了有用内容”。建议每个测试样本保存一行：

```json
{
  "idx": 10863,
  "query_text": "Manifest Permission: SEND_SMS, READ_PHONE_STATE\nAPI call signature: android.telephony.SmsManager",
  "active_feature_count": 12,
  "top_k": 3,
  "retrieved": [
    {
      "rank": 1,
      "doc_id": "RULE_SMS_ABUSE",
      "doc_type": "behavior_rule",
      "score": 0.8123,
      "category": "SMS abuse"
    },
    {
      "rank": 2,
      "doc_id": "FEAT_SEND_SMS",
      "doc_type": "feature_card",
      "score": 0.7911,
      "category": "Manifest Permission"
    }
  ],
  "prompt_path": "results/rag_raw_llm/seed42_test100/k5/prompts/10863.txt"
}
```

可以在最终预测 JSONL 中额外保存 `true_label` 和 `correct`，因为那是评估结果。但知识库和检索器本身不要依赖这些字段。

后续分析可以做三件事：

- 正确样本中，检索到的知识是否和 evidence 对得上。
- 错误样本中，是否检索到了误导性规则。
- Top-K 分数是否过低。如果分数普遍很低，说明 query 或知识库文本写法需要调整。

## 8. 如何保证和 LLM-only 公平对比

公平对比的核心是：除了是否加入检索知识，其余条件尽量一致。

固定这些条件：

- 同一个 `split_root`：例如 `data/processed/fewshot_seed42_test100`。
- 同一个 `k`：例如 `k=5`。
- 同一个测试集：`test.csv`，不要重新抽样。
- 同一个模型：例如已有实验的 `gemma4:e4b`。
- 同一个 temperature：当前代码默认 `0.1`。
- 同一个 JSON 输出格式和解析逻辑。
- 同一个测试顺序和 `max_test_samples` 设置。
- 同一个 raw feature 表达函数：`row_to_feature_text`。

允许改变的只有：

- RAG 实验多一个 `RETRIEVED KNOWLEDGE` 区块。
- RAG 实验多一次检索步骤。
- 输出记录中多保存 retrieval 相关字段。

不要把语义化实验拿来直接和这个实验当唯一对比。更清楚的对比关系是：

| 对比 | 能回答的问题 |
|---|---|
| Raw LLM-only vs Raw + RAG | 检索知识是否提升原始特征输入下的推理 |
| Raw LLM-only vs Semantic LLM-only | 语义化输入本身是否有帮助 |
| Semantic LLM-only vs Raw + RAG | 两种“给模型补知识”的方式谁更适合当前任务 |

如果结果 RAG 没有提升，也不是失败。你可以分析检索日志：是检索质量差、知识库太浅、Top-K 太小、prompt 太长，还是 LLM 没有使用检索内容。

## 9. 数据泄漏检查清单

写完每个阶段后，用这份清单过一遍。

知识库检查：

- `kb_docs.jsonl` 不含 `true_label`、`pred_label`、`class`。
- `kb_docs.jsonl` 不含 `test_source_indices`。
- `kb_docs.jsonl` 不从 `results/` 读取内容。
- 如果有统计信息，确认只来自训练侧数据，并写入 `source` 字段。

query 检查：

- 当前测试样本 query 只由 active features 构成。
- 构造 query 时跳过 `class` 列。
- 不把 `row_idx` 对应的真实标签写进 prompt。

prompt 检查：

- `Few-shot examples` 有标签是允许的，因为 LLM-only 也有。
- `CURRENT TEST SAMPLE FEATURES` 没有标签。
- `RETRIEVED KNOWLEDGE` 没有“这个样本是恶意/良性”这种答案。

实验过程检查：

- 不根据测试集准确率反复改知识库规则。
- 如果调规则，最好只看 retrieval 相关性，不看最终预测对错；或者从训练侧另划一个很小的 dev set。
- 最终报告写清楚知识库来源和构建时间。

可以写一个简单断言：

```python
FORBIDDEN_KEYS = {"class", "true_label", "pred_label", "label", "answer"}

def assert_no_forbidden_fields(doc: dict) -> None:
    lowered_keys = {str(k).lower() for k in doc.keys()}
    bad = lowered_keys & FORBIDDEN_KEYS
    if bad:
        raise ValueError(f"Possible leakage fields in {doc.get('doc_id')}: {bad}")
```

注意：`label` 这个词在知识库中不一定永远非法，例如文档里可能有 `related_label`。但第一版为了稳，可以先不用任何 label 字段。

## 10. 常见坑和调试方法

### 10.1 检索结果全是无关内容

可能原因：

- query 太短，只写了 1 到 2 个特征。
- 知识库 `retrieval_text` 太中文，而你用的是英文 BGE。
- 没有 `normalize_embeddings=True`。
- 用 `IndexFlatIP` 但没有归一化，分数受向量长度影响。

调试方法：

- 打印 query。
- 打印 Top-5 的 `doc_id`, `score`, `retrieval_text`。
- 手动构造典型 query：`SEND_SMS READ_SMS INTERNET SmsManager`。
- 先让规则文档的 `retrieval_text` 显式包含关键特征名。

### 10.2 LLM 只照着规则判恶意，忽略当前样本

prompt 中加约束：

```text
Use the current test sample features as primary evidence.
Retrieved knowledge is background only.
Do not cite a rule unless at least one related active feature appears in the current test sample.
```

还可以在输出 JSON 中要求：

```json
"rules_cited": ["doc_id"],
"evidence": ["active_feature_from_current_sample"]
```

后续分析时检查 `evidence` 是否真的是当前样本 active features。

### 10.3 prompt 太长

RAG 的意义是少塞、精塞。第一版建议：

- `top_k=3`。
- 每条 `prompt_text` 控制在 1 到 2 句。
- 不要把 `raw_response`、`generation_prompt` 这种生成记录塞进知识库 prompt。
- few-shot 的 k 如果很大，先固定 k=5，不要和 Top-K 同时增大。

### 10.4 结果比 LLM-only 差

可能原因不止一个：

- 检索文档误导 LLM。
- 知识库规则过度偏向恶意，导致 `pred_S_ratio` 偏高。
- 检索 Top-K 没命中当前样本真正关键特征。
- RAG prompt 破坏了原来 LLM-only prompt 的格式约束。
- 语义知识太泛，只解释单个权限，没有解释组合行为。

分析顺序：

1. 看 `parse_ok_rate` 是否下降。
2. 看 `pred_S_ratio` 是否明显变化。
3. 抽 5 个 FP 和 5 个 FN，看 retrieval log。
4. 检查错误样本中被引用的 `rules_cited` 是否合理。

## 11. 多种实现路线比较

| 路线 | 做法 | 优点 | 缺点 | 推荐程度 |
|---|---|---|---|---|
| A. Feature card + 少量规则 | 215 个特征解释 + 20 到 40 条手写行为规则 | 最适合当前项目，容易调试，泄漏风险低 | 知识深度有限 | 推荐第一版 |
| B. 只用手写规则 | 只写 20 条规则做 KB | 最快 | 命中率低，很多 query 检不到细粒度特征 | 可做烟雾测试 |
| C. 加入 Android 官方权限/API 文档 | 手工整理官方文档片段 | 更权威 | 工作量大，chunk 清洗麻烦 | 第二版增强 |
| D. 训练样本相似案例检索 | 检索 labeled train examples | 可能提高分类 | 实验语义变了，不再是知识库 RAG | 不推荐混入本实验 |
| E. 混合检索 | feature card + rule + train case | 信息最多 | 很难判断提升来自哪里 | 放到后续扩展 |

当前大作业阶段，推荐路线 A。

## 12. 对 `项目方案.md` 的小修改建议

读完当前仓库后，有几处需要和实际代码对齐：

1. 旧方案里写的数据路径是 `data/Android Malware Dataset for Machine Learning/...`，但当前仓库实际是 `data/drebin-215-dataset-5560malware-9476-benign.csv` 和 `data/dataset-features-categories.csv`。
2. 旧方案里有 `rules/`、`src/`、notebook 目录设想，但当前代码已经采用根目录脚本方式。RAG 第一版建议继续用根目录脚本，减少重构。
3. 当前 `.gitignore` 忽略了 `md/`、`data/`、`results/`。这对本地实验没问题，但如果最后要把文档也提交到 GitHub，需要临时调整 `.gitignore` 或单独提交。
4. `项目方案.md` 中的 RAG 方案偏“全项目初始设计”，现在更适合把它收敛成“新增一个独立 RAG runner，与已有 LLM-only runner 做公平对比”。
5. 如果继续使用 `gemma4:e4b`，运行前用 `ollama list` 确认本机模型名存在。文档和结果元数据里要记录实际模型名。

## 13. 推荐学习资料

只看和你这个实验直接相关的部分：

- Sentence Transformers Quickstart：重点看如何加载模型、`encode` 文本、计算相似度。  
  https://sbert.net/docs/quickstart.html
- BAAI/bge-small-en-v1.5 model card：重点看如何用 `SentenceTransformer("BAAI/bge-small-en-v1.5")` 和模型维度信息。  
  https://huggingface.co/BAAI/bge-small-en-v1.5
- FAISS Getting started：重点看 `Index`、`add`、`search` 的概念。  
  https://github.com/facebookresearch/faiss/wiki/getting-started
- Ollama Python library：重点看 `client.chat(...)` 的调用方式。  
  https://github.com/ollama/ollama-python
- Ollama Structured Outputs：重点看 `format` 和结构化 JSON 输出。  
  https://docs.ollama.com/capabilities/structured-outputs

## 14. 第一版你应该按这个顺序做

1. 新建 `rag_kb_builder.py`，只生成 `kb_docs.jsonl`，不 embedding。
2. 抽查 10 条 doc，确认没有标签泄漏。
3. 在 `rag_kb_builder.py` 中加入 BGE embedding 和 FAISS index 保存。
4. 新建 `rag_retriever.py`，写 `retrieve(query, top_k)`。
5. 用 3 个手写 query 调试检索质量。
6. 新建 `llm_rag_raw_fewshot.py`，先实现 `--dry-run`，只保存 prompt。
7. 肉眼检查 2 到 3 个 prompt。
8. 跑 `--max-test-samples 5`。
9. 跑 `--max-test-samples 20`，看 parse 和检索日志。
10. 最后跑完整 200 条，并与 `llm_only_seed42_test100_k5` 做表格对比。

最重要的心法：第一版先让链路干净、可检查、可复现。不要一开始追求知识库很大，也不要一边看测试集准确率一边改规则。RAG 实验真正有价值的地方，是你不仅能说“准确率变了”，还能拿出 retrieval log 解释“为什么变了”。
