# RAG 知识库 JSON 字段说明

这份文档解释 `data/processed/rag_kb/` 目录下两个 JSON 产物的字段含义：

- `kb_docs.jsonl`
- `kb_metadata.json`

注意：真实的 `.json` / `.jsonl` 文件不能写注释，否则 Python 的 `json.loads()` 会解析失败。下面示例使用的是 `jsonc` 形式，只用于阅读说明。

## 1. `kb_docs.jsonl` 是什么

`kb_docs.jsonl` 是真正的 RAG 知识库文本文件。

它采用 JSONL 格式，也就是：

```text
一行 = 一个 JSON 对象 = 一条知识片段
```

当前知识库里主要有两类文档：

- `feature_card`：单个 Drebin 特征的解释。
- `behavior_rule`：手写的 Android 安全行为知识规则。

## 2. `feature_card` 示例字段说明

真实示例长这样：

```json
{"doc_id":"FEAT_SEND_SMS","doc_type":"feature_card","feature":"SEND_SMS","category":"Manifest Permission","retrieval_text":"...","prompt_text":"...","source":"...","leakage_risk":"no_sample_or_label"}
```

带注释解释如下：

```jsonc
{
  // 知识文档的唯一编号。
  // FEAT_ 表示这是一条“单个特征知识卡”。
  // 后续 retrieval log 会记录检索到了哪些 doc_id。
  "doc_id": "FEAT_SEND_SMS",

  // 文档类型。
  // feature_card 表示这条知识解释的是一个 Drebin 原始特征。
  "doc_type": "feature_card",

  // Drebin 原始特征名。
  // 这个字段必须保留原始写法，方便和测试样本中的 active feature 对照。
  "feature": "SEND_SMS",

  // 特征类别，来自 data/dataset-features-categories.csv。
  // 常见类别包括 Manifest Permission、API call signature、Intent、Commands signature。
  "category": "Manifest Permission",

  // 用于向量检索的文本。
  // 第二阶段会把这个字段送入 sentence-transformers / BGE 做 embedding。
  // 它通常偏英文或中英混合，是为了更适配 bge-small-en-v1.5。
  "retrieval_text": "Android Drebin feature SEND_SMS. Category: Manifest Permission. Meaning: 请求发送短信的系统权限. Related concepts: sms message operation.",

  // 检索命中后真正拼进 LLM prompt 的文本。
  // 它更偏中文说明，方便 LLM 生成中文解释，也方便你人工检查 prompt。
  "prompt_text": "[FEAT_SEND_SMS] SEND_SMS（Manifest Permission）：请求发送短信的系统权限。提示：短信相关能力需要结合联网、设备标识或后台触发行为一起分析。",

  // 这条知识的来源。
  // 用于报告中说明知识库如何构建，也方便以后排查某条知识来自哪里。
  "source": "dataset-features-categories.csv + feature_semantics_gemma.json",

  // 数据泄漏风险标记。
  // no_sample_or_label 表示这条知识不包含测试样本，也不包含 B/S 标签答案。
  "leakage_risk": "no_sample_or_label"
}
```

## 3. `behavior_rule` 示例字段说明

真实示例长这样：

```json
{"doc_id":"RULE_SMS_ABUSE","doc_type":"behavior_rule","category":"SMS abuse","retrieval_text":"...","prompt_text":"...","source":"manual_android_security_knowledge","leakage_risk":"no_sample_or_label"}
```

带注释解释如下：

```jsonc
{
  // 知识文档的唯一编号。
  // RULE_ 表示这是一条“行为规则”，不是单个特征解释。
  "doc_id": "RULE_SMS_ABUSE",

  // 文档类型。
  // behavior_rule 表示它解释的是一组特征组合可能对应的 Android 行为。
  "doc_type": "behavior_rule",

  // 行为规则类别。
  // 这个字段方便人工阅读 retrieval log，例如看到 SMS abuse 就知道是短信相关规则。
  "category": "SMS abuse",

  // 用于向量检索的文本。
  // 这里会故意包含相关 Drebin 特征名，如 SEND_SMS、READ_SMS、SmsManager。
  // 这样当前样本 query 中出现类似特征时，FAISS 更容易检索到这条规则。
  "retrieval_text": "Android security behavior pattern SMS_ABUSE. Related Drebin features and concepts: SEND_SMS READ_SMS RECEIVE_SMS WRITE_SMS SmsManager INTERNET READ_PHONE_STATE.",

  // 检索命中后拼进 LLM prompt 的解释文本。
  // 注意它只讲行为背景，不直接说“判为恶意”或“判为良性”。
  "prompt_text": "[RULE_SMS_ABUSE] 短信发送、读取、接收能力与联网或电话状态读取同时出现时，常用于短信扣费、验证码拦截、短信内容上传等行为分析。",

  // 这条规则的来源。
  // manual_android_security_knowledge 表示它来自脚本中手写的安全知识规则。
  "source": "manual_android_security_knowledge",

  // 数据泄漏风险标记。
  // 表示这条规则不是从测试样本或测试标签中构造出来的。
  "leakage_risk": "no_sample_or_label"
}
```

## 4. `kb_metadata.json` 字段说明

`kb_metadata.json` 不是知识库正文，而是这次知识库构建过程的记录。

带注释解释如下：

```jsonc
{
  // 本次知识库构建时间。
  "created_at": "2026-05-19T15:37:30",

  // 当前阶段名称。
  // stage_1_text_kb_only 表示第一阶段只生成文本知识库。
  "stage": "stage_1_text_kb_only",

  // 本次构建使用的特征类别文件路径。
  "feature_category_path": "data/dataset-features-categories.csv",

  // 本次构建使用的特征语义说明文件路径。
  "feature_semantics_path": "data/processed/paper_features/feature_semantics_gemma.json",

  // 生成的知识库正文文件路径。
  "output_docs_path": "data\\processed\\rag_kb\\kb_docs.jsonl",

  // 知识库文档总数。
  // 当前应为 215 条 feature_card + 16 条 behavior_rule = 231。
  "doc_count": 231,

  // 单个 Drebin 特征知识卡数量。
  // class 标签行已被跳过，所以正常是 215。
  "feature_card_count": 215,

  // 手写行为规则数量。
  "behavior_rule_count": 16,

  // 禁止出现在知识库文档顶层的字段名。
  // 这些字段往往和标签、预测结果、答案有关，容易造成数据泄漏。
  "forbidden_field_names": [
    "answer",
    "class",
    "decision",
    "label",
    "pred_label",
    "prediction",
    "result",
    "true_label",
    "verdict"
  ],

  // 是否已经包含 embedding 向量。
  // 第一阶段还没有做 embedding，所以是 false。
  "contains_embeddings": false,

  // 是否已经包含 FAISS 索引。
  // 第一阶段还没有建索引，所以是 false。
  "contains_faiss_index": false,

  // 是否包含 LLM 输出。
  // 知识库不应该混入模型预测结果，所以这里是 false。
  "contains_llm_outputs": false,

  // 是否包含测试样本。
  // RAG 知识库不能把测试样本放进去，所以这里是 false。
  "contains_test_samples": false,

  // 是否包含样本标签。
  // 知识库不能含有 B/S 标签答案，所以这里是 false。
  "contains_sample_labels": false
}
```

## 5. 为什么不直接给 JSON 文件加注释

标准 JSON 不支持注释，例如下面这种写法是非法的：

```jsonc
{
  // 这是非法 JSON 注释
  "doc_id": "FEAT_SEND_SMS"
}
```

如果把这种注释写进 `kb_docs.jsonl` 或 `kb_metadata.json`，后面的代码读取时会报错。

所以正确做法是：

- 真实数据文件保持纯 JSON / JSONL。
- 字段解释写在这份 Markdown 文档里。
- 如果需要带注释示例，就使用文档中的 `jsonc` 代码块，而不是改真实数据文件。
