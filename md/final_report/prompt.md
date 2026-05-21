# LLM Prompt 模板 — Drebin 少样本恶意软件分类

本文档汇总本次实验使用的全部 LLM Prompt 模板。所有 prompt 共用同一套 **Allowed labels** 与 **Important mapping rule**，保证三种特征表达（raw / semantic-risky-old / semantic-neutral-fixed）和 RAG 增强版本之间公平可比。

实验设计的关键原则：**不引入任何方向性引导词**（如"如果不确定就判 B"、"不要轻易判 S"等）。所有调整都通过特征表达方式、训练池统计、规则库结构来体现，不通过 prompt 中的人工偏置。

---

## 1. raw-LLM-only Prompt 模板

```
Classify the CURRENT TEST SAMPLE only.

Task:
- Binary classification for an Android app.
- Use only the active Drebin features shown in this prompt.

Allowed labels:
- B means benign app.
- S means malware app.

Important mapping rule:
- If your answer is benign app, output "pred_label":"B".
- If your answer is malware app, output "pred_label":"S".

Experiment settings:
- feature_expr: raw
- feature_subset: full
- k_semantics: per_class

Few-shot examples:
Example 0
Label: <B 或 S>
<按 Drebin category 分组列出激活的原始特征名>

Example 1
Label: <B 或 S>
<同上>

... (共 2K 个示例：每类 K 个)

CURRENT TEST SAMPLE FEATURES:
<按 Drebin category 分组列出当前测试样本的激活特征>

Return EXACTLY ONE JSON object and nothing else.
The first character must be { and the last character must be }.
Do not use Markdown code fences.
Use exactly these keys: pred_label, evidence, explanation, confidence.
pred_label must be exactly one of: "B", "S".
evidence must be a JSON array of active feature names from the current test sample.
confidence must be a number from 0 to 1.

Valid output example for benign:
{"pred_label":"B","evidence":["FEATURE_NAME"],"explanation":"short reason","confidence":0.50}
Valid output example for malware:
{"pred_label":"S","evidence":["FEATURE_NAME"],"explanation":"short reason","confidence":0.80}
```

特征渲染（per active feature）：

```
API call signature: Binder, ClassLoader, HttpGet.init, ...
Commands signature: chmod, mount
Intent: android.intent.action.BOOT_COMPLETED
Manifest Permission: ACCESS_NETWORK_STATE, INTERNET, ...
```

---

## 2. semantic-risky-old Prompt 模板（失败对照）

与 raw 模板**仅 `CURRENT TEST SAMPLE FEATURES` 与 few-shot 部分不同**，其余完全一致。
特征改用 `feature_semantics_gemma.json` 的旧风险词描述：

```
API call signature:
- Ljavax.crypto.Cipher: 加密/解密 API，可用于加密敏感数据或网络通信
- TelephonyManager.getDeviceId: 获取设备唯一标识符，可被用于追踪
- DexClassLoader: 动态加载和执行外部代码，可隐藏真实逻辑
...

Manifest Permission:
- SEND_SMS: 发送短信的危险权限，可用于扣费短信
...
```

此模板**保留**作为失败对照——它的描述词把许多在 Drebin 中实际偏 B 的常见特征（Cipher、Binder、反射、mount）渲染成"危险倾向"，会让 LLM 系统性偏向 S。

---

## 3. semantic-neutral-fixed Prompt 模板（修复版）

与 raw 模板仅 `CURRENT TEST SAMPLE FEATURES` 与 few-shot 部分不同。每个 active feature 以 4 字段结构化 block 渲染：

```
Active features (each block is structural; train_stat is training-pool context,
not a direct label):
- feature: Ljavax.crypto.Cipher
  category: API call signature
  meaning: Java standard library class providing symmetric and asymmetric cipher
           operations such as encryption and decryption.
  train_stat: leans_B, P(feature|B)=0.5360, P(feature|S)=0.2640
- feature: SEND_SMS
  category: Manifest Permission
  meaning: Android permission allowing apps to send text messages programmatically.
  train_stat: leans_S, P(feature|B)=0.0590, P(feature|S)=0.5390
...
```

`meaning` 字段由 `gemma4:e4b` 用 16 词禁用清单 + ≤25 词约束的中性 prompt 离线重新生成；不允许含 `malicious / malware / suspicious / harmful / dangerous / risky / attack / exploit / exfiltrate / steal / hidden / secret / payload / victim / abuse / hack / trojan` 任一词。

`train_stat` 由 `feature_stats.py` 在**剔除 test 样本的训练池**上计算：

- `support_B / support_S` 训练池中 B / S 类有该特征的样本数
- `p_feature_given_B / p_feature_given_S` 类条件概率（Laplace 平滑）
- `log_odds_S_vs_B = log(p_S/(1-p_S)) − log(p_B/(1-p_B))`
- `stat_direction = leans_B / leans_S / weak_or_mixed`（阈值 ±0.5）

> **报告说明**：`P(feature|B/S)` 与朴素贝叶斯类条件概率数学等价。它们来自训练池的客观分布，不引入主观倾向。验证方式是检查 `feature_stats_metadata.json` 的 `test_indices_excluded` 与 `test_metadata.json` 的 `test_source_indices` 严格相等。

---

## 4. raw-LLM + RAG (bucketed) Prompt 模板

与 raw 模板共用 Allowed labels / mapping rule，但增加 **General reasoning hint**、**Knowledge usage policy**、两块独立检索区。

```
Classify the CURRENT TEST SAMPLE only.

Allowed labels:
- B means benign app.
- S means malware app.

Important mapping rule:
- If your answer is benign app, output "pred_label":"B".
- If your answer is malware app, output "pred_label":"S".

General reasoning hint:
- Consider the combination of active features rather than any single feature.
- Use the current test sample features as primary evidence.
- Retrieved knowledge is background only.
- Do not cite or rely on a retrieved rule unless at least one related active
  feature appears below.

Knowledge usage policy:
- Retrieved behavior rules may include a related_label field. This is a soft
  prior from the rule base, not a hard rule.
- Do not copy related_label as the final answer unless the current active
  features and few-shot examples support it.
- When a retrieved rule matches current active features, cite its rule id in
  explanation. If no retrieved rule matches, say so briefly in explanation.

Few-shot examples:
Example 0
Label: <B 或 S>
<按 Drebin category 分组列出激活原始特征名>
...

CURRENT TEST SAMPLE FEATURES:
<同上>

RETRIEVED FEATURE KNOWLEDGE:
[1] score=0.78 [FEAT_INTERNET] Feature: INTERNET. Category: Manifest Permission.
    Neutral meaning: ... Training-pool statistic: this feature weak_or_mixed; ...
    Related label: context-dependent (soft prior only, not a direct answer).
[2] score=0.74 [FEAT_LJAVAX_CRYPTO_CIPHER] ... leans_B; P(feature|B)=0.5360,
    P(feature|S)=0.2640. ... Related label: B (soft prior only, ...).

RETRIEVED BEHAVIOR RULES:
[1] score=0.69 [RULE_SMS_ABUSE] Behavior rule: SMS abuse. Related features:
    SEND_SMS READ_SMS RECEIVE_SMS WRITE_SMS SmsManager INTERNET READ_PHONE_STATE.
    Explanation: ... Related label: S (soft prior only, not a direct answer).
[2] score=0.65 [RULE_DEVICE_IDENTIFIER_COLLECTION] ...
[3] score=0.61 [RULE_STANDARD_IPC] ... Related label: context-dependent ...

Return EXACTLY ONE JSON object and nothing else.
... (输出格式同 raw)
```

### 4.1 双 query 构造

`build_query_text(test_row)`（用于检索 `feature_card` 桶）：

```
Android app active Drebin features for feature card retrieval.
API call signature: Binder, ClassLoader, ...
Manifest Permission: INTERNET, READ_PHONE_STATE, ...
Raw active feature names: Binder, ClassLoader, INTERNET, ...
Retrieve feature meanings (feature_card).
```

`build_rule_query_text(test_row)`（用于检索 `behavior_rule` 桶）：

```
Android malware behavior rule query.
Active groups:
- sms: 0 active
- telephony_identifier: 2 active (READ_PHONE_STATE, TelephonyManager.getDeviceId)
- root_system_command: 0 active
- package_install: 0 active
- dynamic_loading: 1 active (DexClassLoader)
- network: 2 active (INTERNET, HttpPost.init)
- privacy_sensor: 0 active
- persistence_overlay: 0 active
Important combinations: present: telephony_identifier, dynamic_loading, network;
  absent: sms, root_system_command, package_install, privacy_sensor, persistence_overlay.
Retrieve behavior rules including benign and context-dependent rules.
```

### 4.2 检索后处理

- `feature_card` 检索 top-k = **2**（独立 FAISS 子索引 `kb_feature.index`）
- `behavior_rule` 检索 top-k = **3**（独立 FAISS 子索引 `kb_rule.index`）
- 不允许在全量索引上做"超采样+掩码筛选"的回退路径

---

## 5. 中性 semantic 离线生成 Prompt（仅用于一次性产 `feature_semantics_neutral_stats.json`）

```
You are documenting Android API and permission features for a research dataset.

TASK
Write a single short sentence describing what the following Android feature
literally is or does.

CONSTRAINTS
- Only describe the literal Android documentation meaning of the feature.
- Do not say whether the feature is malicious, suspicious, dangerous, harmful,
  risky, or benign.
- Do not say what kind of app uses it, what the attacker can do with it, or
  what the user should worry about.
- Do not use any of these words: malicious, malware, suspicious, harmful,
  dangerous, risky, attack, exploit, exfiltrate, steal, hidden, secret,
  payload, victim, abuse, hack.
- Do not write more than 25 words.
- Output strict JSON only, no surrounding text.

INPUT
feature_name: {feature}
feature_category: {category}

OUTPUT JSON SCHEMA
{"feature":"{feature}","meaning":"<one neutral sentence, <=25 words>"}
```

生成后强制校验：

1. `meaning` 字段不含禁用词清单中任一词
2. 单词数 ≤ 25
3. 失败的特征写入 `generation_failures.json`，禁止用风险词模板兜底

---

## 6. JSON 输出 schema

所有 LLM 实验共用：

```json
{
  "pred_label": "B|S",
  "evidence": ["<active feature name>", ...],
  "explanation": "<short reason>",
  "confidence": 0.0-1.0
}
```

解析层（`llm_only_fewshot.infer_label_from_text`）允许 `pred_label` 是 `B/S/benign/malware/malicious/adware/ransomware/spyware/trojan` 这些**严格恶意类术语**；`suspicious/privacy invasive` 已从识别集中**完全移除**，避免低阈值映射放大模型偏 S 的倾向。

---

## 7. 公平对比承诺

| 维度 | 共同 |
|---|---|
| split | `data/processed/fewshot_seed42_test100`（B=100, S=100，每类 100 条 test，本次取前 50 条） |
| seed | 42 |
| K（每类示例数） | 1 / 3 / 5（三组独立实验） |
| 模型 | `gemma4:e4b`（Ollama 本地） |
| temperature | 0.1 |
| num_ctx | 12288 |
| JSON schema | 同上 |
| 解析层 | 同一份 `infer_label_from_text` |
