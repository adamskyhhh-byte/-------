# Claude 对预测不准确问题生成的审查报告

> **生成时间**：2026-05-20
> **审查范围**：三组 LLM 实验（raw-LLM-only / semantic-LLM-only / raw-LLM+RAG）
> **测试样本规模**：每组 30 条（fewshot_seed42_test100/test.csv 的前 30 条）
> **被测模型**：Ollama 本地部署的 `gemma4:e4b`（参数 8.0B，量化 Q4_K_M）
> **温度**：0.1
> **结论摘要**：**用户的观察完全属实**——三组实验都存在显著的"过度判 S"偏差，
> recall_malware 全部为 1.0（S 一个不漏），但 recall_benign 介于 0.0 - 0.69 之间，
> 大量 B 样本被误判为 S。**根因不在 prompt 的限定词，而在数据/检索/上下文三个工程层面**。

---

## 一、复现实验结果

我把三个实验脚本各跑了 30 条样本，结果保存在 `results/claude_audit_30/` 下（与已有结果隔离）。**测试集前 30 条样本的真实标签分布是 16 B + 14 S**。

### 1.1 混淆矩阵汇总

| 实验 | 真 B / S | TP (S→S) | TN (B→B) | FP (B→S) | FN (S→B) | None | accuracy | recall_S | recall_B | pred_S_ratio |
|------|---------|----------|----------|----------|----------|------|----------|----------|----------|--------------|
| **raw-LLM-only** | 16 / 14 | 14 | 11 | **5** | 0 | 0 | 0.833 | 1.000 | 0.688 | 0.633 |
| **semantic-LLM-only** | 16 / 14 | 14 | **0** | **13** | 0 | **3** | 0.467 | 1.000 | **0.000** | **0.900** |
| **raw-LLM+RAG (top3)** | 16 / 14 | 14 | 6 | **10** | 0 | 0 | 0.667 | 1.000 | 0.375 | 0.800 |

观察：

1. **三组实验都把 S 全部抓到（recall_S = 1.0），但代价是大量误报 B → S**。
2. **semantic 模式最严重**：16 个真实 B 没有一个被识别出来，全部被判为 S 或解析失败。
3. **RAG 没有"挽救" raw 模式**——RAG 反而比 raw-LLM-only 多了 5 个 FP（从 5 到 10）。
4. semantic 出现 3 条 `pred=None`，是因为模型超时（idx=18）或返回的 evidence 字段是**模型幻觉出来的伪特征名**（idx=24/25，详见 §3.3）。

### 1.2 与既有结果对照（用户冒烟测试同步验证）

我也读了仓库里已有的 `results/feature_expr_llm/seed42_test100/k5/raw_full_metrics.json`（36 条样本）和
`results/codex_audit/rag_raw_llm/seed42_test100/k1/`（30 条样本 + k=1）的指标：

| 历史结果 | total | pred_S_ratio | recall_S | recall_B |
|---------|-------|--------------|----------|----------|
| raw-LLM-only (36 条) | 36 | **0.889** | 1.000 | 0.200 |
| semantic-LLM-only (10 条) | 10 | **0.900** | 0.800 | 0.000 |
| RAG-top3 k=1 (30 条) | 30 | **1.000** | 1.000 | 0.000 |

**结论**：用户冒烟测试时观察到的"三个模型大部分时候都在预测 S，很少预测 B"在我的复现实验中完全成立。raw 模式偶有恢复，但 semantic 和 RAG 模式持续严重偏 S。

---

## 二、数据层根因（最重要、也最反直觉）

### 2.1 Drebin few-shot 训练样本里 **B 反而比 S 特征更多**

我统计了 `data/processed/fewshot_seed42_test100/k5/train.csv`（10 条 few-shot）的活跃特征数：

| 类别 | 数量 | 平均 active_n | 中位数 | min – max |
|------|------|---------------|--------|-----------|
| B（良性）| 5 | **53.4** | 56 | 30 – 73 |
| S（恶意）| 5 | **31.6** | 26 | 11 – 67 |

测试集 200 条的整体统计也呈同一规律（B 平均 34.5 > S 平均 26.4）。

> **反直觉点**：通用 LLM 的常识是"权限越多越可疑"，但 Drebin 数据集里**良性 App 的激活特征数反而显著高于恶意 App**。few-shot 仅 10 条示例，远不足以让 LLM 自己推翻这种先验。

### 2.2 FP 与 TN 的真实差异：**权限多 + API 少 → LLM 错判 S**

我把 raw-LLM-only 中"B → S 误报"（FP, n=5）和"B → B 正确"（TN, n=11）的特征类别均值做了对比：

| 特征类别 | FP 均值 | TN 均值 | 差异（FP − TN） |
|----------|---------|---------|-----------------|
| Manifest Permission | **14.8** | 8.0 | **+6.8** |
| API call signature | 21.4 | **30.5** | **−9.1** |
| Intent | 2.4 | 1.1 | +1.3 |
| Commands signature | 0.8 | 1.0 | −0.2 |

**核心发现**：**LLM 误判为 S 的 B 样本，恰恰是"权限请求多但 API 调用少"的样本**。
这完全符合"权限 = 隐私敏感 = 可疑"的人类直觉，但与 Drebin 数据集的真实分布相反——
有些良性工具型 App 会申请大量权限以备用，但运行时 API 调用并不多。

### 2.3 反例案例分析：idx=19（B → S 误报）

active features（仅 10 个）：
```
INTERNET, Binder, RECORD_AUDIO, IBinder, android.os.IBinder, VIBRATE,
HttpPost.init, HttpUriRequest, onBind, WRITE_EXTERNAL_STORAGE
```

LLM 解释：
> "The combination of network activity (HttpPost.init, HttpUriRequest) and system interaction
> (Binder, IBinder, onBind) suggests potential malicious communication or data exfiltration,
> even with limited permissions."

**问题**：HttpPost + Binder + onBind 是几乎所有需要后台通信的 App（音乐、闹钟、笔记、阅读）都会用的标准组合，LLM 把它视为 "data exfiltration" 完全是常识偏见。

---

## 三、Prompt / 上下文层根因

### 3.1 三个实验 prompt 的实际 token 数

我对三个实验第 0 个测试样本对应的 prompt 文件做了字节数测量并按"4 字节≈1 token"的英文粗估算（中文密集时甚至更不利）：

| 实验 | prompt 字节数 | 估算 tokens | 是否超 Ollama 默认 `context_length=4096` |
|------|---------------|-------------|-----------------------------------------|
| raw-LLM-only | 11 687 | ~2 921 | ❌ 安全 |
| **semantic-LLM-only** | **36 403** | **~9 100** | ✅ **超 2.2 倍** |
| raw-LLM+RAG | 12 771 | ~3 192 | ❌ 安全 |

> 通过 `curl http://localhost:11434/api/ps` 也可以确认当前模型加载时 `context_length = 4096`。

### 3.2 Semantic prompt 被默默截断 = strict_parse_ok 全部失败

semantic 模式：

- `strict_parse_ok_rate = 0/30 = 0.0`（30 条全部不严格 JSON）
- 但 `parse_ok_rate = 27/30`（27 条靠 `recovered_json` 兜底）

也就是说 **30 条 semantic 输出里没有一条按 prompt 要求只输出 `pred_label/evidence/explanation/confidence` 这四个键**。
对比 raw 和 RAG 模式 strict_parse_ok 都接近 100%（30/30 与 29/30）。
这只能解释为 prompt 被截断后，LLM 失去了对输出格式约定的"末尾视野"。

### 3.3 Semantic 模式的"幻觉证据"是截断的直接产物

semantic 模式 idx=24（true=B, pred=None）的 raw 输出 evidence：

```json
"evidence": [
  "android.content.Context", "android.content.Intent",
  "android.widget.Toast", "android.widget.TextView",
  "android.view.View", "android.view.ViewGroup",
  "android.widget.LinearLayout", "android.widget.Button",
  "android.widget.EditText", "android.widget.RelativeLayout",
  "android.widget.ImageView", "android.graphics.Bitmap",
  "android.graphics.drawable.Drawable", ...
]
```

**这些"特征"在 Drebin 215 维特征空间里根本不存在**——Drebin 的特征是 `transact / onServiceConnected / Ljavax.crypto.Cipher / SEND_SMS` 这种命名。LLM 看不见完整的 prompt 后段，开始用"常识里 Android App 应该有的组件"自行编造。这是上下文截断的硬证据。

### 3.4 Raw / RAG prompt 中的"善意提示"反而起了反作用

raw 与 RAG 的 prompt 包含：

```
- B means benign / normal / non-malicious.
- S means malware / malicious / suspicious.
```

这里把 `S` 与 **"suspicious"**（可疑）等价了——但 "suspicious" 是一个比 "malicious"
门槛低得多的词，LLM 内部一旦判断"看起来有点像可疑模式"就有充分理由输出 S。
这相当于在 prompt 里默认把判 S 的阈值往下拉。

---

## 四、RAG 检索层根因

### 4.1 RAG 几乎从来不命中"判别规则"

`data/processed/rag_kb/kb_docs.jsonl` 一共有 231 条文档：

| 文档类型 | 数量 | 占比 |
|---------|------|------|
| `feature_card`（特征语义说明） | 215 | 93.1% |
| `behavior_rule`（行为判别规则） | 16 | 6.9% |

我对 30 条样本 × top-3 = **90 次检索命中** 的类型做了统计：

| 命中类型 | 次数 | 占比 |
|---------|------|------|
| `feature_card` | 89 | **98.9%** |
| `behavior_rule` | 1 | **1.1%**（只有 idx=4 命中了 `RULE_SMS_ABUSE`） |

**结论**：本应承担"提示规则证据"作用的 16 条 RULE_ 文档，在 90 次检索里只被命中过 1 次。
RAG 的判别功能近乎完全失效。

### 4.2 为什么 RULE_ 文档检索不到？

| 因素 | 描述 |
|------|------|
| **数量不平衡** | feature_card 与 behavior_rule 是 215:16（≈13:1），先验上就更容易被检索器选中。 |
| **检索 query 的构造方式** | `build_query_text()` 用的是 active features 的英文原名拼接，自然与"英文特征名"对应的 feature_card 在 embedding 空间里距离更近。 |
| **跨语言对齐弱** | feature_card 是"英文特征名 + 中文短描述"，behavior_rule 是"英文 doc_id + 中文段落"；query 是"英文特征名 list"，BGE-small-en 对纯中文 rule 的语义对齐弱。 |
| **rule 描述过于抽象** | RULE_ 文本写的是宏观行为模式（"短信滥用"、"开机持久化"），不会与具体特征名 token 匹配。 |

### 4.3 检索到的 feature_card 反而强化"偏 S"倾向

最常被检索到的 feature_card 是：

- `FEAT_LJAVAX_CRYPTO_CIPHER`（加密 Cipher）
- `FEAT_LJAVA_LANG_CLASS_GETCLASSES`（反射类信息）
- `FEAT_LJAVA_LANG_OBJECT_GETCLASS`（反射获取对象类型）
- `FEAT_ANDROID_OS_IBINDER`（Binder 接口）

这些都是任何中型 App（含正规应用）都会用的标准库 API。但是当 LLM 看到 prompt 里同时列出这些
"加密 / 反射 / Binder" 的中文说明，再叠加 §2.2 中的"权限多 → 可疑"常识偏见，会进一步把样本拉向 S。
也就是说 **RAG 注入的"中性技术语义"在没有"良性比照"的情况下会被 LLM 解读为风险信号**。

### 4.4 RAG 与 raw-LLM-only 差距的解释

- raw-LLM-only：accuracy=0.833, FP=5
- raw-LLM+RAG：accuracy=0.667, FP=10

RAG 反而劣化了 5 个 FP。原因正是 §4.3：检索注入的全是"敏感技术 API 的中性中文说明"，几乎没有任何 "B 类规则"，
在 prompt 里相当于给 LLM 增加了"可疑印象"而没有提供"反驳证据"。

---

## 五、模型层根因

### 5.1 Gemma 4 E4B 的安全保守倾向

通用 LLM 在面对"判断一个 Android App 是否恶意"这类安全判断时，普遍存在**Type-I 错误成本远小于 Type-II 错误成本**的内置偏好——
"宁可错杀 B 也别漏放 S"。这点从三组实验 recall_malware 全 = 1.0、recall_benign 普遍 < 0.7 可以清晰看出。
这不是 prompt 设计问题，而是基础模型的预训练倾向。

### 5.2 同一组特征 LLM 给出截然相反的解释

- idx=0（true=B, pred=B, raw 模式）解释："standard Android application functionality... **common for legitimate apps**"
- idx=9（true=B, pred=S, raw 模式）解释："The sample exhibits **multiple indicators of malicious behavior**, including network communication..."

但 idx=0 与 idx=9 的特征覆盖度高度重叠（都有 HttpGet/HttpPost、Binder、TelephonyManager、ClassLoader、INTERNET、READ_PHONE_STATE）。
**同一组特征 LLM 可以解释成 "common for legitimate apps" 也可以解释成 "indicators of malicious behavior"**——
这说明 LLM 没有稳定的判别函数，而是在每条样本上做"随机 framing"。
而且 LLM 给 FP 的 confidence 仍然是 0.95，所以 confidence 本身完全不可作为校准信号。

---

## 六、整体结论：四条根因的叠加

| 层级 | 根因 | 量化证据 |
|------|------|---------|
| **数据** | Drebin few-shot 中 B 平均特征数（53.4）≫ S（31.6），与 LLM "权限多即可疑" 常识反向 | §2.1 §2.2 |
| **Prompt / 上下文** | semantic prompt ≈9100 tokens 超 Ollama 默认 4096 上下文窗口 → 截断 → 输出退化为幻觉 | §3.1 §3.2 §3.3 |
| **RAG 检索** | 215 feature_card vs 16 behavior_rule 严重不平衡，BGE-small 跨语言对齐弱 → 98.9% 检索命中都是"中性特征语义说明"而非判别规则 | §4.1 §4.2 §4.3 |
| **基础模型** | Gemma 通用安全保守偏见 + 缺乏稳定判别函数（同样特征不同样本给反向解释） | §5.1 §5.2 |

semantic 模式叠加了上下文截断 → 比 raw 还差；RAG 模式叠加了"中性特征语义注入" → 比 raw 也差。
所以三个实验里 raw-LLM-only 反而是最稳健的（accuracy=0.833），但**这并非因为它好，而是因为 semantic / RAG 引入了更多噪声**。

---

## 七、修改建议（不依赖引导性限定词）

> **遵循用户要求**：以下所有修改均**不通过在 prompt 里加 "请尽量预测 B" / "不要默认 S" 这类引导词**完成。
> 这些方法都是工程层面的对应原因修复，不影响实验结论的可信度。

### 7.1 优先级 P0（必改、最便宜、效果最大）

#### 7.1.1 把 Ollama 的 `num_ctx` 拉到 ≥12288，根治 semantic 截断

在 `llm_only_fewshot.py::call_ollama()` 的 `options` 字段里增加 `num_ctx`：

```python
options={"temperature": temperature, "num_ctx": 12288},
```

这是 Ollama Python SDK 公开支持的参数，作用是请求模型实例的上下文窗口（在硬件许可时被分配更大的 KV cache），
不影响 prompt 内容本身。这一项**应该首先单独验证**：semantic 模式在 num_ctx=12288 下重跑 30 条，
观察 strict_parse_ok_rate 是否能恢复到接近 1.0、evidence 是否会回到真实 Drebin 特征名。
如果 GPU 显存不够，可以退化到 8192，仍优于默认的 4096。

#### 7.1.2 把 prompt 里 `S = "suspicious"` 这个不必要的语义放宽词去掉

当前 prompt：

```
- S means malware / malicious / suspicious.
```

把它精简成只保留 `S means malware`。这不是"在 prompt 里加引导词"，而是**移除原本就存在的、把判 S 阈值降低的过度宽松定义**。
同时 `If your internal answer is malware, malicious, or suspicious` 也对应改成 `If your internal answer is malware`。
这一条是去噪而不是引导，**不改变实验对照的公平性**——三组实验都用同一份新 prompt 即可。

### 7.2 优先级 P1（中等成本，针对 RAG 失效问题）

#### 7.2.1 RAG 检索结构改成"分桶检索"（hybrid retrieval）

修改 `rag_retriever.py`，让单次 query 分别从 `feature_card` 桶和 `behavior_rule` 桶里
各取 top-k'（例如各取 2 条 → 总共 4 条），而不是混在一起按相似度全局排序。
这样保证 16 条规则在每次推理里都有机会被命中，而不是被 215 条 feature_card 淹没。

伪代码示意：

```python
def retrieve(self, query_text, top_k):
    hits_feat = self._retrieve_in_subset(query_text, subset='feature_card', top_k=2)
    hits_rule = self._retrieve_in_subset(query_text, subset='behavior_rule', top_k=2)
    return hits_feat + hits_rule
```

#### 7.2.2 把 behavior_rule 文本英化或加英文版本

当前 RULE_ 文档是纯中文，但 query 是英文 active feature 名拼接。
可以离线给每条 RULE 加一个英文摘要字段（不影响 prompt 里展示的中文版），让 embedding 模型的对齐更准。
这不属于"提示词引导"，属于检索语料工程。

#### 7.2.3 引入良性规则（Benign Rule）

当前 16 条 behavior_rule 全部是"恶意行为模式"。可以补几条良性模式规则（例如"仅请求 INTERNET + ACCESS_NETWORK_STATE 是常见基础联网应用"、
"标准 IPC 模式（bindService + onServiceConnected）多数 App 都会用"），让 LLM 在判别时有"反向证据"参考。
这是给数据/规则库做平衡，不是 prompt 引导词。

### 7.3 优先级 P2（结构性提升，可选）

#### 7.3.1 把 K-shot 采样改成 Core-set / 分层采样

现在的 random K-shot 由于样本随机性，可能恰好选到"特征数 B≫S"的 few-shot 组合（§2.1）。
可以用 KMeans 聚类后再选代表样本（项目方案 §七.4 已设计 `coreset_kshot` 函数），
减少 few-shot 样本本身的偏差。这是采样策略改进，不修改 prompt。

#### 7.3.2 评估指标里强制报告 `pred_S_ratio` 和 confusion matrix，不要只看 accuracy

`metrics_from_records()` 已经在记 `pred_B_count / pred_S_count / pred_S_ratio`，
但实验对照表里应当**显式地把这一列放在 accuracy 旁边**，防止"accuracy 看起来还行其实是因为 S 全猜对"的虚高问题。
对类别不平衡场景，macro_f1 比 accuracy 更值得作为主指标。

#### 7.3.3 给 LLM 调用换一个不开 `format='json'` 的"思考"通道

`format='json'` 强制 JSON 输出，但同时**会让 Gemma 在生成时缩减推理链**（模型已知必须直接吐 JSON）。
可以增加一个 `think-then-json` 两段式调用：第一次让 LLM 自由文本分析，第二次让 LLM 把分析转 JSON。
这是工程结构上的调整，与 prompt 引导词无关。

### 7.4 不建议的改动（明确避免）

- ❌ 在 prompt 里加 "请注意大部分样本是良性"。这会污染对照实验。
- ❌ 在 prompt 里加 "如果你不确定请预测 B"。这是把人类先验注入模型，使报告不可信。
- ❌ 后处理直接把 confidence < 0.7 的 S 强制改成 B。这是数据造假。
- ❌ 修改测试集让真实分布变化。这是数据泄漏。

---

## 八、推荐的验证流程（不修改代码即可启动）

1. **先做最便宜的实验**：仅修改 `call_ollama()` 加上 `num_ctx=12288`，semantic-LLM-only 重跑 30 条，
   核对 `strict_parse_ok_rate` 是否从 0.0 提升到 ≥0.9；如果可以，semantic 实验本身的可信度首先恢复。
2. **再做 RAG 分桶检索改造**：按 §7.2.1 改 `retrieve()`，RAG-LLM 重跑 30 条，
   核对 `pred_S_ratio` 是否从 0.8 下降到接近真实分布 0.467。
3. **最后做 prompt 去歧义**：删除 `suspicious` 关键词，三组重跑，
   观察 recall_benign 是否整体抬升、recall_malware 是否仍 ≥0.9。

完成上述三步后再考虑 P2 的结构性改动。

---

## 附录 A：复现命令

```bash
# 1. raw-LLM-only（30 条）
python llm_feature_expr_fewshot.py \
  --split-root data/processed/fewshot_seed42_test100 \
  --k 5 --feature-expr raw --feature-subset full \
  --feature-category-path data/dataset-features-categories.csv \
  --output-root results/claude_audit_30/feature_expr_llm \
  --model gemma4:e4b --temperature 0.1 --request-timeout 120.0 \
  --max-test-samples 30

# 2. semantic-LLM-only（30 条）
python llm_feature_expr_fewshot.py \
  --split-root data/processed/fewshot_seed42_test100 \
  --k 5 --feature-expr semantic --feature-subset full \
  --feature-semantics data/processed/paper_features/feature_semantics_gemma.json \
  --feature-category-path data/dataset-features-categories.csv \
  --output-root results/claude_audit_30/feature_expr_llm \
  --model gemma4:e4b --temperature 0.1 --request-timeout 180.0 \
  --max-test-samples 30

# 3. raw-LLM+RAG（30 条）
python llm_rag_raw_fewshot.py \
  --split-root data/processed/fewshot_seed42_test100 \
  --k 5 --top-k 3 \
  --kb-dir data/processed/rag_kb \
  --feature-category-path data/dataset-features-categories.csv \
  --embedding-model BAAI/bge-small-en-v1.5 \
  --model gemma4:e4b --temperature 0.1 --request-timeout 180.0 \
  --output-root results/claude_audit_30/rag_raw_llm \
  --max-test-samples 30 --local-files-only
```

---

## 附录 B：本报告所引用的数据文件

| 文件 | 用途 |
|------|------|
| `results/claude_audit_30/feature_expr_llm/seed42_test100/k5/raw_full.jsonl` | raw 模式 30 条预测明细 |
| `results/claude_audit_30/feature_expr_llm/seed42_test100/k5/semantic_full.jsonl` | semantic 模式 30 条预测明细 |
| `results/claude_audit_30/rag_raw_llm/seed42_test100/k5/rag_raw_top3.jsonl` | RAG 30 条预测明细 |
| `results/claude_audit_30/rag_raw_llm/seed42_test100/k5/retrieval_logs.jsonl` | RAG 检索日志（含 doc_id 与分数） |
| `data/processed/fewshot_seed42_test100/k5/train.csv` | k=5 few-shot 训练集（10 条）|
| `data/processed/fewshot_seed42_test100/test.csv` | 测试集（200 条）|
| `data/processed/rag_kb/kb_docs.jsonl` | RAG 知识库（231 条文档）|

