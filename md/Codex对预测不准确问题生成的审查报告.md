# Codex 对预测不准确问题生成的审查报告

## 1. 审查结论

我按当前项目的三组实验重新跑了 `seed42_test100/k1` 的前 30 个测试样本，并把本次审查结果单独保存到 `results/codex_audit/`，没有修改任何代码。

结论是：问题确实存在，但三组程度不同。`raw-data-LLM-only` 在本次 30 样本中没有严重塌缩，预测 S 占 53.3%；`semantic-data-LLM-only` 明显偏向 S，预测 S 占 83.3%；`raw-data-LLM+RAG` 完全塌缩到 S，30 个样本全部预测为 S。

因此，主要问题不是 JSON 解析或标签映射错误，而是 LLM 在语义化特征和 RAG 知识注入后，把大量 Android 常见 API、权限、反射、加密、IPC 等特征过度解释为恶意证据。

## 2. 本次复现实验配置

PPT 要求覆盖恶意/良性分类、LLM-only、RAG+LLM、解释与错误案例分析。本次审查对应 Task A 和 Task D，并聚焦三组方法：

| 实验 | 运行入口 | 关键参数 | 输出位置 |
|---|---|---|---|
| raw-data-LLM-only | `llm_feature_expr_fewshot.py --feature-expr raw` | `k=1`, `max-test-samples=30`, `temperature=0.1` | `results/codex_audit/feature_expr_llm/seed42_test100/k1/raw_full.*` |
| semantic-data-LLM-only | `llm_feature_expr_fewshot.py --feature-expr semantic` | 同上，使用 `feature_semantics_gemma.json` | `results/codex_audit/feature_expr_llm/seed42_test100/k1/semantic_full.*` |
| raw-data-LLM+RAG | `llm_rag_raw_fewshot.py` | `k=1`, `top-k=3`, `local-files-only`, `max-test-samples=30` | `results/codex_audit/rag_raw_llm/seed42_test100/k1/rag_raw_top3.*` |

测试集前 30 条的真实标签分布是：B=16，S=14，所以如果模型全部预测 S，准确率会落到 46.7%，并且 benign recall 会是 0。

## 3. 复跑结果

| 实验 | Accuracy | Macro F1 | Malware Recall | Benign Recall | 预测 B | 预测 S | S 比例 |
|---|---:|---:|---:|---:|---:|---:|---:|
| raw-data-LLM-only | 0.800 | 0.800 | 0.857 | 0.750 | 14 | 16 | 0.533 |
| semantic-data-LLM-only | 0.633 | 0.597 | 1.000 | 0.313 | 5 | 25 | 0.833 |
| raw-data-LLM+RAG | 0.467 | 0.318 | 1.000 | 0.000 | 0 | 30 | 1.000 |

补充观察：

- 三组解析成功率都接近或等于 100%，所以不是解析器把输出错转成 S。
- RAG 组 16 个良性样本全部被误判为 S，`FP=16, TN=0, FN=0, TP=14`。
- RAG 组错误样本平均置信度约 0.952，说明模型不是“不确定”，而是非常自信地错判。

## 4. 关键证据

### 4.1 semantic-only 为什么更偏 S

`paper_prompt_utils.py` 会把每个 active feature 展开成“特征名: 中文语义描述”，相关逻辑在 `row_to_semantic_feature_text` 中。这个设计本身是合理的，但当前语义文本会把中性的 API 名称展开成安全语境中非常敏感的词。

例如一个真实标签为 B 的样本，在 semantic prompt 中出现：

- `DexClassLoader`: 动态加载和执行外部代码
- `Runtime.exec`: 执行外部命令或程序
- `TelephonyManager.getDeviceId`: 获取设备唯一标识符
- `Ljavax.crypto.Cipher`: 数据加密或解密
- `registerReceiver`: 监听系统事件

这些词单独看都“像恶意软件”，但在 Drebin 全量数据中，其中不少特征并不是恶意强指示器。比如：

| 特征 | B 中出现率 | S 中出现率 | 方向 |
|---|---:|---:|---|
| `Ljavax.crypto.Cipher` | 0.536 | 0.264 | 更偏 B |
| `Ljava.lang.Object.getClass` | 0.807 | 0.593 | 更偏 B |
| `android.os.IBinder` | 0.847 | 0.676 | 更偏 B |
| `DexClassLoader` | 0.290 | 0.057 | 更偏 B |
| `mount` | 0.685 | 0.448 | 更偏 B |
| `SEND_SMS` | 0.059 | 0.539 | 更偏 S |
| `READ_SMS` | 0.076 | 0.375 | 更偏 S |

也就是说，LLM 的安全常识先验和 Drebin 数据集里的统计事实不一致。语义化文本放大了这个先验，导致 benign recall 从 raw-only 的 0.750 掉到 0.313。

### 4.2 RAG 为什么最严重

RAG 组的问题更明显。30 个样本的 90 条 top-3 检索命中里：

- `feature_card`: 89 条
- `behavior_rule`: 1 条

也就是说，RAG 大多数时候没有检索到真正的行为规则，而是在检索单个特征解释卡。最常被检索的文档是：

| doc_id | 命中次数 |
|---|---:|
| `FEAT_LJAVAX_CRYPTO_CIPHER` | 19 |
| `FEAT_LJAVA_LANG_CLASS_GETCLASSES` | 18 |
| `FEAT_LJAVA_LANG_OBJECT_GETCLASS` | 9 |
| `FEAT_ANDROID_OS_IBINDER` | 7 |
| `FEAT_LJAVAX_CRYPTO_SPEC_SECRETKEYSPEC` | 7 |
| `FEAT_IBINDER` | 6 |

这些高频命中文档恰好大多是常见 Android API 或在 B 类中更常出现的特征。RAG 本应提供“规则依据”，但当前实际效果是把模型注意力反复引向 `Cipher`、反射、Binder、动态加载等容易触发恶意联想的词。

`llm_rag_raw_fewshot.py` 的 `build_query_text` 会把当前样本全部 active raw features 拼成长 query；`format_retrieved_docs` 再把 top-k 文档塞进 `RETRIEVED KNOWLEDGE`。由于知识库里有 215 条 `feature_card` 和 16 条 `behavior_rule`，单个特征解释卡在向量检索中压倒了行为规则。

### 4.3 当前输入缺少“反证”信息

当前 prompt 只列出 active features，不列出 inactive features。恶意软件判断中，很多良性证据其实来自“没有出现某些高危组合”，例如没有短信读写、没有安装卸载、没有 root/su、没有 APN 修改等。

更关键的是，Drebin 数据里良性样本平均 active feature 数反而更高：

| 类别 | 全量平均 active feature 数 | 本次 30 样本平均 active feature 数 |
|---|---:|---:|
| B | 37.75 | 40.19 |
| S | 25.89 | 26.50 |

LLM 很容易把“特征很多”误读成“行为复杂且可疑”，但这个数据集里特征多并不天然代表恶意。

### 4.4 k=1 的示例校准能力不足

本次复跑沿用当前调试目录 `k1`。训练侧只有 1 个 S 示例和 1 个 B 示例。那个 B 示例本身也包含 `Cipher`、`Runtime.exec`、`mount`、`TelephonyManager.getDeviceId` 等高敏感特征。raw-only 还能从示例中学到“这些特征不必然是 S”，但 semantic 和 RAG 又额外强调了这些特征的风险含义，于是校准被冲掉。

## 5. 对“是否大部分预测 S”的判断

如果严格看本次 30 样本复跑：

- raw-data-LLM-only：没有明显“大部分预测 S”，S 比例为 53.3%。
- semantic-data-LLM-only：是，S 比例为 83.3%。
- raw-data-LLM+RAG：是，而且最严重，S 比例为 100%。

如果参考项目里已有的 smoke test 日志，小样本结果也支持这个趋势：例如旧的 `results/rag_raw_llm/seed42_test100/k1/rag_raw_top3_metrics.json` 中 5 条样本全部预测 S；`results/feature_expr_llm/seed42_test100/k1/semantic_full_metrics.json` 中 10 条样本预测 S 比例为 70%。

## 6. 建议的修改方向

下面这些建议都尽量避免“在 prompt 中加入不要轻易判 S”这类人为引导词，而是从数据表示、检索机制、知识库结构和实验设计上修正偏差。

### 6.1 给 RAG 分离 feature card 和 behavior rule

不要把 215 条单特征解释卡和 16 条行为规则混在同一个 top-k 向量池里竞争。建议改成两路：

- active feature exact match：只用于解释当前样本里出现的特征含义。
- behavior rule retrieval：只在规则库上检索或匹配，规则按特征组合重叠度、类别覆盖度、训练集支持度打分。

这样 RAG 注入的内容才更接近 PPT 要求的“安全规则知识库”，而不是重复解释单个 API。

### 6.2 在知识库中加入数据统计，而不是人类倾向词

对每个 feature 或 rule，用训练集计算：

- `support_B`
- `support_S`
- `P(feature|B)`
- `P(feature|S)`
- log-odds 或 mutual information

例如 `Ljavax.crypto.Cipher` 在 B 中更常见，就不应被当作恶意证据强化。这个修改不是给 LLM 加人为倾向，而是把训练数据中的经验分布显式提供给模型。

### 6.3 semantic 描述要降风险词密度

当前 semantic 描述没有直接泄漏标签，但会放大安全风险联想。建议把语义文件改成更中性的字段结构：

- feature name
- category
- Android API/permission literal meaning
- whether it is common in benign training samples
- whether it participates in any matched behavior rule

尽量不要让单个 feature 的描述自动带出“隐藏真实逻辑”“外部代码执行”“窃取”等强结论，除非它来自规则匹配或训练统计支持。

### 6.4 补充 compact 的 inactive 高危组信息

不需要列出全部 215 个 inactive features，但可以按预定义组给出数据事实，例如：

- SMS group active count
- root/system-command group active count
- package-install group active count
- telephony-identifier group active count
- dynamic-loading group active count

这属于输入表示改造，不是告诉模型“应该判 B”。它能让模型看到“高危组合是否真的出现”，减少只凭若干常见 API 误判。

### 6.5 做特征筛选，而不是 full active list 全塞

`feature_subset=full` 会把大量弱区分、甚至 benign 更常见的特征塞进 prompt。建议增加数据驱动的 `feature_subset`：

- top mutual information features
- top absolute log-odds features
- matched rule features
- per-category capped features

raw/semantic/RAG 三组都用同一套筛选策略，保持公平。

### 6.6 k=1 只适合 smoke test，正式结论应至少比较 k=1/3/5

当前 k=1 的示例太少，单个示例选择会显著影响 LLM。建议正式报告里至少放：

- Random k=1/3/5
- Core-set k=1/3/5
- 每个 k 下三组方法的 pred_S_ratio、benign recall、malware recall

如果偏 S 主要在 k=1 出现，这是 few-shot 校准不足；如果 k=5 仍然出现，才说明语义/RAG 管线有系统性偏差。

## 7. 下一步优先级

优先修 RAG。当前 RAG 组 30/30 全部预测 S，且 89/90 次检索命中都是 `feature_card`，这说明 RAG 并没有真正发挥“规则增强”的作用。

其次修 semantic 表示。semantic-only 的 raw name + description 会让模型把许多中性或 benign 更常见的 API 解释成恶意线索，需要引入训练集统计或更中性的语义结构。

raw-only 暂时可作为相对稳定的参照组。本次 30 样本中 raw-only 指标最好，说明“原始特征名 + few-shot 示例”反而比未经校准的语义化/RAG 更稳。
