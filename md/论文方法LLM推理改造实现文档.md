# 论文方法 LLM 推理改造实现文档

## 1. 文档定位

本文档用于下一阶段代码实现。当前阶段只确定实现规格，不修改任何代码。

本轮目标不是完整复现原计划中的 24 组实验矩阵，而是在当前 raw feature LLM baseline 的基础上，新增一个使用语义解释特征的对照实验。两组实验只改变特征表达方式，其他口径尽量保持一致。

最终对照关系：

| 实验 | feature_expr | feature_subset | 说明 |
|---|---|---|---|
| raw baseline | `raw` | `full` | 使用原始激活特征名，作为已有基线 |
| semantic experiment | `semantic` | `full` | 使用 Gemma 离线生成的特征语义描述 |

不做 `chi2`、`mi`、`weighted` 子集，不做 `0-shot / 2-shot / 4-shot` 矩阵，不跑 24 组配置。

## 2. 已确认决策

| 决策点 | 最终选择 | 实现含义 |
|---|---|---|
| 决策 1：数据划分 | 方案 A | 继续使用当前固定测试集 `data/processed/fewshot_seed42_test100` |
| 决策 2：特征筛选数据源 | 方案 A | 若后续恢复特征筛选，用排除测试集后的全量训练池计算；本轮 full-only 暂不触发 Chi2/MI |
| 决策 3：Shot 含义 | 方案 B | 沿用当前项目 `k` 语义，`k=5` 表示每类 5 条 few-shot 示例，总共 10 条 |
| 决策 4：语义映射深度 | 方案 C | 使用 Gemma 离线为全部 215 个特征生成中性语义描述，人工只做快速校验和修正 |
| 决策 5：输出格式 | 方案 A | 继续要求 LLM 输出 JSON，便于自动解析和统一评估 |

## 3. 实验边界

第一版只实现 raw full 与 semantic full 的公平对照。

保留内容：

1. 固定测试集：`data/processed/fewshot_seed42_test100/test.csv`。
2. 固定 few-shot 训练口径：使用 `k*/train.csv`，其中 `k` 表示每类样本数。
3. 固定标签语义：`B = benign app / 良性应用`，`S = malware app / 恶意软件`。
4. 固定特征集合：全部 215 个 Drebin 特征。
5. 固定评估逻辑：accuracy、precision、recall_benign、recall_malware、f1、macro_f1、confusion matrix、pred_S_ratio、parse_ok_rate。

明确不做：

1. 不新增 24 组 Prompt 矩阵。
2. 不实现 `chi2`、`mi`、`weighted` 三类特征子集。
3. 不改变数据划分脚本。
4. 不修改 `llm_only_fewshot.py` 作为原始 baseline 的定位。
5. 不做 RAG、多 seed、模型微调。

## 4. 建议新增文件

建议新增以下文件，尽量不改动已有 baseline 文件：

```text
generate_feature_semantics.py
paper_prompt_utils.py
llm_feature_expr_fewshot.py
```

文件职责：

| 文件 | 职责 |
|---|---|
| `generate_feature_semantics.py` | 使用 Gemma 离线生成 215 个特征的中性语义描述，并导出可人工校验的文件 |
| `paper_prompt_utils.py` | 提供 raw/semantic 两种特征文本构造函数，保证两组实验只改变特征表达 |
| `llm_feature_expr_fewshot.py` | 运行 raw 或 semantic full 实验，复用现有 JSON 解析和指标口径 |

说明：`llm_only_fewshot.py` 继续保留为现有 raw baseline。新 runner 可以支持 `--feature-expr raw|semantic|both`，但默认建议只跑 `semantic`，避免不必要重复调用 LLM。

## 5. 语义描述生成

### 5.1 输入来源

语义生成只使用特征名和特征类别，不读取样本标签，不读取训练/测试样本取值，避免把标签信息泄漏进描述。

输入文件：

```text
data/drebin-215-dataset-5560malware-9476-benign.csv
data/dataset-features-categories.csv
```

特征列表来源：Drebin 主 CSV 的所有列，排除 `class` 列，理论上应为 215 个。

类别来源：`data/dataset-features-categories.csv`，用于帮助 Gemma 理解特征属于权限、API、Intent、Command 等哪一类。

### 5.2 Gemma 生成原则

每个特征生成一条中性、短句式中文描述。描述只解释 Android 行为含义，不直接暗示“恶意”“可疑”“危险”，除非特征名本身明确包含该含义。

建议约束：

1. 描述长度控制在 10 到 30 个中文字符左右。
2. 不输出分类判断。
3. 不输出风险评分。
4. 不输出“该应用是恶意软件”这类结论。
5. 对权限类特征使用“请求/使用某权限”的说法。
6. 对 API 类特征使用“调用某系统接口以...”的说法。
7. 对 Intent 类特征使用“响应/监听某系统事件”的说法。
8. 对 Command 类特征使用“执行/调用某命令能力”的说法。

生成稳定性约束：

1. 生成 feature semantics 时 `temperature` 固定为 `0` 或尽可能低。
2. 若 Gemma 输出不是合法 JSON，最多重试 2 次。
3. 若重试后仍失败，则使用模板 fallback 生成描述，不能让单个特征阻塞整体文件生成。
4. 脚本需要支持 `--resume`，已有语义描述不重复生成。
5. 脚本需要支持 `--max-retries 2` 和 `--fallback-template`。
6. 生成语义描述时使用的 prompt 模板必须保存，便于后续报告复盘。

语义描述污染检查：

生成 `description` 后需要做自动关键词检查，标记可能污染分类判断的词。检查结果不直接删除描述，但必须写入 review CSV，方便人工快速定位。

默认黑名单关键词：

```text
恶意
可疑
危险
高危
攻击
木马
窃取
勒索
病毒
风险
malicious
suspicious
risky
trojan
attack
```

说明：`读取`、`发送短信`、`联网`、`执行命令` 这类行为词可以保留，因为它们描述的是特征本身；黑名单主要用于标记“恶意/可疑/高危/风险”这类容易把分类结论提前写入特征语义的词。

示例：

```text
SEND_SMS -> 请求发送短信的权限
RECEIVE_BOOT_COMPLETED -> 监听设备启动完成事件
TelephonyManager.getDeviceId -> 读取设备标识信息
Runtime.exec -> 调用系统命令执行接口
```

### 5.3 输出文件

建议输出到：

```text
data/processed/paper_features/feature_semantics_gemma.json
data/processed/paper_features/feature_semantics_review.csv
data/processed/paper_features/feature_semantics_generation_prompt.txt
```

JSON 推荐结构：

```json
{
  "SEND_SMS": {
    "category": "Manifest Permission",
    "description": "请求发送短信的权限",
    "source": "gemma_offline",
    "review_status": "unchecked",
    "raw_response": "Gemma original output",
    "generation_prompt": "Prompt text used to generate this feature description",
    "flagged_words": []
  }
}
```

`feature_semantics_generation_prompt.txt` 保存生成语义描述时使用的完整 prompt 模板。JSON 中的 `generation_prompt` 保存单个特征实际使用的生成 prompt；如果后续觉得 JSON 过大，也可以改为保存 `generation_prompt_path`，但必须能追溯每个描述是怎么生成的。

review CSV 建议至少包含以下列：

```text
feature,category,description,review_status,flagged_words,source
```

其中 `flagged_words` 存放命中的黑名单关键词，多个词用分号分隔；没有命中则为空。

人工校验后可以把 `review_status` 改为 `approved` 或 `edited`。若 `flagged_words` 非空，建议默认将 `review_status` 标为 `needs_review`。代码实际运行时读取最终 JSON，若某个特征缺少描述，则退回原始特征名，不能中断整个实验。

注意：语义描述生成阶段与分类推理阶段相互独立。`feature_semantics_gemma.json` 生成完成后视为固定输入资源；后续 semantic 实验只能读取该文件，不允许在分类推理过程中重新调用 LLM 生成、补全或改写特征描述。

## 6. 特征表达构造

两种表达都只列出当前样本中激活值为 `1` 的特征，且都使用 full 特征集合。

### 6.1 raw 表达

raw 表达沿用当前 `fewshot_utils.row_to_feature_text` 的思想：按类别列出激活特征名。

示例：

```text
Manifest Permission: SEND_SMS, READ_PHONE_STATE
API call signature: TelephonyManager.getDeviceId
```

### 6.2 semantic 表达

semantic 表达把激活特征转换成自然语言描述，同时保留原始特征名，便于调试和追溯。

建议格式：

```text
Manifest Permission:
- SEND_SMS: 请求发送短信的权限
- READ_PHONE_STATE: 请求读取手机状态的权限

API call signature:
- TelephonyManager.getDeviceId: 读取设备标识信息
```

如果没有激活特征，返回：

```text
Active features: none
```

## 7. Prompt 口径

Prompt 主体尽量与当前 `llm_only_fewshot.py` 保持一致，只替换“样本特征文本”部分。

必须保持一致的内容：

1. 分类任务说明：Android 应用良性/恶意二分类。
2. 标签定义：`B = benign app / 良性应用`，`S = malware app / 恶意软件`。
3. few-shot 示例来源：同一个 `k*/train.csv`。
4. 测试样本来源：同一个 `test.csv`。
5. 输出格式：严格 JSON。

最终分类 prompt 中不要使用会扩大 S 判定范围的混合词，例如 `suspicious`、`potentially malicious`、`risky`。标签说明只保留 `benign app` 与 `malware app`，避免模型因为“可疑/风险”措辞而过度预测 `S`。

不要再使用 `shots` 作为实验变量。实现中使用：

```text
k = 每类 few-shot 示例数量
fewshot_total = 2 * k
k_semantics = "per_class"
```

例如 `--k 5` 表示 prompt 内包含 5 条 B 示例和 5 条 S 示例，总共 10 条 few-shot 示例。

## 8. 运行命令设计

### 8.1 生成语义描述

```bash
python generate_feature_semantics.py \
  --data-path data/drebin-215-dataset-5560malware-9476-benign.csv \
  --feature-category-path data/dataset-features-categories.csv \
  --model gemma4:e4b \
  --temperature 0 \
  --max-retries 2 \
  --resume \
  --fallback-template \
  --output-json data/processed/paper_features/feature_semantics_gemma.json \
  --output-review-csv data/processed/paper_features/feature_semantics_review.csv \
  --output-generation-prompt data/processed/paper_features/feature_semantics_generation_prompt.txt
```

### 8.2 运行已有 raw baseline

如需复跑现有 baseline：

```bash
python llm_only_fewshot.py \
  --split-root data/processed/fewshot_seed42_test100 \
  --k 5
```

也可以直接复用已有结果：

```text
results/predictions/llm_only_seed42_test100_k5.jsonl
```

### 8.3 运行 semantic full 实验

```bash
python llm_feature_expr_fewshot.py \
  --split-root data/processed/fewshot_seed42_test100 \
  --k 5 \
  --feature-expr semantic \
  --feature-subset full \
  --feature-semantics data/processed/paper_features/feature_semantics_gemma.json
```

### 8.4 小样本冒烟测试

```bash
python llm_feature_expr_fewshot.py \
  --split-root data/processed/fewshot_seed42_test100 \
  --k 5 \
  --feature-expr semantic \
  --feature-subset full \
  --feature-semantics data/processed/paper_features/feature_semantics_gemma.json \
  --max-test-samples 5
```

## 9. 输出格式

建议输出目录：

```text
results/feature_expr_llm/seed42_test100/k5/
```

semantic 实验输出：

```text
semantic_full.jsonl
semantic_full_metrics.json
```

若新 runner 也复跑 raw，则输出：

```text
raw_full.jsonl
raw_full_metrics.json
```

汇总表：

```text
feature_expr_metrics.csv
```

Prompt 保存目录：

```text
results/feature_expr_llm/seed42_test100/k5/prompts/semantic_full/
```

分类推理时必须保存每条样本实际发送给 LLM 的 prompt。为避免 JSONL 过大，推荐把完整 prompt 写入独立 `.txt` 文件，JSONL 里保存 `prompt_path`。

JSONL 单条记录建议结构：

```json
{
  "experiment_id": "semantic_full_k5",
  "feature_expr": "semantic",
  "feature_subset": "full",
  "k": 5,
  "k_semantics": "per_class",
  "fewshot_total": 10,
  "idx": 10871,
  "true_label": "B",
  "pred_label": "B",
  "correct": true,
  "parse_ok": true,
  "strict_parse_ok": true,
  "parse_source": "strict_json",
  "prompt_path": "results/feature_expr_llm/seed42_test100/k5/prompts/semantic_full/10871.txt",
  "raw": "original model output"
}
```

指标字段建议在当前 baseline 基础上加入预测分布统计：

```json
{
  "experiment_id": "semantic_full_k5",
  "feature_expr": "semantic",
  "feature_subset": "full",
  "k": 5,
  "k_semantics": "per_class",
  "total": 200,
  "accuracy_strict": 0.82,
  "precision": 0.81,
  "recall_benign": 0.75,
  "recall_malware": 0.84,
  "f1": 0.82,
  "macro_f1": 0.82,
  "pred_B_count": 40,
  "pred_S_count": 160,
  "pred_S_ratio": 0.80,
  "parse_ok_rate": 1.0,
  "TP": 84,
  "TN": 80,
  "FP": 20,
  "FN": 16
}
```

最终报告表只需要两行：

| experiment_id | feature_expr | feature_subset | k | accuracy_strict | precision | recall_benign | recall_malware | f1 | macro_f1 | pred_S_ratio | parse_ok_rate |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| raw_full_k5 | raw | full | 5 |  |  |  |  |  |  |  |  |
| semantic_full_k5 | semantic | full | 5 |  |  |  |  |  |  |  |  |

## 10. 实现步骤

1. 新增 `generate_feature_semantics.py`。
   - 读取 215 个特征名。
   - 读取特征类别。
   - 调用本地 Gemma/Ollama 生成中性中文描述。
   - 固定低温生成，默认 `--temperature 0`。
   - 支持断点续写，已有描述不重复生成。
   - 支持非法 JSON 重试，默认 `--max-retries 2`。
   - 支持模板 fallback，避免单个特征生成失败导致整体中断。
   - 保存每个特征的 `raw_response`，便于人工排查语义解释错误。
   - 保存生成语义描述时使用的 prompt 模板到 `feature_semantics_generation_prompt.txt`。
   - 在每个特征记录中保存 `generation_prompt` 或可追溯的 `generation_prompt_path`。
   - 对 `description` 做黑名单关键词检查，并把命中结果写入 `flagged_words`。
   - 输出 JSON 和 review CSV。

2. 新增 `paper_prompt_utils.py`。
   - 提供 `load_feature_semantics(path)`。
   - 提供 `row_to_raw_feature_text(row, feature_categories)`。
   - 提供 `row_to_semantic_feature_text(row, feature_categories, feature_semantics)`。
   - 保证 raw 与 semantic 只改变特征文本，不改变标签和样本选择逻辑。

3. 新增 `llm_feature_expr_fewshot.py`。
   - CLI 参数包含 `--split-root`、`--k`、`--feature-expr`、`--feature-subset`、`--feature-semantics`。
   - `--feature-subset` 第一版只允许 `full`。
   - 分类阶段只能读取固定的 `feature_semantics_gemma.json`，不能动态调用 LLM 解释特征。
   - 复用现有 JSON 输出约束、Ollama 调用、解析和指标计算逻辑。
   - 为每条样本保存完整 prompt 文件，并在 JSONL 中记录 `prompt_path`。
   - 指标中加入 `pred_B_count`、`pred_S_count`、`pred_S_ratio`、`recall_benign`。
   - 输出 JSONL、metrics JSON 和汇总 CSV。

4. 做小样本 dry run 或 smoke test。
   - 确认 semantic prompt 内出现语义描述。
   - 确认 raw/semantic 使用相同 train/test 样本。
   - 确认 `k=5` 时示例总数为 10。
   - 确认 prompt 文件已保存，JSONL 中存在 `prompt_path`。
   - 确认 JSONL 和 metrics 正常生成。

5. 跑完整 semantic full。
   - 与 raw baseline 的 `accuracy_strict`、`f1`、`macro_f1`、`pred_S_ratio` 等指标横向比较。

## 11. 验收标准

语义映射验收：

1. `feature_semantics_gemma.json` 覆盖 215 个特征。
2. 每个特征都有非空 `description`。
3. 每个特征都保留 `raw_response`。
4. 每个特征都能追溯生成语义时使用的 prompt，至少存在 `feature_semantics_generation_prompt.txt`。
5. review CSV 包含 `flagged_words` 列。
6. 描述不包含明显分类结论，例如“恶意应用会...”。
7. 人工快速校验后，明显错误的描述已修正。
8. `flagged_words` 非空的记录已人工检查，确认没有把分类倾向写入特征语义。

实验验收：

1. semantic 实验只使用 `feature_subset=full`。
2. semantic 实验与 raw baseline 使用同一个 `split-root` 和同一个 `k`。
3. `k_semantics` 明确记录为 `per_class`。
4. 测试样本数量与 raw baseline 一致。
5. 输出 JSONL 每条记录都包含 `true_label`、`pred_label`、`parse_ok`、`prompt_path`。
6. 每条记录对应的 prompt 文件真实存在。
7. metrics 文件包含 `pred_B_count`、`pred_S_count`、`pred_S_ratio`、`recall_benign`、`recall_malware`。
8. metrics 文件可以直接与 raw baseline 横向对比。

## 12. 给下一阶段实现的提醒

下一阶段不要从原计划的 24 组矩阵开始实现。先完成最小闭环：

```text
Gemma 生成 215 特征语义描述
-> 人工快速校验
-> semantic full LLM 推理
-> 与 raw full baseline 对比
```

如果后续还想恢复论文中的特征子集或 shot 矩阵，再基于这个闭环扩展；第一版不要把复杂度加回来。
