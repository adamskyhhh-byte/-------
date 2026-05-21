# 论文方法 LLM 推理改造计划

## 1. 目标

在不修改现有 `llm_only_fewshot.py` 的前提下，新增一套独立实验代码，用论文中提出的方法改造原有 LLM-only 推理流程。

现有 `llm_only_fewshot.py` 继续保留为原始 LLM-only baseline。新方法单独实现、单独运行、单独保存结果，便于后续和原始方案公平对比。

## 2. 论文方法核心

论文方案不是单纯增加 few-shot 示例，而是把以下几个因素组合成完整实验框架：

1. 特征筛选：Full / Chi2 / Mutual Information / 加权组合。
2. 特征表达：原始特征名 / 语义化自然语言描述。
3. Prompt 结构：0-shot / 2-shot / 4-shot。
4. 类别均衡 few-shot：示例中良性与恶意样本数量保持平衡。
5. 结构化输出解析：统一提取模型最终判断结果。
6. 统一测试集评估：所有配置共用同一个测试集，保证结果可比。

## 3. 新增代码范围

建议新增以下文件，现有源代码不直接修改：

```text
paper_feature_selection.py
paper_prompt_utils.py
llm_paper_method.py
```

各文件职责如下：

| 文件 | 作用 |
|---|---|
| `paper_feature_selection.py` | 计算 Chi2、互信息、加权组合特征排名，并导出 Top 特征子集 |
| `paper_prompt_utils.py` | 构造原始特征文本、语义化特征文本、few-shot 示例和最终 Prompt |
| `llm_paper_method.py` | 运行论文式 LLM 推理实验，遍历 Prompt 配置并保存结果 |

现有文件保持不动：

```text
fewshot_utils.py
prepare_fewshot_split.py
baseline_fewshot.py
llm_only_fewshot.py
```

## 4. 特征筛选计划

从训练侧数据计算 215 维特征的重要性，生成四类特征子集：

1. `full`：使用全部 215 个特征。
2. `chi2`：按卡方检验得分选 Top N。
3. `mi`：按互信息得分选 Top N。
4. `weighted`：将 Chi2 和 MI 归一化后加权融合，再选 Top N。

默认 Top N 建议使用论文中的 `50`。

输出文件建议放在：

```text
data/processed/paper_features/
```

建议输出：

```text
feature_rankings.csv
selected_features_full.json
selected_features_chi2_top50.json
selected_features_mi_top50.json
selected_features_weighted_top50.json
```

## 5. 特征语义化计划

当前 `row_to_feature_text` 更接近“按类别列出活跃特征名”，还不是论文中的自然语言行为描述。

新方法中需要支持两种表达方式：

| 表达方式 | 说明 |
|---|---|
| `raw` | 直接列出当前样本中激活的特征名 |
| `semantic` | 将激活特征转成自然语言行为片段 |

语义化示例：

```text
SEND_SMS -> 请求发送短信权限
RECEIVE_BOOT_COMPLETED -> 设备启动后自动运行
TelephonyManager.getDeviceId -> 读取设备识别码
Runtime.exec -> 执行系统命令
```

第一版不建议为全部 215 个特征手写完整长描述，可以采用“规则模板 + 高风险特征手工补充”的方式：

1. 对权限类特征使用权限模板。
2. 对 API call signature 使用 API 行为模板。
3. 对 Intent 使用系统事件模板。
4. 对 Commands signature 使用命令执行模板。
5. 对重点高风险特征单独写更明确的中文描述。

## 6. Prompt 构造计划

每个测试样本的 Prompt 由以下部分组成：

1. 任务说明：Android 应用恶意/良性二分类。
2. 标签定义：`B = 良性`，`S = 恶意`。
3. Few-shot 示例：0、2 或 4 条。
4. 当前测试样本的特征输入。
5. 输出格式约束。

实验变量：

| 变量 | 取值 |
|---|---|
| `shots` | `0` / `2` / `4` |
| `feature_expr` | `raw` / `semantic` |
| `feature_subset` | `full` / `chi2` / `mi` / `weighted` |

总实验组合：

```text
3 种 shots × 2 种表达方式 × 4 种特征子集 = 24 组配置
```

Few-shot 示例必须类别均衡：

```text
2-shot = 1 条 B + 1 条 S
4-shot = 2 条 B + 2 条 S
0-shot = 不提供示例
```

## 7. LLM 运行计划

新增主脚本 `llm_paper_method.py`。

建议命令：

```bash
python llm_paper_method.py --split-root data/processed/fewshot_seed42_test100 --max-test-samples 5
```

完整实验命令后续可设计为：

```bash
python llm_paper_method.py --split-root data/processed/fewshot_seed42_test100
```

该脚本负责：

1. 加载固定测试集。
2. 加载 few-shot 示例池。
3. 加载四类特征子集。
4. 遍历 24 组 Prompt 配置。
5. 调用 Ollama。
6. 解析模型输出。
7. 保存 JSONL 和指标文件。

## 8. 输出格式计划

建议结果目录：

```text
results/paper_llm/seed42_test100/
```

每条预测记录保存为 JSONL：

```json
{
  "prompt_label": "4_shot_语义_weighted",
  "feature_subset": "weighted",
  "feature_expr": "semantic",
  "shots": 4,
  "idx": 10871,
  "true_label": "B",
  "pred_label": "B",
  "correct": true,
  "parse_ok": true,
  "raw": "original model output"
}
```

指标文件保存：

```json
{
  "prompt_label": "4_shot_语义_weighted",
  "accuracy_strict": 0.82,
  "precision": 0.81,
  "recall_malware": 0.84,
  "f1": 0.82,
  "macro_f1": 0.82,
  "parse_ok_rate": 1.0,
  "TP": 84,
  "TN": 80,
  "FP": 20,
  "FN": 16
}
```

最终汇总表建议保存为：

```text
paper_llm_metrics.csv
```

用于复现论文表 4-3 的结构：

| Prompt组合 | 原始全集 | Chi2 | 互信息 | 加权组合 |
|---|---:|---:|---:|---:|
| zero_shot_原始 |  |  |  |  |
| zero_shot_语义 |  |  |  |  |
| 2_shot_原始 |  |  |  |  |
| 2_shot_语义 |  |  |  |  |
| 4_shot_原始 |  |  |  |  |
| 4_shot_语义 |  |  |  |  |

## 9. 验收计划

第一步：只生成特征排名，不调用 LLM。

```bash
python paper_feature_selection.py --split-root data/processed/fewshot_seed42_test100
```

需要确认：

1. 能生成四类特征子集。
2. Top 50 特征数量正确。
3. 特征名都存在于原始数据列中。

第二步：只生成 Prompt 预览，不调用 LLM。

```bash
python llm_paper_method.py --split-root data/processed/fewshot_seed42_test100 --dry-run --max-test-samples 2
```

需要确认：

1. 0-shot 不含示例。
2. 2-shot 含 1 条 B 和 1 条 S。
3. 4-shot 含 2 条 B 和 2 条 S。
4. `raw` 和 `semantic` 两种表达确实不同。
5. `full` 与 Top 50 子集输入长度明显不同。

第三步：小样本 LLM 冒烟测试。

```bash
python llm_paper_method.py --split-root data/processed/fewshot_seed42_test100 --max-test-samples 5
```

需要确认：

1. JSONL 正常生成。
2. 每条记录有 `true_label` 和 `pred_label`。
3. `parse_ok_rate` 可统计。
4. 指标文件正常保存。

第四步：完整运行 24 组配置。

```bash
python llm_paper_method.py --split-root data/processed/fewshot_seed42_test100
```

需要确认：

1. 每组配置都有独立结果。
2. 汇总 CSV 能直接用于报告表格。
3. 能和原始 `llm_only_fewshot.py` 的结果进行横向对比。

## 10. 决策

以下问题需要在正式实现前确认。

### 决策 1：数据划分

方案 A：继续使用当前固定测试集 `fewshot_seed42_test100`。  
方案 B：重新做论文式 `70% train / 30% test`。

理由：可以直接和现有 LLM-only baseline、传统 ML few-shot baseline 公平对比。

### 决策 2：特征筛选的数据来源

方案 A：用“排除测试集后的全量训练池”计算 Chi2 / MI。  
方案 B：只用 few-shot 的 `train.csv` 计算 Chi2 / MI。

理由：few-shot 训练样本太少，只用它做特征排名会非常不稳定。

### 决策 3：Shot 含义

方案 A：按论文含义，`2-shot` 表示总共 2 条示例，`4-shot` 表示总共 4 条示例。  
方案 B：沿用当前项目语义，`k=2` 表示每类 2 条，总共 4 条。

理由：新方法目标是复现论文实验变量，旧项目语义可继续保留在原始 baseline 中。

### 决策 4：语义映射深度

方案 A：先做“规则模板 + 高风险特征手工补充”。  
方案 B：为全部 215 个特征逐个手写语义描述。

理由：第一版更快落地，也足够验证论文方法是否有效。

### 决策 5：输出格式

方案 A：继续要求模型输出 JSON，方便自动评估。  
方案 B：按论文文本要求输出“最终判断结果：恶意/良性”。

理由：项目已有 JSON 解析和严格评估逻辑；可以在 JSON 中额外保留中文最终判断字段，兼顾论文表述。

## 11. 实现边界

第一版不做以下内容：

1. 不修改现有 `llm_only_fewshot.py`。
2. 不修改现有 `fewshot_utils.py`。
3. 不重写现有数据切分脚本。
4. 不做 RAG。
5. 不做多 seed。
6. 不做模型微调。
7. 不手工为 215 个特征逐个写完整语义。

第一版只目标明确地完成：

```text
论文式特征筛选 + 特征语义化 + 类别均衡 few-shot + 24 组 Prompt 实验矩阵
```

