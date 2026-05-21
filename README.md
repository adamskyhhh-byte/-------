# 基于 LLM / RAG 的 Drebin Android 恶意软件少样本检测实验

本项目是一个面向课程大作业与实验报告的 Android 恶意软件检测实验仓库。项目以 Drebin-215 二值特征数据集为基础，对比传统机器学习、纯 LLM few-shot、特征语义增强 LLM，以及 RAG 增强 LLM 在少样本场景下的二分类效果。

标签约定：

- `B`：Benign，良性应用
- `S`：Malware，恶意应用

核心问题不是只看“能不能分类”，而是进一步观察：在每类只有 K 个训练样本时，LLM 是否会因为安全常识、特征表达方式或检索知识而产生系统性偏差。

## 项目结构

```text
.
├── data/
│   ├── drebin-215-dataset-5560malware-9476-benign.csv   # Drebin 原始特征数据
│   ├── dataset-features-categories.csv                  # 215 个特征的类别映射
│   └── processed/                                       # 少样本划分、特征统计、RAG 知识库等中间产物
├── results/                                             # 各实验线输出的预测、指标、混淆矩阵和汇总表
├── logs/                                                # 长时间实验运行日志
├── md/                                                  # 方案、审查、修复与最终报告文档
│   └── final_report/                                    # 最终报告、图表与误差分析
├── prepare_fewshot_split.py                             # 固定 test 集并生成 K-shot train 集
├── baseline_fewshot.py                                  # 传统 ML / MLP baseline
├── llm_only_fewshot.py                                  # raw feature 的 LLM-only few-shot
├── generate_feature_semantics.py                        # 生成中性特征语义说明
├── feature_stats.py                                     # 训练池特征统计，避免 test 泄漏
├── llm_feature_expr_fewshot.py                          # raw / semantic 特征表达对照实验
├── rag_kb_builder.py                                    # 构建修复版 RAG 知识库与 FAISS 索引
├── rag_retriever.py                                     # RAG 检索器
├── llm_rag_raw_fewshot.py                               # raw feature + RAG 的 LLM few-shot 实验
├── RAG_test.py                                          # 单独调试 RAG 检索效果
└── collect_results.py                                   # 汇总指标到 Excel 并生成图表
```

## 环境准备

建议使用 Python 3.10+。本项目目前没有单独的 `requirements.txt`，可先按下面命令安装主要依赖：

```bash
python -m pip install pandas numpy scikit-learn matplotlib openpyxl ollama sentence-transformers faiss-cpu
```

LLM 实验默认调用本地 Ollama：

```bash
ollama serve
ollama pull gemma4:e4b
```

RAG 默认使用 Hugging Face 模型 `BAAI/bge-small-en-v1.5` 生成 embedding。首次运行需要联网下载模型；如果本地已缓存模型，可以在相关脚本中加入 `--local-files-only`。

## 数据说明

默认原始数据路径：

```text
data/drebin-215-dataset-5560malware-9476-benign.csv
```

该数据集包含：

- 15,036 个 Android 应用样本
- 215 个 Drebin 二值特征
- `class` 标签列，取值为 `B` 或 `S`
- 配套特征类别文件 `data/dataset-features-categories.csv`

少样本实验采用固定测试集策略：先从全量数据中每类抽取固定数量测试样本，再从剩余训练池中为每个 K 值抽取每类 K 条训练样本。这样 K=1、K=3、K=5 可以共享同一份 test 集，便于公平对比。

## 复现实验流程

### 1. 生成少样本划分

```bash
python prepare_fewshot_split.py --k 1 --seed 42 --test-per-class 100
python prepare_fewshot_split.py --k 3 --seed 42 --test-per-class 100
python prepare_fewshot_split.py --k 5 --seed 42 --test-per-class 100
```

输出位置：

```text
data/processed/fewshot_seed42_test100/
├── test.csv
├── test_metadata.json
├── k1/train.csv
├── k3/train.csv
└── k5/train.csv
```

### 2. 计算训练池特征统计

```bash
python feature_stats.py --data-path data/drebin-215-dataset-5560malware-9476-benign.csv --test-metadata data/processed/fewshot_seed42_test100/test_metadata.json --output-dir data/processed/feature_stats/seed42_test100
```

这一步会剔除 test 样本后再统计每个特征在 B / S 中的出现频率、log odds 和倾向方向，用于后续 semantic / RAG 实验，避免测试集信息泄漏。

### 3. 运行传统 baseline

```bash
python baseline_fewshot.py --split-root data/processed/fewshot_seed42_test100 --k 1 --max-test-samples 50 --output-root results/final_50/baseline
python baseline_fewshot.py --split-root data/processed/fewshot_seed42_test100 --k 3 --max-test-samples 50 --output-root results/final_50/baseline
python baseline_fewshot.py --split-root data/processed/fewshot_seed42_test100 --k 5 --max-test-samples 50 --output-root results/final_50/baseline
```

baseline 包含 Logistic Regression、Linear SVM、Random Forest、Decision Tree、KNN 和 MLP Neural Net。输出包括 `baseline_metrics.csv`、`run_metadata.json` 和各模型混淆矩阵图片。

### 4. 运行 LLM-only few-shot

```bash
python llm_only_fewshot.py --split-root data/processed/fewshot_seed42_test100 --k 5 --max-test-samples 50 --model gemma4:e4b
```

该脚本只把 few-shot 示例和当前样本的 active Drebin 原始特征名写入 prompt，不注入 RAG 知识。输出为 JSONL 预测记录和 metrics JSON。

### 5. 生成中性特征语义

```bash
python generate_feature_semantics.py --style neutral-fixed --feature-stats data/processed/feature_stats/seed42_test100/feature_stats.csv --output-json data/processed/paper_features/feature_semantics_neutral_stats.json --resume --fallback-template
```

该步骤会为 215 个 Drebin 特征生成中性英文释义，并尽量避免 `malware`、`suspicious`、`dangerous` 等风险词污染 LLM 判断。

### 6. 运行特征表达对照实验

```bash
python llm_feature_expr_fewshot.py --split-root data/processed/fewshot_seed42_test100 --k 5 --feature-expr all --feature-semantics-risky-old data/processed/paper_features/feature_semantics_gemma.json --feature-semantics-neutral-fixed data/processed/paper_features/feature_semantics_neutral_stats.json --feature-stats data/processed/feature_stats/seed42_test100/feature_stats.csv --max-test-samples 50 --output-root results/final_50/feature_expr_llm
```

`--feature-expr all` 会运行 raw、semantic-risky-old、semantic-neutral-fixed 三种表达方式，便于观察“特征语义写法”对 LLM 分类偏差的影响。

### 7. 构建 RAG 知识库

```bash
python rag_kb_builder.py --feature-semantics data/processed/paper_features/feature_semantics_neutral_stats.json --feature-stats data/processed/feature_stats/seed42_test100/feature_stats.csv --output-dir data/processed/rag_kb_fixed --show-progress
```

输出包括：

- `kb_docs.jsonl`：feature card 与 behavior rule 文档
- `rules.csv`：规则表
- `kb.index`：全量 FAISS 索引
- `kb_feature.index`：特征卡片子索引
- `kb_rule.index`：行为规则子索引
- `kb_metadata.json`：知识库元数据

### 8. 运行 RAG + LLM 实验

```bash
python llm_rag_raw_fewshot.py --split-root data/processed/fewshot_seed42_test100 --k 5 --retrieval-mode bucketed --feature-top-k 2 --rule-top-k 3 --kb-dir data/processed/rag_kb_fixed --max-test-samples 50 --output-root results/final_50/rag_raw_llm
```

bucketed 检索会分别从 feature card 和 behavior rule 两个桶中取知识文档，再注入 prompt。这样可以避免只检索到同一种类型的上下文。

### 9. 汇总最终结果

```bash
python collect_results.py --results-root results/final_50 --feature-stats data/processed/feature_stats/seed42_test100/feature_stats.csv --output-xlsx results/final_50/results.xlsx --figures-dir md/final_report/figures
```

最终报告可查看：

- `md/final_report/report.md`
- `md/final_report/prompt.md`
- `md/final_report/error_cases.md`
- `results/final_50/results.xlsx`

## 已有实验结论摘要

当前 `md/final_report/report.md` 中记录的最终实验规模为 test 前 50 条样本、K=1 / 3 / 5 三组设置。主要结论如下：

- 传统 ML / NN baseline 仍然很强，K=3 的 MLP Neural Net 达到最高 accuracy。
- raw LLM-only 没有稳定超过传统 baseline。
- 带风险词的旧版 semantic 表达会明显诱导 LLM 偏向 `S`。
- neutral-fixed semantic 减少了风险词污染，但 K 变大时 prompt 体积快速膨胀，可能触发上下文截断与解析失败。
- raw + RAG bucketed 是更稳定的 LLM 增强方向，因为知识以少量检索片段注入，而不是把全部语义和统计信息塞进 prompt。

## 调试与注意事项

- 如果报错缺少 `faiss`，安装 `faiss-cpu`。
- 如果报错缺少 `SentenceTransformer`，安装 `sentence-transformers`。
- 如果 Ollama 调用失败，确认本地服务已启动，并且模型名与 `--model` 一致。
- 如果 RAG 首次运行很慢，通常是在下载或加载 embedding 模型。
- 如果 prompt 很长导致 LLM 输出非 JSON，可优先检查 `num_ctx`、`--max-test-samples`、语义表达长度和保存下来的 `prompts/` 文件。
- `data/`、`results/`、`logs/` 中可能包含较大的本地数据和实验产物，提交代码仓库前建议确认是否需要纳入版本管理。

## 推荐阅读顺序

1. `md/final_report/report.md`：完整实验报告
2. `prepare_fewshot_split.py`：理解固定测试集和 K-shot 数据生成
3. `llm_only_fewshot.py`：理解 prompt、Ollama 调用和 JSON 解析
4. `rag_kb_builder.py` 与 `rag_retriever.py`：理解 RAG 知识库构建与检索
5. `llm_rag_raw_fewshot.py`：理解最终 RAG + LLM 推理流程
