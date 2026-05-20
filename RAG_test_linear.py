# %% [markdown]
# # RAG 原始 Drebin 特征 few-shot 学习 demo
#
# 文件作用：
# 这是 LLM + RAG 实验的“初学者友好版 / notebook 风格版”。
# 真正可复用的工程代码在 `rag_retriever.py` 和 `llm_rag_raw_fewshot.py`；
# 这个文件导入那些 helper，然后把完整流程按学习顺序一步一步展示出来。
#
# 阅读方式：
# 在 VS Code 或 Jupyter 里从上到下运行每个 cell。
# 每个小节对应 RAG 实验里的一个概念，而不是严格的软件模块划分。
#
# 它覆盖项目指导里的阶段三、阶段四和阶段五：
#
# - 阶段三：给单个测试样本构造 RAG prompt。
# - 阶段四：运行 5 条样本的烟雾测试。
# - 阶段五：运行完整测试集，并读取指标做对比。
#
# 如果缺依赖，先安装：
#
#   python -m pip install sentence-transformers faiss-cpu ollama scikit-learn

# %%
from pathlib import Path

import pandas as pd

from fewshot_utils import load_feature_categories, load_fewshot_split, normalize_label
from llm_rag_raw_fewshot import (
    active_feature_names,
    build_query_text,
    build_rag_prompt,
    output_dir_for,
)
from rag_retriever import RagRetriever

# `display` 在 notebook 里会自动存在。
# 这个兜底函数是为了让本文件作为普通 Python 脚本运行时也不报错。
try:
    display
except NameError:
    def display(obj):
        print(obj)


# %% [markdown]
# ## 1. 基本实验设置
#
# 如果要和 LLM-only baseline 公平对比，这些设置应尽量保持一致。
# 只有 `top_k` 这类 RAG 相关设置是新增变量。

# %%
# 这些设置故意和 `llm_rag_raw_fewshot.py` 的命令行默认值保持接近。
# 如果你想在交互式学习时尝试不同 k-shot 或不同 top-k，就改这里。
split_root = "data/processed/fewshot_seed42_test100"
k = 5
top_k = 3

kb_dir = "data/processed/rag_kb"
feature_category_path = "data/dataset-features-categories.csv"
embedding_model = "BAAI/bge-small-en-v1.5"
llm_model = "gemma4:e4b"
local_files_only = False  # embedding 模型下载过一次后，可以改成 True 离线加载。

temperature = 0.1
request_timeout = 60.0

print("split_root:", split_root)
print("k:", k)
print("top_k:", top_k)
print("kb_dir:", kb_dir)


# %% [markdown]
# ## 2. 读取同一个 few-shot split
#
# 训练集行会变成 few-shot examples。
# 测试集行会被逐条分类。
# 当前测试样本的真实标签绝不能放进 prompt。

# %%
# split loader 会返回：
# - train_df：展示给 LLM 的 few-shot examples；
# - test_df：需要分类的测试样本。
train_df, test_df = load_fewshot_split(split_root, k)
feature_categories = load_feature_categories(feature_category_path)

print("train rows:", len(train_df))
print("test rows:", len(test_df))
print("feature categories:", len(feature_categories))

display(train_df.head(2))
display(test_df.head(2))


# %% [markdown]
# ## 3. 加载 RAG 检索器
#
# 如果 `kb.index` 和 `kb_embeddings.npy` 还不存在，这一步会根据 `kb_docs.jsonl`
# 自动构建它们。知识库不能包含测试样本或标签。

# %%
# `RagRetriever` 会检查知识库、加载或构建 FAISS，并准备 BGE embedding 模型。
# 如果模型已经下载过，设置 `local_files_only=True` 可以避免联网检查。
retriever = RagRetriever(
    kb_dir=kb_dir,
    model_name=embedding_model,
    batch_size=32,
    rebuild=False,
    local_files_only=local_files_only,
)

print("knowledge docs:", len(retriever.docs))
print("FAISS vectors:", retriever.index.ntotal)
print("first doc:", retriever.docs[0]["doc_id"])


# %% [markdown]
# ## 4. 阶段三：构造一个 RAG prompt
#
# 选一条测试样本。
# 先把它的 active Drebin features 转成 query，再检索 top-k 知识文档，
# 最后把这些知识插入最终 prompt。

# %%
sample_index = 0
sample_row = test_df.iloc[sample_index]

# 阶段三有三个关键小步骤：
# 1. 根据当前样本 active features 构造 query。
# 2. 检索 top-k 知识文档。
# 3. 把检索结果插入最终 LLM prompt。
query_text = build_query_text(sample_row, feature_categories)
hits = retriever.retrieve(query_text, top_k=top_k)
prompt = build_rag_prompt(
    train_df=train_df,
    test_row=sample_row,
    feature_categories=feature_categories,
    hits=hits,
)

print("true label is kept outside the prompt:", normalize_label(sample_row["class"]))
print("active feature count:", len(active_feature_names(sample_row)))
print("\nQUERY TEXT:\n")
print(query_text[:1500])

print("\nRETRIEVED DOCS:\n")
for hit in hits:
    doc = hit.doc
    print(f"{hit.rank}. score={hit.score:.4f} {doc['doc_id']} ({doc['doc_type']})")

print("\nPROMPT PREVIEW:\n")
print(prompt[:3000])


# %% [markdown]
# ## 5. 保存阶段三 prompt，方便人工检查
#
# 在正式调用 LLM 之前，最好先看一下保存的 prompt。重点检查：
#
# - Few-shot examples 里可以有标签。
# - 当前测试样本区域不能出现真实标签。
# - Retrieved knowledge 里不能出现 `true_label`、`pred_label` 或 `class`。

# %%
# 保存下来的 prompt 是最重要的调试材料。
# 如果 RAG 表现奇怪，先看 prompt，再考虑改代码或重跑完整测试集。
demo_prompt_dir = Path("results/rag_raw_llm/demo_prompts")
demo_prompt_dir.mkdir(parents=True, exist_ok=True)
demo_prompt_path = demo_prompt_dir / f"sample_{sample_index}_top{top_k}.txt"
demo_prompt_path.write_text(prompt, encoding="utf-8")

print("saved:", demo_prompt_path)


# %% [markdown]
# ## 6. 阶段四：烟雾测试命令
#
# 下面这个命令只跑 5 条样本。
# 它会生成 prompt、预测结果、metrics 和 retrieval logs。
# 确认 Ollama 已启动、模型名正确之后，再复制到终端运行。

# %%
# 这里只打印命令，不直接执行。
# 这样你可以先确认 Ollama 正在运行，再把命令复制到终端。
smoke_command = (
    "python llm_rag_raw_fewshot.py "
    f"--split-root {split_root} "
    f"--k {k} "
    f"--top-k {top_k} "
    "--max-test-samples 5"
)
if local_files_only:
    smoke_command += " --local-files-only"

print(smoke_command)


# %% [markdown]
# 如果只想保存 prompt 和 retrieval log，不想调用 LLM，就加 `--dry-run`。

# %%
dry_run_command = smoke_command + " --dry-run"
print(dry_run_command)


# %% [markdown]
# ## 7. 阶段五：完整运行命令
#
# 这个命令会跑完整测试集。
# 它和 LLM-only 实验使用同一个 split、k、模型、temperature 和 raw feature 格式。

# %%
# 这是完整实验命令，可能会比较久。
# 因为每条测试样本都需要一次检索和一次 LLM 调用。
full_command = (
    "python llm_rag_raw_fewshot.py "
    f"--split-root {split_root} "
    f"--k {k} "
    f"--top-k {top_k} "
    f"--model {llm_model} "
    f"--temperature {temperature}"
)
if local_files_only:
    full_command += " --local-files-only"

print(full_command)


# %% [markdown]
# ## 8. 运行后读取指标
#
# 跑完阶段四或阶段五之后，用这个 cell 查看保存的指标。

# %%
# 输出路径和 runner 使用同一套约定：
# results/rag_raw_llm/<split-name>/k<k>/
class Args:
    split_root = split_root
    k = k
    output_root = "results/rag_raw_llm"


rag_output_dir = output_dir_for(Args)
metrics_path = rag_output_dir / f"rag_raw_top{top_k}_metrics.json"
pred_path = rag_output_dir / f"rag_raw_top{top_k}.jsonl"
retrieval_log_path = rag_output_dir / "retrieval_logs.jsonl"

print("expected output dir:", rag_output_dir)
print("metrics exists:", metrics_path.exists(), metrics_path)
print("predictions exists:", pred_path.exists(), pred_path)
print("retrieval logs exists:", retrieval_log_path.exists(), retrieval_log_path)

if metrics_path.exists():
    import json

    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    selected = {
        "accuracy_strict": metrics.get("accuracy_strict"),
        "macro_f1": metrics.get("macro_f1"),
        "recall_malware": metrics.get("recall_malware"),
        "precision": metrics.get("precision"),
        "parse_ok_rate": metrics.get("parse_ok_rate"),
        "pred_S_ratio": metrics.get("pred_S_ratio"),
        "total": metrics.get("total"),
    }
    display(pd.DataFrame([selected]))


# %% [markdown]
# ## 9. 与 LLM-only 指标对比
#
# 如果已经跑过 `llm_only_fewshot.py`，把 `llm_only_metrics_path` 指向它的
# metrics JSON 文件。这里的对比表故意保持简洁，方便阅读。

# %%
# 这个对比 cell 故意写得很小。
# 写报告时，最值得和 LLM-only 对比的是：
# accuracy_strict、macro_f1、recall_malware、precision、parse_ok_rate、
# pred_S_ratio。
llm_only_metrics_path = Path("results/predictions/llm_only_seed42_test100_k5_metrics.json")

if metrics_path.exists() and llm_only_metrics_path.exists():
    import json

    rag_metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    llm_only_metrics = json.loads(llm_only_metrics_path.read_text(encoding="utf-8"))

    rows = []
    for name, data in [("Raw LLM-only", llm_only_metrics), ("Raw + RAG", rag_metrics)]:
        rows.append(
            {
                "experiment": name,
                "accuracy_strict": data.get("accuracy_strict"),
                "macro_f1": data.get("macro_f1"),
                "recall_malware": data.get("recall_malware"),
                "precision": data.get("precision"),
                "parse_ok_rate": data.get("parse_ok_rate"),
                "pred_S_ratio": data.get("pred_S_ratio"),
            }
        )
    display(pd.DataFrame(rows))
else:
    print("Run both experiments first, then rerun this cell.")
    print("RAG metrics:", metrics_path)
    print("LLM-only metrics:", llm_only_metrics_path)
