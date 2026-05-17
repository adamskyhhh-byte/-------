from __future__ import annotations
"""
传统机器学习少样本 baseline 脚本。

这个文件读取 prepare_fewshot_split.py 生成的 train.csv 和 test.csv，
用 5 个传统机器学习模型做二分类实验：
1. Logistic Regression
2. Linear SVM
3. Random Forest
4. Decision Tree
5. KNN

注意：这里不再重新 train_test_split，因为少样本实验要求使用固定划分：
训练只用 k*/train.csv，测试只用 split 根目录下的 test.csv。

常用命令：
python baseline_fewshot.py --split-root data/processed/fewshot_seed42_test100 --k 5
"""

import argparse
import json
import time
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier

from fewshot_utils import dataframe_to_xy, load_fewshot_split


def parse_args() -> argparse.Namespace:
    """读取命令行参数，例如 split-root、K、输出目录、随机种子。"""
    parser = argparse.ArgumentParser(description="Run fixed-split few-shot ML baselines.")
    parser.add_argument("--split-root", required=True)
    parser.add_argument("--k", type=int, required=True)
    parser.add_argument("--output-root", default="results/baseline")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def safe_name(name: str) -> str:
    """把模型名转换成适合文件名的写法。"""
    return name.lower().replace(" ", "_").replace("-", "_")


def count_labels(labels: pd.Series) -> dict[str, int]:
    """统计数字标签中的 B/S 数量。这里 0=B，1=S。"""
    b = int((labels == 0).sum())
    s = int((labels == 1).sum())
    return {"B": b, "S": s, "total": int(len(labels))}


def build_models(seed: int, n_train: int) -> dict[str, object]:
    """构建本次实验要比较的 5 个传统机器学习模型。"""
    # KNN 的邻居数不能超过训练样本数，否则会报错。
    # 同时尽量用奇数，减少投票平票的情况。
    n_neighbors = min(5, n_train)
    if n_neighbors > 1 and n_neighbors % 2 == 0:
        n_neighbors -= 1
    n_neighbors = max(1, n_neighbors)

    return {
        "Logistic Regression": Pipeline(
            [
                # 线性模型通常配合标准化更稳。
                ("scaler", StandardScaler()),
                (
                    "clf",
                    LogisticRegression(max_iter=1000, random_state=seed, class_weight="balanced"),
                ),
            ]
        ),
        "Linear SVM": Pipeline(
            [
                ("scaler", StandardScaler()),
                ("clf", LinearSVC(random_state=seed, class_weight="balanced", max_iter=10000)),
            ]
        ),
        # 树模型不需要标准化，直接吃 0/1 特征即可。
        "Random Forest": RandomForestClassifier(
            n_estimators=200, random_state=seed, class_weight="balanced"
        ),
        "Decision Tree": DecisionTreeClassifier(random_state=seed, class_weight="balanced"),
        "KNN": Pipeline(
            [
                ("scaler", StandardScaler()),
                ("clf", KNeighborsClassifier(n_neighbors=n_neighbors)),
            ]
        ),
    }


def score_for_auc(model: object, X_test: pd.DataFrame):
    """取出可用于计算 ROC-AUC 的分数。不同模型提供分数的接口不一样。"""
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X_test)[:, 1]
    if hasattr(model, "decision_function"):
        return model.decision_function(X_test)
    return None


def save_confusion_matrix(cm, model_name: str, output_dir: Path) -> None:
    """把混淆矩阵保存成图片，方便报告里展示。"""
    fig, ax = plt.subplots(figsize=(4.5, 4))
    ax.imshow(cm, interpolation="nearest", cmap="Blues")
    ax.set_title(model_name)
    ax.set_xticks([0, 1], labels=["B", "S"])
    ax.set_yticks([0, 1], labels=["B", "S"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", color="black")
    fig.tight_layout()
    fig.savefig(output_dir / f"confusion_matrix_{safe_name(model_name)}.png", dpi=160)
    plt.close(fig)


def evaluate_model(model_name: str, model: object, X_train, y_train, X_test, y_test, output_dir: Path):
    """训练单个模型、测试单个模型，并整理指标。"""
    # time.perf_counter() 适合测耗时，比普通 time.time() 更精确。
    start = time.perf_counter()
    model.fit(X_train, y_train)
    training_time = time.perf_counter() - start

    start = time.perf_counter()
    y_pred = model.predict(X_test)
    inference_time = time.perf_counter() - start
    inference_ms = inference_time * 1000 / max(1, len(X_test))

    # labels=[0, 1] 固定顺序：第 0 类是 B，第 1 类是 S。
    # ravel 后对应 TN, FP, FN, TP。
    cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
    tn, fp, fn, tp = [int(v) for v in cm.ravel()]
    save_confusion_matrix(cm, model_name, output_dir)

    auc_scores = score_for_auc(model, X_test)
    try:
        # 有些模型/极端数据可能无法算 AUC，这时保存 None，不让整个实验崩掉。
        roc_auc = float(roc_auc_score(y_test, auc_scores)) if auc_scores is not None else None
    except ValueError:
        roc_auc = None

    return {
        "model": model_name,
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, pos_label=1, zero_division=0)),
        "recall_malware": float(recall_score(y_test, y_pred, pos_label=1, zero_division=0)),
        "f1": float(f1_score(y_test, y_pred, pos_label=1, zero_division=0)),
        "macro_f1": float(f1_score(y_test, y_pred, average="macro", zero_division=0)),
        "roc_auc": roc_auc,
        "fpr": float(fp / (fp + tn)) if (fp + tn) else 0.0,
        "fnr": float(fn / (fn + tp)) if (fn + tp) else 0.0,
        "TP": tp,
        "TN": tn,
        "FP": fp,
        "FN": fn,
        "training_time_sec": float(training_time),
        "inference_time_ms_per_sample": float(inference_ms),
    }


def main() -> None:
    args = parse_args()
    split_root = Path(args.split_root)
    # 输出目录形如 results/baseline/fewshot_seed42_test100_k5/。
    output_dir = Path(args.output_root) / f"{split_root.name}_k{args.k}"
    output_dir.mkdir(parents=True, exist_ok=True)

    # 读取固定划分的数据，并转成 sklearn 能训练的 X/y。
    train_df, test_df = load_fewshot_split(split_root, args.k)
    X_train, y_train = dataframe_to_xy(train_df)
    X_test, y_test = dataframe_to_xy(test_df)

    # 打印实验设置，方便确认这次跑的是哪个 K、哪个 seed。
    train_size = count_labels(y_train)
    test_size = count_labels(y_test)
    print(f"K = {args.k}")
    print(f"seed = {args.seed}")
    print(f"train samples = B:{train_size['B']}, S:{train_size['S']}, total:{train_size['total']}")
    print(f"test samples = B:{test_size['B']}, S:{test_size['S']}, total:{test_size['total']}")

    rows = []
    for model_name, model in build_models(args.seed, len(X_train)).items():
        print(f"Training {model_name}...")
        rows.append(evaluate_model(model_name, model, X_train, y_train, X_test, y_test, output_dir))

    # 每个模型一行，保存成 CSV，后续可以直接放进论文/报告表格。
    metrics = pd.DataFrame(rows)
    metrics.to_csv(output_dir / "baseline_metrics.csv", index=False)

    # 保存本次运行的基本信息，便于以后复现实验。
    run_metadata = {
        "split_root": str(split_root),
        "k": args.k,
        "seed": args.seed,
        "train_size": train_size,
        "test_size": test_size,
        "models": list(build_models(args.seed, len(X_train)).keys()),
    }
    (output_dir / "run_metadata.json").write_text(
        json.dumps(run_metadata, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"Saved: {output_dir / 'baseline_metrics.csv'}")


if __name__ == "__main__":
    main()
