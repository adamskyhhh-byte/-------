"""
离线生成 Drebin 特征语义描述的脚本。

这个文件用于论文方法中的“语义映射”阶段：它只读取特征名和特征类别，
调用本地 Ollama/Gemma 为每个特征生成中性的中文短描述，并输出 JSON 与
人工 review CSV。注意：这里不读取样本标签，也不根据训练/测试取值生成描述，
避免把分类答案泄漏进特征语义。
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import pandas as pd

from fewshot_utils import load_feature_categories
from llm_only_fewshot import extract_json_object


PROMPT_TEMPLATE = """你正在为 Android 恶意软件检测论文实验生成“特征语义说明”。

请只根据下面的特征名和特征类别，写一句中性的中文短描述。

要求：
1. 描述长度尽量控制在 10 到 30 个中文字符左右。
2. 只解释 Android 行为含义，不做良性/恶意分类判断。
3. 不输出风险评分。
4. 不使用“恶意、可疑、危险、高危、攻击、木马、病毒、风险”等分类倾向词，除非特征名本身明确包含该含义。
5. 权限类可写成“请求/使用某权限”。
6. API 类可写成“调用某系统接口以...”。
7. Intent 类可写成“监听/响应某系统事件”。
8. Command 类可写成“执行/调用某命令能力”。

特征名：{feature}
特征类别：{category}

请严格返回一个 JSON 对象，不要 Markdown，不要解释：
{{"description":"中文短描述"}}
"""

FLAGGED_WORDS = [
    "恶意",
    "可疑",
    "危险",
    "高危",
    "攻击",
    "木马",
    "窃取",
    "勒索",
    "病毒",
    "风险",
    "malicious",
    "suspicious",
    "risky",
    "trojan",
    "attack",
]


def parse_args() -> argparse.Namespace:
    """读取命令行参数，控制输入文件、模型、重试次数和输出路径。"""
    parser = argparse.ArgumentParser(description="Generate neutral Chinese feature semantics.")
    parser.add_argument("--data-path", default="data/drebin-215-dataset-5560malware-9476-benign.csv")
    parser.add_argument("--feature-category-path", default="data/dataset-features-categories.csv")
    parser.add_argument("--model", default="gemma4:e4b")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--request-timeout", type=float, default=60.0)
    parser.add_argument("--max-retries", type=int, default=2)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--fallback-template", action="store_true")
    parser.add_argument(
        "--output-json",
        default="data/processed/paper_features/feature_semantics_gemma.json",
    )
    parser.add_argument(
        "--output-review-csv",
        default="data/processed/paper_features/feature_semantics_review.csv",
    )
    parser.add_argument(
        "--output-generation-prompt",
        default="data/processed/paper_features/feature_semantics_generation_prompt.txt",
    )
    return parser.parse_args()


def load_feature_names(data_path: str | Path, label_col: str = "class") -> list[str]:
    """从 Drebin 主 CSV 的表头读取特征名，并排除 class 标签列。"""
    columns = pd.read_csv(data_path, nrows=0).columns.tolist()
    return [str(column) for column in columns if str(column) != label_col]


def load_existing_semantics(path: Path, resume: bool) -> dict[str, dict[str, Any]]:
    """断点续写时读取已有 JSON；不续写或文件不存在则返回空字典。"""
    if not resume or not path.exists():
        return {}
    with path.open(encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"Existing semantics file is not a JSON object: {path}")
    return data


def call_ollama_json(model: str, prompt: str, temperature: float, request_timeout: float) -> str:
    """调用本地 Ollama 模型，尽量要求模型直接输出 JSON。"""
    import ollama

    client = ollama.Client(timeout=request_timeout)
    messages = [
        {
            "role": "system",
            "content": "You are a strict JSON API. Return one JSON object only.",
        },
        {"role": "user", "content": prompt},
    ]
    try:
        response = client.chat(
            model=model,
            messages=messages,
            format="json",
            options={"temperature": temperature},
        )
    except TypeError:
        response = client.chat(
            model=model,
            messages=messages,
            options={"temperature": temperature},
        )
    return response["message"]["content"]


def fallback_description(feature: str, category: str) -> str:
    """当 LLM 多次失败时，用模板生成不中断实验的中性描述。"""
    lowered_category = category.lower()
    if "permission" in lowered_category:
        return f"请求使用 {feature} 权限"
    if "intent" in lowered_category:
        return f"监听或响应 {feature} 事件"
    if "command" in lowered_category:
        return f"调用 {feature} 命令能力"
    if "api" in lowered_category:
        return f"调用 {feature} 系统接口"
    return f"使用 Android 特征 {feature}"


def flagged_words(description: str) -> list[str]:
    """检查描述中是否包含可能污染分类判断的关键词。"""
    lowered = description.lower()
    return [word for word in FLAGGED_WORDS if word.lower() in lowered]


def build_record(
    feature: str,
    category: str,
    description: str,
    source: str,
    raw_response: str,
    generation_prompt: str,
) -> dict[str, Any]:
    """按文档约定组装单个特征的语义记录。"""
    hits = flagged_words(description)
    return {
        "category": category,
        "description": description,
        "source": source,
        "review_status": "needs_review" if hits else "unchecked",
        "raw_response": raw_response,
        "generation_prompt": generation_prompt,
        "flagged_words": hits,
    }


def generate_one_feature(
    feature: str,
    category: str,
    args: argparse.Namespace,
) -> dict[str, Any]:
    """为单个特征生成语义描述，失败时按参数决定是否使用模板兜底。"""
    prompt = PROMPT_TEMPLATE.format(feature=feature, category=category)
    last_raw = ""
    last_error = ""

    for attempt in range(args.max_retries + 1):
        try:
            last_raw = call_ollama_json(args.model, prompt, args.temperature, args.request_timeout)
            data = extract_json_object(last_raw)
            description = str(data.get("description", "")).strip()
            if description:
                return build_record(
                    feature,
                    category,
                    description,
                    "gemma_offline",
                    last_raw,
                    prompt,
                )
            last_error = "JSON does not contain non-empty description"
        except Exception as exc:
            last_error = str(exc)
        print(f"[retry {attempt + 1}/{args.max_retries + 1}] {feature}: {last_error}")

    if not args.fallback_template:
        raise RuntimeError(f"Failed to generate description for {feature}: {last_error}")

    description = fallback_description(feature, category)
    return build_record(
        feature,
        category,
        description,
        "fallback_template",
        last_raw or last_error,
        prompt,
    )


def save_review_csv(path: Path, semantics: dict[str, dict[str, Any]]) -> None:
    """把语义描述导出成便于人工检查的 CSV。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["feature", "category", "description", "review_status", "flagged_words", "source"],
        )
        writer.writeheader()
        for feature in sorted(semantics):
            record = semantics[feature]
            writer.writerow(
                {
                    "feature": feature,
                    "category": record.get("category", ""),
                    "description": record.get("description", ""),
                    "review_status": record.get("review_status", ""),
                    "flagged_words": ";".join(record.get("flagged_words", [])),
                    "source": record.get("source", ""),
                }
            )


def main() -> None:
    args = parse_args()
    output_json = Path(args.output_json)
    output_review_csv = Path(args.output_review_csv)
    output_generation_prompt = Path(args.output_generation_prompt)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_generation_prompt.parent.mkdir(parents=True, exist_ok=True)

    feature_names = load_feature_names(args.data_path)
    feature_categories = load_feature_categories(args.feature_category_path)
    semantics = load_existing_semantics(output_json, args.resume)

    output_generation_prompt.write_text(PROMPT_TEMPLATE, encoding="utf-8")
    print(f"Loaded {len(feature_names)} features. Existing records: {len(semantics)}")

    for index, feature in enumerate(feature_names, start=1):
        if args.resume and feature in semantics and str(semantics[feature].get("description", "")).strip():
            print(f"[{index}/{len(feature_names)}] skip existing: {feature}")
            continue

        category = feature_categories.get(feature, "Uncategorized")
        print(f"[{index}/{len(feature_names)}] generating: {feature}")
        semantics[feature] = generate_one_feature(feature, category, args)

        # 每生成一个特征就落盘，长时间运行时更安全，也方便断点续写。
        output_json.write_text(json.dumps(semantics, indent=2, ensure_ascii=False), encoding="utf-8")

    output_json.write_text(json.dumps(semantics, indent=2, ensure_ascii=False), encoding="utf-8")
    save_review_csv(output_review_csv, semantics)
    print(f"Saved: {output_json}")
    print(f"Saved: {output_review_csv}")
    print(f"Saved: {output_generation_prompt}")


if __name__ == "__main__":
    main()
