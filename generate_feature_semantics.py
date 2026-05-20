from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import pandas as pd

from fewshot_utils import load_feature_categories
from llm_only_fewshot import extract_json_object


PROMPT_TEMPLATE = """You describe one Android Drebin feature for a classification prompt.

Requirements:
- Use neutral, technical wording.
- Explain what the feature is or what Android capability it relates to.
- Do not claim the feature proves malware or benign behavior.
- Avoid risk-loaded words such as malicious, suspicious, risky, trojan, attack.
- Keep the description concise.

Feature: {feature}
Category: {category}
Training-pool statistic: {stat_direction}

Return exactly one JSON object and no Markdown:
{{"description":"neutral technical description"}}
"""

FLAGGED_WORDS = [
    "malicious",
    "suspicious",
    "risky",
    "trojan",
    "attack",
    "malware",
    "dangerous",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate neutral Drebin feature semantics.")
    parser.add_argument("--data-path", default="data/drebin-215-dataset-5560malware-9476-benign.csv")
    parser.add_argument("--feature-category-path", default="data/dataset-features-categories.csv")
    parser.add_argument("--feature-stats", default=None)
    parser.add_argument("--style", choices=["legacy", "neutral-fixed"], default="legacy")
    parser.add_argument("--model", default="gemma4:e4b")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--request-timeout", type=float, default=60.0)
    parser.add_argument("--num-ctx", type=int, default=12288)
    parser.add_argument("--max-retries", type=int, default=2)
    parser.add_argument("--max-feature-batch", type=int, default=1)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--fallback-template", action="store_true")
    parser.add_argument("--output-json", default="data/processed/paper_features/feature_semantics_gemma.json")
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
    columns = pd.read_csv(data_path, nrows=0).columns.tolist()
    return [str(column) for column in columns if str(column) != label_col]


def load_feature_stats(path: str | Path | None) -> dict[str, dict[str, Any]]:
    if path is None:
        return {}
    path = Path(path)
    if not path.exists():
        return {}
    df = pd.read_csv(path)
    if "feature" not in df.columns:
        return {}
    return {str(row["feature"]): row.to_dict() for _, row in df.iterrows()}


def load_existing_semantics(path: Path, resume: bool) -> dict[str, dict[str, Any]]:
    if not resume or not path.exists():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Existing semantics file is not a JSON object: {path}")
    return data


def call_ollama_json(
    model: str,
    prompt: str,
    temperature: float,
    request_timeout: float,
    num_ctx: int,
) -> str:
    import ollama

    client = ollama.Client(timeout=request_timeout)
    messages = [
        {"role": "system", "content": "You are a strict JSON API. Return one JSON object only."},
        {"role": "user", "content": prompt},
    ]
    try:
        response = client.chat(
            model=model,
            messages=messages,
            format="json",
            options={"temperature": temperature, "num_ctx": num_ctx},
        )
    except TypeError:
        response = client.chat(
            model=model,
            messages=messages,
            options={"temperature": temperature, "num_ctx": num_ctx},
        )
    return response["message"]["content"]


def fallback_description(feature: str, category: str) -> str:
    lowered = category.lower()
    if "permission" in lowered:
        return f"Android permission or capability represented by {feature}."
    if "intent" in lowered:
        return f"Android intent or broadcast-related feature {feature}."
    if "command" in lowered:
        return f"System command or shell-related feature {feature}."
    if "api" in lowered:
        return f"Android or Java API usage feature {feature}."
    return f"Android Drebin feature {feature}."


def flagged_words(description: str) -> list[str]:
    lowered = description.lower()
    return [word for word in FLAGGED_WORDS if word in lowered]


def build_record(
    feature: str,
    category: str,
    description: str,
    source: str,
    raw_response: str,
    generation_prompt: str,
    stats: dict[str, Any],
) -> dict[str, Any]:
    hits = flagged_words(description)
    return {
        "category": category,
        "description": description,
        "source": source,
        "review_status": "needs_review" if hits else "unchecked",
        "raw_response": raw_response,
        "generation_prompt": generation_prompt,
        "flagged_words": hits,
        "stat_direction": stats.get("stat_direction"),
        "p_feature_given_B": stats.get("p_feature_given_B"),
        "p_feature_given_S": stats.get("p_feature_given_S"),
        "log_odds_S_vs_B": stats.get("log_odds_S_vs_B"),
    }


def generate_one_feature(
    feature: str,
    category: str,
    stats: dict[str, Any],
    args: argparse.Namespace,
) -> dict[str, Any]:
    prompt = PROMPT_TEMPLATE.format(
        feature=feature,
        category=category,
        stat_direction=str(stats.get("stat_direction", "unknown")),
    )
    last_raw = ""
    last_error = ""

    for attempt in range(args.max_retries + 1):
        try:
            last_raw = call_ollama_json(
                args.model,
                prompt,
                args.temperature,
                args.request_timeout,
                args.num_ctx,
            )
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
                    stats,
                )
            last_error = "JSON does not contain non-empty description"
        except Exception as exc:
            last_error = str(exc)
        print(f"[retry {attempt + 1}/{args.max_retries + 1}] {feature}: {last_error}")

    if args.style == "neutral-fixed":
        return build_record(
            feature,
            category,
            "unknown (generation failed)",
            "generation_failed",
            last_raw or last_error,
            prompt,
            stats,
        )

    if not args.fallback_template:
        raise RuntimeError(f"Failed to generate description for {feature}: {last_error}")

    return build_record(
        feature,
        category,
        fallback_description(feature, category),
        "fallback_template",
        last_raw or last_error,
        prompt,
        stats,
    )


def save_review_csv(path: Path, semantics: dict[str, dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "feature",
        "category",
        "description",
        "review_status",
        "flagged_words",
        "source",
        "stat_direction",
    ]
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
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
                    "stat_direction": record.get("stat_direction", ""),
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
    feature_stats = load_feature_stats(args.feature_stats)
    semantics = load_existing_semantics(output_json, args.resume)
    failures: list[dict[str, Any]] = []

    output_generation_prompt.write_text(PROMPT_TEMPLATE, encoding="utf-8")
    print(f"Loaded {len(feature_names)} features. Existing records: {len(semantics)}")

    for index, feature in enumerate(feature_names, start=1):
        if args.resume and feature in semantics and str(semantics[feature].get("description", "")).strip():
            print(f"[{index}/{len(feature_names)}] skip existing: {feature}")
            continue

        category = feature_categories.get(feature, "Uncategorized")
        print(f"[{index}/{len(feature_names)}] generating: {feature}")
        semantics[feature] = generate_one_feature(
            feature,
            category,
            feature_stats.get(feature, {}),
            args,
        )
        if semantics[feature].get("source") == "generation_failed":
            failures.append({"feature": feature, "category": category})

        output_json.write_text(json.dumps(semantics, indent=2, ensure_ascii=False), encoding="utf-8")

    output_json.write_text(json.dumps(semantics, indent=2, ensure_ascii=False), encoding="utf-8")
    if failures:
        failures_path = output_json.parent / "generation_failures.json"
        failures_path.write_text(json.dumps(failures, indent=2, ensure_ascii=False), encoding="utf-8")
    save_review_csv(output_review_csv, semantics)
    print(f"Saved: {output_json}")
    print(f"Saved: {output_review_csv}")
    print(f"Saved: {output_generation_prompt}")


if __name__ == "__main__":
    main()
