"""
LLM-only 少样本分类脚本。

这个文件读取同一套少样本划分数据：
1. k*/train.csv 作为 few-shot examples，全部写进 prompt。
2. test.csv 作为待预测样本，每次让本地 Ollama 模型预测一个样本。
3. 要求模型输出严格 JSON，并把每个样本的预测结果保存到 JSONL。
4. 最后统计 parse_ok_rate、accuracy_strict、macro_f1 等指标。

这里的 “LLM-only” 意思是：只靠 prompt 里的少样本示例和当前样本特征，
不加入 RAG 检索规则，方便后续和 LLM+RAG 公平对比。

常用命令：
python llm_only_fewshot.py --split-root data/processed/fewshot_seed42_test100 --k 5 --max-test-samples 5
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score

from fewshot_utils import (
    LABEL_TO_ID,
    load_feature_categories,
    load_fewshot_split,
    normalize_label,
    row_to_feature_text,
)


# 额外放一个 system prompt，把模型角色压成“只返回 JSON 的接口”。
# Ollama 的部分模型即使 user prompt 写得很严格，也可能继续输出 Markdown；
# system prompt + format="json" 能显著减少这种情况。
SYSTEM_PROMPT = (
    "You are a strict JSON API for binary Android malware classification. "
    "Return one JSON object only. Do not use Markdown. Do not explain outside JSON."
)

# 这些是 Gemma 常见的不规范字段名。解析器会把它们尽量转成 pred_label。
LABEL_FIELD_NAMES = {
    "pred_label",
    "prediction",
    "label",
    "classification",
    "class",
    "result",
    "verdict",
    "decision",
    "answer",
}
CONCLUSION_FIELD_NAMES = {"conclusion", "final", "summary"}
EVIDENCE_FIELD_NAMES = {
    "evidence",
    "indicators",
    "suspicious_indicators",
    "malicious_indicators",
    "features",
}
EXPLANATION_FIELD_NAMES = {
    "explanation",
    "reason",
    "reasoning",
    "malicious_reasoning",
    "classification_reasoning",
    "conclusion",
}


def parse_args() -> argparse.Namespace:
    """读取命令行参数，例如模型名、温度、最多测试多少条。"""
    parser = argparse.ArgumentParser(description="Run LLM-only few-shot Drebin classification.")
    parser.add_argument("--split-root", required=True)
    parser.add_argument("--k", type=int, required=True)
    parser.add_argument("--model", default="gemma4:e4b")
    parser.add_argument("--feature-category-path", default="data/dataset-features-categories.csv")
    parser.add_argument("--output-root", default="results/predictions")
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--request-timeout", type=float, default=60.0)
    parser.add_argument("--max-test-samples", type=int, default=None)
    parser.add_argument(
        "--reparse-jsonl",
        default=None,
        help="只重新解析已有 JSONL 的 raw 字段，不重新调用 LLM。",
    )
    return parser.parse_args()


def build_prompt(train_df, test_row, feature_categories: dict[str, str]) -> str:
    """把 few-shot 示例和当前测试样本拼成一次 LLM 请求的 prompt。"""
    examples = []
    for idx, row in train_df.iterrows():
        # few-shot 示例必须带真实标签 B/S，让模型学习这次任务怎么判断。
        label = normalize_label(row["class"])
        features = row_to_feature_text(row, feature_categories=feature_categories)
        examples.append(f"Example {idx}\nLabel: {label}\n{features}")

    # 当前测试样本只给特征，不给真实标签，避免答案泄漏。
    test_features = row_to_feature_text(test_row, feature_categories=feature_categories)
    return (
        "Classify the CURRENT TEST SAMPLE only.\n\n"
      "Allowed labels:\n"
      "- B means benign / normal / non-malicious.\n"
      "- S means malware / malicious / suspicious.\n\n"
      "Important mapping rule:\n"
      "- If your internal answer is benign, output \"pred_label\":\"B\".\n"
      "- If your internal answer is malware, malicious, or suspicious, output \"pred_label\":\"S\".\n\n"
      "General reasoning hint:\n"
      "- Consider the combination of active features rather than any single feature.\n\n"
      "Few-shot examples:\n"
      + "\n\n".join(examples)
      + "\n\nCURRENT TEST SAMPLE FEATURES:\n"
      + test_features
      + "\n\nReturn EXACTLY ONE JSON object and nothing else.\n"
      "The first character must be { and the last character must be }.\n"
      "Do not use Markdown code fences.\n"
      "Do not use keys such as label, prediction, classification, malicious, or analysis.\n"
      "Use exactly these keys: pred_label, evidence, explanation, confidence.\n"
      "pred_label must be exactly one of: \"B\", \"S\".\n"
      "evidence must be a JSON array of active feature names from the current test sample.\n"
      "confidence must be a number from 0 to 1.\n\n"
      "Valid output example for benign:\n"
      "{\"pred_label\":\"B\",\"evidence\":[\"FEATURE_NAME\"],\"explanation\":\"short reason\",\"confidence\":0.50}\n"
      "Valid output example for malware:\n"
      "{\"pred_label\":\"S\",\"evidence\":[\"FEATURE_NAME\"],\"explanation\":\"short reason\",\"confidence\":0.80}"
    )


def call_ollama(model: str, prompt: str, temperature: float, request_timeout: float) -> str:
    """调用本地 Ollama 模型，并返回模型生成的原始文本。"""
    import ollama

    client = ollama.Client(timeout=request_timeout)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ]

    try:
        # Ollama 支持 format="json" 时，会更强地约束模型返回 JSON。
        response = client.chat(
            model=model,
            messages=messages,
            format="json",
            options={"temperature": temperature},
        )
    except TypeError:
        # 如果本机 ollama Python 包版本较旧，不支持 format 参数，就退回普通调用。
        response = client.chat(
            model=model,
            messages=messages,
            options={"temperature": temperature},
        )
    return response["message"]["content"]


def extract_json_object(text: str) -> dict[str, Any]:
    """从模型输出里提取第一个 JSON 对象。"""
    stripped = text.strip()
    try:
        data = json.loads(stripped)
        if isinstance(data, dict):
            return data
    except json.JSONDecodeError:
        pass

    # 比正则更稳：从每个 { 开始尝试 JSONDecoder，能处理 ```json 前缀。
    decoder = json.JSONDecoder()
    for start, char in enumerate(text):
        if char != "{":
            continue
        try:
            data, _ = decoder.raw_decode(text[start:])
        except json.JSONDecodeError:
            continue
        if isinstance(data, dict):
            return data

    raise ValueError("No JSON object found")


def infer_label_from_text(value: Any) -> str | None:
    """从非标准文本里尽量推断 B/S。这个函数只用于 LLM 不守格式时兜底。"""
    if value is None:
        return None

    text = str(value).strip()
    if not text:
        return None

    lowered = text.lower()
    normalized = re.sub(r"[^a-z0-9]+", " ", lowered).strip()

    benign_exact = {"b", "benign", "normal", "clean", "safe", "legitimate", "non malicious"}
    malware_exact = {
        "s",
        "malware",
        "malicious",
        "suspicious",
        "suspicious malicious",
        "privacy invasive",
        "spyware",
        "trojan",
    }
    if normalized in benign_exact:
        return "B"
    if normalized in malware_exact:
        return "S"

    # 纯文本回答常见形式：Malicious / **Malicious** / Benign。
    starts = re.sub(r"^[\s`*_#>\-]+", "", lowered)
    if starts.startswith(("benign", "normal", "clean", "safe", "legitimate")):
        return "B"
    if starts.startswith(("malicious", "malware", "suspicious", "spyware", "trojan")):
        return "S"

    # 先处理否定式，避免 "not malicious" 被后面的 malicious 规则误判为 S。
    if re.search(r"\b(not|non|no)\s+malicious\b", normalized):
        return "B"

    label_words = r"(prediction|predicted|label|classification|classified|classify|verdict|decision|answer|conclusion)"
    benign_words = r"(benign|normal|clean|safe|legitimate|non[-\s]?malicious)"
    malware_words = r"(malware|malicious|suspicious|privacy[-\s]?invasive|spyware|trojan)"

    if re.search(label_words + r".{0,80}" + benign_words, lowered, flags=re.DOTALL):
        return "B"
    if re.search(label_words + r".{0,80}" + malware_words, lowered, flags=re.DOTALL):
        return "S"

    # 最后的弱兜底：如果全文只出现一边的明显关键词，就按那一边处理。
    benign_hits = len(re.findall(r"\b(benign|normal|clean|safe|legitimate)\b", normalized))
    malware_hits = len(
        re.findall(r"\b(malware|malicious|suspicious|spyware|trojan)\b", normalized)
    )
    if benign_hits > 0 and malware_hits == 0:
        return "B"
    if malware_hits > 0 and benign_hits == 0:
        return "S"
    return None


def infer_label_from_value(key: str, value: Any) -> str | None:
    """根据字段名和值推断标签，比如 malicious=true 或 prediction=benign。"""
    key_normalized = re.sub(r"[^a-z0-9]+", "_", key.lower()).strip("_")

    if isinstance(value, bool):
        if key_normalized in {"malicious", "malware", "is_malicious", "is_malware"}:
            return "S" if value else "B"
        if key_normalized in {"benign", "is_benign"}:
            return "B" if value else "S"

    if isinstance(value, str):
        return infer_label_from_text(value)
    return None


def find_first_by_key(data: Any, field_names: set[str]) -> Any:
    """递归查找 JSON 里第一个指定字段名的值。"""
    if isinstance(data, dict):
        for key, value in data.items():
            if key.lower() in field_names:
                return value
        for value in data.values():
            found = find_first_by_key(value, field_names)
            if found is not None:
                return found
    elif isinstance(data, list):
        for item in data:
            found = find_first_by_key(item, field_names)
            if found is not None:
                return found
    return None


def infer_label_from_json(data: Any) -> str | None:
    """优先从 JSON 的标签字段推断；不行再看布尔字段、结论字段和解释字段。"""
    if isinstance(data, dict):
        # 第一优先级：prediction / label / classification 这类明显标签字段。
        for key, value in data.items():
            if key.lower() in LABEL_FIELD_NAMES:
                label = infer_label_from_value(key, value)
                if label:
                    return label

        # 第二优先级：malicious=true / benign=false 这类布尔字段。
        for key, value in data.items():
            if isinstance(value, bool):
                label = infer_label_from_value(key, value)
                if label:
                    return label

        # 第三优先级：conclusion/final/summary 往往比中间分析更可靠。
        for key, value in data.items():
            if key.lower() in CONCLUSION_FIELD_NAMES:
                label = infer_label_from_text(value)
                if label:
                    return label

        # 第四优先级：递归看嵌套对象，例如 analysis.conclusion。
        for value in data.values():
            if isinstance(value, (dict, list)):
                label = infer_label_from_json(value)
                if label:
                    return label

        # 最后再看解释字段。这个优先级低，是为了避免“不是 malicious”被误读。
        for key, value in data.items():
            if key.lower() in EXPLANATION_FIELD_NAMES:
                label = infer_label_from_value(key, value)
                if label:
                    return label

    if isinstance(data, list):
        for item in data:
            label = infer_label_from_json(item)
            if label:
                return label
    return None


def normalize_evidence(value: Any) -> list[str]:
    """把 evidence / indicators 统一整理成字符串列表。"""
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item) for item in value]
    return [str(value)]


def normalize_confidence(value: Any) -> float | None:
    """把模型输出的 confidence 转成 0~1 的数字，转不了就返回 None。"""
    try:
        confidence = float(value) if value is not None else None
    except (TypeError, ValueError):
        return None
    if confidence is None:
        return None
    if confidence > 1:
        confidence = confidence / 100
    return max(0.0, min(1.0, confidence))


def strict_json_prediction_label(raw: str) -> str | None:
    """检查模型是否严格按要求输出了四字段 JSON，并返回其中的 B/S 标签。"""
    try:
        data = json.loads(raw.strip())
    except json.JSONDecodeError:
        return None

    required_keys = {"pred_label", "evidence", "explanation", "confidence"}
    if not isinstance(data, dict) or set(data.keys()) != required_keys:
        return None
    if not isinstance(data.get("evidence"), list):
        return None
    if not isinstance(data.get("explanation"), str):
        return None
    if normalize_confidence(data.get("confidence")) is None:
        return None

    try:
        return normalize_label(data.get("pred_label"))
    except ValueError:
        return None


def parse_model_output(raw: str) -> dict[str, Any]:
    """解析模型原始输出，标准 JSON 和常见非标准输出都会尽量转成 B/S。"""
    strict_label = strict_json_prediction_label(raw)
    try:
        data = extract_json_object(raw)
        parse_source = "json"
    except Exception:
        data = {}
        parse_source = "text_fallback"

    pred_label = infer_label_from_json(data) or infer_label_from_text(raw)
    if pred_label is None:
        raise ValueError("Could not infer B/S label from model output")

    strict_parse_ok = strict_label is not None
    if strict_parse_ok:
        pred_label = strict_label
        parse_source = "strict_json"
    elif parse_source == "json":
        parse_source = "recovered_json"
    else:
        parse_source = "recovered_text"

    evidence = normalize_evidence(find_first_by_key(data, EVIDENCE_FIELD_NAMES))
    explanation_value = find_first_by_key(data, EXPLANATION_FIELD_NAMES)
    explanation = str(explanation_value) if explanation_value is not None else ""
    confidence = normalize_confidence(find_first_by_key(data, {"confidence"}))

    return {
        "pred_label": pred_label,
        "evidence": evidence,
        "explanation": explanation,
        "confidence": confidence,
        "parse_source": parse_source,
        "strict_parse_ok": strict_parse_ok,
    }


def metrics_from_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    """根据 JSONL 记录计算 LLM-only 的评估指标。"""
    total = len(records)
    parse_ok = [record for record in records if record.get("parse_ok")]
    strict_parse_ok = [record for record in records if record.get("strict_parse_ok")]
    recovered_parse = [
        record for record in records if record.get("parse_ok") and not record.get("strict_parse_ok")
    ]
    parse_source_counts: dict[str, int] = {}
    for record in records:
        source = str(record.get("parse_source", "failed"))
        parse_source_counts[source] = parse_source_counts.get(source, 0) + 1
    y_true = [LABEL_TO_ID[record["true_label"]] for record in records]

    strict_pred = []
    valid_true = []
    valid_pred = []
    for record in records:
        true_id = LABEL_TO_ID[record["true_label"]]
        pred_label = record.get("pred_label")
        if pred_label in LABEL_TO_ID:
            pred_id = LABEL_TO_ID[pred_label]
            valid_true.append(true_id)
            valid_pred.append(pred_id)
            strict_pred.append(pred_id)
        else:
            # 解析失败：strict 指标里按“预测错了”处理，避免忽略失败样本导致虚高。
            strict_pred.append(1 - true_id)

    cm = confusion_matrix(y_true, strict_pred, labels=[0, 1]) if total else [[0, 0], [0, 0]]
    tn, fp, fn, tp = [int(v) for v in cm.ravel()]

    return {
        "parse_ok_rate": float(len(parse_ok) / total) if total else 0.0,
        "strict_parse_ok_rate": float(len(strict_parse_ok) / total) if total else 0.0,
        "recovered_parse_rate": float(len(recovered_parse) / total) if total else 0.0,
        "accuracy_strict": float(accuracy_score(y_true, strict_pred)) if total else 0.0,
        "accuracy_valid_only": (
            float(accuracy_score(valid_true, valid_pred)) if valid_true else None
        ),
        "precision": float(precision_score(y_true, strict_pred, pos_label=1, zero_division=0))
        if total
        else 0.0,
        "recall_malware": float(recall_score(y_true, strict_pred, pos_label=1, zero_division=0))
        if total
        else 0.0,
        "f1": float(f1_score(y_true, strict_pred, pos_label=1, zero_division=0)) if total else 0.0,
        "macro_f1": float(f1_score(y_true, strict_pred, average="macro", zero_division=0))
        if total
        else 0.0,
        "TP": tp,
        "TN": tn,
        "FP": fp,
        "FN": fn,
        "total": total,
        "parse_ok": len(parse_ok),
        "strict_parse_ok": len(strict_parse_ok),
        "recovered_parse": len(recovered_parse),
        "parse_source_counts": parse_source_counts,
    }


def build_record_from_raw(idx: int, true_label: str, raw: str) -> dict[str, Any]:
    """把一次模型原始输出整理成 JSONL 记录。"""
    try:
        parsed = parse_model_output(raw)
        return {
            "idx": int(idx),
            "true_label": true_label,
            "pred_label": parsed["pred_label"],
            "parse_ok": True,
            "evidence": parsed["evidence"],
            "explanation": parsed["explanation"],
            "confidence": parsed["confidence"],
            "parse_source": parsed["parse_source"],
            "strict_parse_ok": parsed["strict_parse_ok"],
            "raw": raw,
        }
    except Exception as exc:
        return {
            "idx": int(idx),
            "true_label": true_label,
            "pred_label": None,
            "parse_ok": False,
            "strict_parse_ok": False,
            "error": str(exc),
            "raw": raw or str(exc),
        }


def main() -> None:
    args = parse_args()
    split_root = Path(args.split_root)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    train_df, test_df = load_fewshot_split(split_root, args.k)
    feature_categories = load_feature_categories(args.feature_category_path)

    # 调试时可以只跑前几条，避免一次完整 LLM 实验耗时太久。
    if args.max_test_samples is not None:
        test_df = test_df.head(args.max_test_samples).copy()

    stem = f"llm_only_{split_root.name.replace('fewshot_', '')}_k{args.k}"
    if args.reparse_jsonl:
        stem = f"{stem}_reparsed"
    jsonl_path = output_root / f"{stem}.jsonl"
    metrics_path = output_root / f"{stem}_metrics.json"

    records: list[dict[str, Any]] = []
    if args.reparse_jsonl:
        with open(args.reparse_jsonl, encoding="utf-8") as source, jsonl_path.open(
            "w", encoding="utf-8"
        ) as target:
            for line_no, line in enumerate(source):
                old_record = json.loads(line)
                true_label = normalize_label(old_record["true_label"])
                record = build_record_from_raw(
                    idx=int(old_record.get("idx", line_no)),
                    true_label=true_label,
                    raw=str(old_record.get("raw", "")),
                )
                records.append(record)
                target.write(json.dumps(record, ensure_ascii=False) + "\n")
                print(f"[reparse {line_no + 1}] true={true_label} pred={record['pred_label']}")

        metrics = metrics_from_records(records)
        metrics.update(
            {
                "split_root": str(split_root),
                "k": args.k,
                "model": args.model,
                "temperature": args.temperature,
                "request_timeout": args.request_timeout,
                "max_test_samples": args.max_test_samples,
                "reparse_jsonl": args.reparse_jsonl,
            }
        )
        metrics_path.write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"Saved: {jsonl_path}")
        print(f"Saved: {metrics_path}")
        return

    with jsonl_path.open("w", encoding="utf-8") as handle:
        for out_idx, (row_idx, row) in enumerate(test_df.iterrows()):
            true_label = normalize_label(row["class"])
            prompt = build_prompt(train_df, row, feature_categories)
            raw = ""
            try:
                raw = call_ollama(args.model, prompt, args.temperature, args.request_timeout)
                record = build_record_from_raw(int(row_idx), true_label, raw)
            except Exception as exc:
                # 调用失败或解析失败都不会中断整个实验，而是记为 parse_ok=false。
                record = {
                    "idx": int(row_idx),
                    "true_label": true_label,
                    "pred_label": None,
                    "parse_ok": False,
                    "error": str(exc),
                    "raw": raw or str(exc),
                }

            records.append(record)
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            print(f"[{out_idx + 1}/{len(test_df)}] true={true_label} pred={record['pred_label']}")

    metrics = metrics_from_records(records)
    metrics.update(
        {
            "split_root": str(split_root),
            "k": args.k,
            "model": args.model,
            "temperature": args.temperature,
            "request_timeout": args.request_timeout,
            "max_test_samples": args.max_test_samples,
        }
    )
    metrics_path.write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Saved: {jsonl_path}")
    print(f"Saved: {metrics_path}")


if __name__ == "__main__":
    main()
