"""实时检查 LLM 实验进度。"""
import json
from pathlib import Path

ROOT = Path("results/final_50")

def main() -> None:
    print("=== feature_expr_llm progress ===")
    for k_dir in sorted((ROOT / "feature_expr_llm" / "seed42_test100").glob("k*")):
        K = k_dir.name
        for expr in ("raw", "semantic-risky-old", "semantic-neutral-fixed"):
            jsonl = k_dir / f"{expr}_full.jsonl"
            prompt_dir = k_dir / "prompts" / f"{expr}_full"
            n_jsonl = sum(1 for _ in jsonl.open(encoding="utf-8")) if jsonl.exists() else 0
            n_prompt = sum(1 for _ in prompt_dir.iterdir()) if prompt_dir.exists() else 0
            print(f"  {K}/{expr}: jsonl={n_jsonl}/50, prompts={n_prompt}")

    print("\n=== rag_raw_llm progress ===")
    for k_dir in sorted((ROOT / "rag_raw_llm" / "seed42_test100").glob("k*")):
        K = k_dir.name
        jsonl = k_dir / "rag_raw_bucketed_f2_r3.jsonl"
        prompt_dir = k_dir / "prompts"
        n_jsonl = sum(1 for _ in jsonl.open(encoding="utf-8")) if jsonl.exists() else 0
        n_prompt = sum(1 for _ in prompt_dir.iterdir()) if prompt_dir.exists() else 0
        print(f"  {K}: jsonl={n_jsonl}/50, prompts={n_prompt}")

    # 已有 metrics 文件
    print("\n=== finalized metrics files ===")
    for f in sorted(ROOT.glob("**/*_metrics.json")):
        try:
            data = json.loads(f.read_text(encoding="utf-8"))
            acc = data.get("accuracy_strict") or data.get("accuracy")
            mf1 = data.get("macro_f1")
            ps = data.get("pred_S_ratio")
            print(f"  {f.relative_to(ROOT)}: acc={acc}, macro_f1={mf1}, pred_S_ratio={ps}")
        except Exception as exc:
            print(f"  {f.relative_to(ROOT)}: error {exc}")


if __name__ == "__main__":
    main()
