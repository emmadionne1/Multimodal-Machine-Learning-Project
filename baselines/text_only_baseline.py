"""
multimodal_prompting_baselines (TEXT-ONLY)
script to run zero-shot and few-shot baselines for an LLM on the ChartQA dataset
models to run: any text LLM, e.g. Qwen/Qwen3-4B-Instruct-2507
"""

from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from multimodal_prompting_baselines import levenshtein_distance, load_hf_dataset, extract_answer, safe_name, normalize_answer, is_text_answer, is_answer_correct_relaxed, is_answer_correct_exact
import argparse
import re
import torch
import os
import csv
from datetime import datetime

def load_model_and_tokenizer(model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=torch.float16 if torch.cuda.is_available() else None,
        device_map="auto" if torch.cuda.is_available() else None,
    )
    return model, tokenizer

def get_fewshot_examples(dataset):
    examples = []
    for i, ex in enumerate(dataset):
        if i > 2:
            break
        examples.append({"question": ex["query"], "answer": ex["label"][0]})
    return examples

def create_prompt(question, examples, answer=None):
    system_prompt = "You are a chart question-answering assistant. Respond with the final answer only, no explanations."

    parts = []
    parts.append(f"System: {system_prompt}")

    if len(examples) > 0:
        parts.append("Here are some examples:")
        for example in examples:
            parts.append(f"Question: {example['question']}\nAnswer: {example['answer']}")
        parts.append("Now answer this question:")

    parts.append(f"Question: {question}\nAnswer:")

    if answer is not None:
        parts.append(str(answer))

    return "\n".join(parts)

def generate_answer(question, model, tokenizer, examples, max_new_tokens=32):
    prompt = create_prompt(question, examples)

    inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=True)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    gen = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )

    new_tokens = gen[0][inputs["input_ids"].shape[-1]:]
    if new_tokens.numel() == 0:
        return ""

    text = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
    return extract_answer(text)

def run_eval(model, tokenizer, eval_dataset, examples, model_name, dataset_name, split_name, experiment_type, max_new_tokens=32, print_n=20, print_every=0, out_dir="preds"):
    correct_relaxed, correct_exact, total = 0.0, 0.0, 0
    n_empty = 0

    lev_sum = 0.0
    lev_n = 0

    os.makedirs(out_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    fname = f"preds_{safe_name(dataset_name)}_{safe_name(split_name)}_{safe_name(experiment_type)}_{safe_name(model_name)}_k{len(examples)}_t{max_new_tokens}_{ts}.csv"
    path = os.path.join(out_dir, fname)

    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["idx", "question", "gold", "pred", "relaxed_correct", "exact_correct", "is_text", "levenshtein"])

        for i, ex in enumerate(tqdm(eval_dataset)):
            pred = generate_answer(ex["query"], model, tokenizer, examples, max_new_tokens=max_new_tokens)
            gold = str(ex["label"][0]).strip()

            c_relaxed = is_answer_correct_relaxed(pred, gold)
            c_exact = is_answer_correct_exact(pred, gold)

            if pred == "":
                n_empty += 1

            correct_relaxed += c_relaxed
            correct_exact += c_exact
            total += 1

            is_text = is_text_answer(pred, gold)
            lev = ""
            if is_text:
                lev = levenshtein_distance(normalize_answer(pred), normalize_answer(gold))
                lev_sum += lev
                lev_n += 1

            w.writerow([i, ex["query"], gold, pred, c_relaxed, c_exact, int(is_text), lev])

            if i < print_n or (print_every and (i + 1) % print_every == 0):
                print(f"\n[{i}]")
                print(f"Q: {ex['query']}")
                print(f"GOLD: {gold}")
                print(f"PRED: {pred}")
                print(f"EMPTY: {int(pred == '')}")
                print(f"RELAXED_CORRECT: {int(c_relaxed)}")
                print(f"EXACT_CORRECT: {int(c_exact)}")
                if is_text:
                    print(f"LEV: {lev}")
                print(f"RUNNING_RELAXED_ACC: {correct_relaxed/total:.4f}")
                print(f"RUNNING_EXACT_ACC: {correct_exact/total:.4f}")
                if lev_n > 0:
                    print(f"RUNNING_LEV_AVG: {lev_sum/lev_n:.4f}")
                print(f"N_EMPTY: {n_empty}")

    relaxed_acc = correct_relaxed / max(total, 1)
    exact_acc = correct_exact / max(total, 1)
    lev_avg = (lev_sum / lev_n) if lev_n > 0 else float("nan")

    print(f"Relaxed accuracy: {relaxed_acc:.4f}")
    print(f"Exact accuracy: {exact_acc:.4f}")
    print(f"Levenshtein avg (text only): {lev_avg:.4f}")
    print(f"Empty preds: {n_empty}/{total}")
    print(f"Saved: {path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment_type", type=str, required=True, choices=['zero', 'few'])
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen3-4B-Instruct-2507")
    parser.add_argument("--dataset_name", type=str, default="HuggingFaceM4/ChartQA")
    parser.add_argument("--max_new_tokens", type=int, default=32)
    parser.add_argument("--print_n", type=int, default=20)
    parser.add_argument("--print_every", type=int, default=0)
    parser.add_argument("--out_dir", type=str, default="preds")
    args = parser.parse_args()

    EXPERIMENT_TYPE = args.experiment_type
    MODEL_NAME = args.model_name
    DATASET_NAME = args.dataset_name
    MAX_NEW_TOKENS = args.max_new_tokens

    model, tokenizer = load_model_and_tokenizer(MODEL_NAME)
    model.eval()

    dataset = load_hf_dataset(DATASET_NAME)
    ds = dataset["val"]
    examples = get_fewshot_examples(ds) if EXPERIMENT_TYPE == "few" else []

    run_eval(
        model,
        tokenizer,
        ds,
        examples,
        model_name=MODEL_NAME,
        dataset_name=DATASET_NAME,
        split_name="val",
        experiment_type=EXPERIMENT_TYPE,
        max_new_tokens=MAX_NEW_TOKENS,
        print_n=args.print_n,
        print_every=args.print_every,
        out_dir=args.out_dir
    )