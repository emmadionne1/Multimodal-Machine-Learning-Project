"""
multimodal_prompting_baselines
script to run zero-shot and few-shot baselines for a small VLM on the ChartQA dataset
models to run: Qwen/Qwen3-VL-4B-Instruct, Qwen/Qwen2.5-VL-3B-Instruct
"""

from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoProcessor, AutoModelForImageTextToText, Pix2StructProcessor, Pix2StructForConditionalGeneration
import argparse
import math
import re
import torch
import os
import csv
from datetime import datetime

def load_model_and_processor(model_name):
    if model_name == "google/matcha-chartqa":
        processor = Pix2StructProcessor.from_pretrained(model_name)
        model = Pix2StructForConditionalGeneration.from_pretrained(model_name)
    else:
        processor = AutoProcessor.from_pretrained(model_name)
        model = AutoModelForImageTextToText.from_pretrained(model_name, dtype=torch.float16, device_map="auto")
    return model, processor

def load_hf_dataset(dataset_name):
    return load_dataset(dataset_name)

def get_fewshot_examples(dataset):
    examples = []
    for i, ex in enumerate(dataset):
        if i > 2:
            break
        examples.append({"image": ex["image"], "question": ex["query"], "answer": ex["label"][0]})
    return examples

def create_message(question, image, examples, answer=None):
    system_prompt = "You are a chart question-answering assistant. Respond with the final answer only, no explanations."
    user_content = []
    
    if len(examples) > 0:
        user_content.append({"type": "text", "text": "Here are some examples:\n"})
        for example in examples:
            user_content.append({"type": "image", "image": example["image"]})
            user_content.append({"type": "text", "text": f"Question: {example['question']}\nAnswer: {example['answer']}\n"})
        user_content.append({"type": "text", "text": "Now answer this question:\n"})
    
    user_content.append({"type": "image", "image": image})
    user_content.append({"type": "text", "text": f"Question: {question}\nAnswer:"})

    if answer is None:
        return [
            {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
            {"role": "user", "content": user_content}
        ]
    else:
        return [
            {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": [{"type": "text", "text": answer}]}
        ]

def extract_answer(text):
    text = str(text).strip()
    if not text:
        return ""
    if "Answer:" in text:
        text = text.split("Answer:", 1)[-1]
    text = text.strip()
    lines = text.splitlines()
    if len(lines) == 0:
        return ""
    text = lines[0].strip()
    text = re.sub(r"^(final answer|final|answer)\s*[:\-]\s*", "", text, flags=re.IGNORECASE).strip()
    return text

def generate_answer(question, image, model, processor, examples, max_new_tokens=32):
    try:
        message = create_message(question, image, examples)
        prompt = processor.apply_chat_template(message, tokenize=False, add_generation_prompt=True)
    except:
        prompt = question

    inputs = processor(text=[prompt], images=[ex["image"] for ex in examples]+[image], return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    gen = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    try:
        text = processor.tokenizer.decode(gen[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True).strip()
    except:
        text = processor.decode(gen[0], skip_special_tokens=True).strip()

    return extract_answer(text)

def normalize_answer(s):
    s = str(s).lower().strip()
    s = re.sub(r"\s+", " ", s)
    return s

def to_float(s):
    s = str(s).replace(",", "").strip()
    if s.endswith("%"):
        try:
            return float(s[:-1]) / 100.0
        except:
            return None
    try:
        return float(s)
    except:
        return None

def is_answer_correct_relaxed(pred, gold):
    pred_norm, gold_norm = normalize_answer(pred), normalize_answer(gold)
    pred_num, gold_num = to_float(pred_norm), to_float(gold_norm)

    if pred_num is not None and gold_num is not None:
        if math.isclose(gold_num, 0.0):
            return 1.0 if math.isclose(pred_num, 0.0) else 0.0
        rel_err = abs(pred_num - gold_num) / abs(gold_num)
        return float(rel_err <= 0.05)

    return float(pred_norm == gold_norm)

def is_answer_correct_exact(pred, gold):
    pred_norm, gold_norm = normalize_answer(pred), normalize_answer(gold)
    pred_num, gold_num = to_float(pred_norm), to_float(gold_norm)

    if pred_num is not None and gold_num is not None:
        return float(pred_num == gold_num)

    return float(pred_norm == gold_norm)

def levenshtein_distance(a, b):
    a = str(a)
    b = str(b)
    if a == b:
        return 0
    if len(a) == 0:
        return len(b)
    if len(b) == 0:
        return len(a)

    if len(a) < len(b):
        a, b = b, a

    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            ins = cur[j - 1] + 1
            dele = prev[j] + 1
            sub = prev[j - 1] + (ca != cb)
            cur.append(min(ins, dele, sub))
        prev = cur
    return prev[-1]

def is_text_answer(pred, gold):
    pred_norm, gold_norm = normalize_answer(pred), normalize_answer(gold)
    return (to_float(pred_norm) is None) and (to_float(gold_norm) is None)

def safe_name(s):
    s = str(s)
    s = s.replace("/", "__")
    s = re.sub(r"[^a-zA-Z0-9_\-\.]+", "_", s)
    return s.strip("_")

def run_eval(model, processor, eval_dataset, examples, model_name, dataset_name, split_name, experiment_type, max_new_tokens=32, out_dir="preds_vlm"):
    correct_relaxed, correct_exact, total = 0.0, 0.0, 0

    lev_sum = 0.0
    lev_n = 0

    os.makedirs(out_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    fname = f"vlm_preds_{safe_name(dataset_name)}_{safe_name(split_name)}_{safe_name(experiment_type)}_{safe_name(model_name)}_k{len(examples)}_t{max_new_tokens}_{ts}.csv"
    path = os.path.join(out_dir, fname)

    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["idx", "question", "gold", "pred", "relaxed_correct", "exact_correct", "is_text", "levenshtein"])

        for i, ex in enumerate(tqdm(eval_dataset)):
            pred = generate_answer(ex["query"], ex["image"], model, processor, examples, max_new_tokens=max_new_tokens)
            gold = str(ex["label"][0]).strip()

            c_relaxed = is_answer_correct_relaxed(pred, gold)
            c_exact = is_answer_correct_exact(pred, gold)

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

    relaxed_acc = correct_relaxed / max(total, 1)
    exact_acc = correct_exact / max(total, 1)
    lev_avg = (lev_sum / lev_n) if lev_n > 0 else float("nan")

    print(f"Relaxed accuracy: {relaxed_acc:.4f}")
    print(f"Exact accuracy: {exact_acc:.4f}")
    print(f"Levenshtein avg (text only): {lev_avg:.4f}")
    print(f"Saved: {path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment_type", type=str, required=True, choices=['zero', 'few'])
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-VL-3B-Instruct")
    parser.add_argument("--dataset_name", type=str, default="HuggingFaceM4/ChartQA")
    parser.add_argument("--max_new_tokens", type=int, default=32)
    parser.add_argument("--out_dir", type=str, default="preds_vlm")
    args = parser.parse_args()

    EXPERIMENT_TYPE = args.experiment_type
    MODEL_NAME = args.model_name
    DATASET_NAME = args.dataset_name
    MAX_NEW_TOKENS = args.max_new_tokens

    model, processor = load_model_and_processor(MODEL_NAME)
    model.eval()

    dataset = load_hf_dataset(DATASET_NAME)
    if "val" in dataset:
        split_name = "val"
        ds = dataset["val"]
    elif "validation" in dataset:
        split_name = "validation"
        ds = dataset["validation"]
    else:
        split_name = "test"
        ds = dataset["test"]

    examples = get_fewshot_examples(ds) if EXPERIMENT_TYPE == "few" else []

    run_eval(
        model,
        processor,
        ds,
        examples,
        model_name=MODEL_NAME,
        dataset_name=DATASET_NAME,
        split_name=split_name,
        experiment_type=EXPERIMENT_TYPE,
        max_new_tokens=MAX_NEW_TOKENS,
        out_dir=args.out_dir
    )