"""
multimodal_prompting_baselines (TEXT-ONLY)
script to run zero-shot and few-shot baselines for an LLM on the ChartQA dataset
models to run: any text LLM, e.g. Qwen/Qwen3-4B-Instruct-2507
"""

from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
import argparse
import math
import re
import torch

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

def load_hf_dataset(dataset_name):
    # dataset: https://huggingface.co/datasets/HuggingFaceM4/ChartQA
    return load_dataset(dataset_name)

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

def generate_answer(question, model, tokenizer, examples):
    prompt = create_prompt(question, examples)

    inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=True)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    gen = model.generate(
        **inputs,
        max_new_tokens=32,
        do_sample=False,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )

    # decode only newly generated tokens (like your VLM code tries to do)
    new_tokens = gen[0][inputs["input_ids"].shape[-1]:]
    text = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
    return text

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

def is_answer_correct(pred, gold):
    pred_norm, gold_norm = normalize_answer(pred), normalize_answer(gold)
    pred_num, gold_num = to_float(pred_norm), to_float(gold_norm)

    # numerical: 5% diff allowed
    if pred_num is not None and gold_num is not None:
        if math.isclose(gold_num, 0.0):
            return 1.0 if math.isclose(pred_num, 0.0) else 0.0
        rel_err = abs(pred_num - gold_num) / abs(gold_num)
        return float(rel_err <= 0.05)

    # non-numerical: needs an exact string match, case insensitive
    return float(pred_norm == gold_norm)

def run_eval(model, tokenizer, eval_dataset, examples):
    correct, total = 0, 0
    for ex in tqdm(eval_dataset):
        pred = generate_answer(ex["query"], model, tokenizer, examples)
        correct += is_answer_correct(pred, ex["label"][0])
        total += 1
    print(f"Relaxed accuracy: {correct/total}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment_type", type=str, required=True, choices=['zero', 'few'])
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen3-4B-Instruct-2507")
    parser.add_argument("--dataset_name", type=str, default="HuggingFaceM4/ChartQA")
    args = parser.parse_args()

    # set up experiment parameters
    EXPERIMENT_TYPE = args.experiment_type
    MODEL_NAME = args.model_name
    DATASET_NAME = args.dataset_name

    # load model, tokenizer
    model, tokenizer = load_model_and_tokenizer(MODEL_NAME)
    model.eval()

    # load val dataset, fewshot examples
    dataset = load_hf_dataset(DATASET_NAME)
    if "val" in dataset:
        ds = dataset["val"]
    elif "validation" in dataset:
        ds = dataset["validation"]
    else:
        ds = dataset["test"]

    examples = get_fewshot_examples(ds) if EXPERIMENT_TYPE == "few" else []

    # compute relaxed accuracy over dataset
    run_eval(model, tokenizer, ds, examples)