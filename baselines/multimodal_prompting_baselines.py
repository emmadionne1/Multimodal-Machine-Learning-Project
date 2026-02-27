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

def load_model_and_processor(model_name):
    if model_name == "google/matcha-chartqa":
        processor = Pix2StructProcessor.from_pretrained(model_name)
        model = Pix2StructForConditionalGeneration.from_pretrained(model_name)
    else:
        processor = AutoProcessor.from_pretrained(model_name)
        model = AutoModelForImageTextToText.from_pretrained(model_name, dtype=torch.float16, device_map="auto")
    return model, processor

def load_hf_dataset(dataset_name):
    # dataset: https://huggingface.co/datasets/HuggingFaceM4/ChartQA
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

def generate_answer(question, image, model, processor, examples):
    try:
        message = create_message(question, image, examples)
        prompt = processor.apply_chat_template(message, tokenize=False, add_generation_prompt=True)
    except:
        prompt = question

    inputs = processor(text=[prompt], images=[ex["image"] for ex in examples]+[image], return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    gen = model.generate(**inputs, max_new_tokens=32, do_sample=False)
    try:
        return processor.tokenizer.decode(gen[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True).strip()
    except:
        return processor.decode(gen[0], skip_special_tokens=True).strip()

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

def run_eval(model, processor, eval_dataset, examples):
    correct, total = 0, 0
    for ex in tqdm(eval_dataset):
        pred = generate_answer(ex["query"], ex["image"], model, processor, examples)
        correct += is_answer_correct(pred, ex["label"][0])
        total += 1
    print(f"Relaxed accuracy: {correct/total}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment_type", type=str, required=True, choices=['zero', 'few'])
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-VL-3B-Instruct")
    parser.add_argument("--dataset_name", type=str, default="HuggingFaceM4/ChartQA")
    args = parser.parse_args()

    # set up experiment parameters
    EXPERIMENT_TYPE = args.experiment_type
    MODEL_NAME = args.model_name
    DATASET_NAME = args.dataset_name

    # load model, processor
    model, processor = load_model_and_processor(MODEL_NAME)
    model.eval()

    # load val dataset, fewshot examples
    ds = load_hf_dataset(DATASET_NAME)["val"]
    examples = get_fewshot_examples(ds) if EXPERIMENT_TYPE == "few" else []

    # compute relaxed accuracy over dataset
    run_eval(model, processor, ds, examples)