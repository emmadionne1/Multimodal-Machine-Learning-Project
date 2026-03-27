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
import evaluate
from rouge_score import rouge_scorer
from qwen_vl_utils import process_vision_info


bleu = evaluate.load("bleu")

def collate_fn(batch):
    # This keeps the batch as a list of dictionaries 
    # so we can process them as a group later.
    return batch

def load_model_and_processor(model_name):
    if model_name == "google/matcha-chartqa":
        processor = Pix2StructProcessor.from_pretrained(model_name)
        model = Pix2StructForConditionalGeneration.from_pretrained(model_name)
    else:
        processor = AutoProcessor.from_pretrained(model_name)
        model = AutoModelForImageTextToText.from_pretrained(model_name, 
                                                            dtype=torch.bfloat16, 
                                                            device_map="auto",
                                                            attn_implementation="flash_attention_2")
        processor.tokenizer.padding_side = "left"
    return model, processor

def load_hf_dataset(name, files):
    return load_dataset(name, files)

def get_fewshot_examples(dataset):
    examples = []
    for i, ex in enumerate(dataset):
        if i > 2:
            break
        examples.append({"image": ex["images"][0], "question": ex["texts"][0]["user"], "answer": ex["texts"][0]["assistant"]})
    return examples

def create_message(question, image, examples, answer=None):
    system_prompt = "You are a chart summarization assistant. Respond with the final explanation of the chart, in a few sentences or less"
    user_content = []
    
    if len(examples) > 0:
        user_content.append({"type": "text", "text": "Here are some examples:\n"})
        for example in examples:
            user_content.append({"type": "image", "image": example["image", "max_pixels": 768 * 768]})
            user_content.append({"type": "text", "text": f"Question: {example['question']}\nAnswer: {example['answer']}\n"})
        user_content.append({"type": "text", "text": "Now answer this question:\n"})
    
    user_content.append({"type": "image", "image": image, "max_pixels": 768 * 768})
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
    message = create_message(question, image, examples)
    prompt = processor.apply_chat_template(message, tokenize=False, add_generation_prompt=True)

    image_inputs, _ = process_vision_info(message)
    inputs = processor(text=[prompt], images=image_inputs, padding=True, return_tensors="pt").to(model.device)
    # inputs = {k: v.to(model.dvice) for k, v in inputs.items()}

    gen = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, gen)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]

    return extract_answer(output_text)

def normalize_answer(s):
    s = str(s).lower().strip()
    s = re.sub(r"\s+", " ", s)
    return s

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

def safe_name(s):
    s = str(s)
    s = s.replace("/", "__")
    s = re.sub(r"[^a-zA-Z0-9_\-\.]+", "_", s)
    return s.strip("_")

def run_eval(model, processor, eval_dataset, examples, scorer, model_name, dataset_name, split_name, experiment_type, max_new_tokens=32, out_dir="preds_vlm", batch_size=4):
    total = 0
    lev_sum = 0.0
    bleu_sum = 0.0
    rouge_l_f1 = 0.0
    rouge_l_precision = 0.0
    rouge_l_recall = 0.0

    os.makedirs(out_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    fname = f"vlm_preds_{safe_name(dataset_name)}_{safe_name(split_name)}_{safe_name(experiment_type)}_{safe_name(model_name)}_k{len(examples)}_t{max_new_tokens}_{ts}.csv"
    path = os.path.join(out_dir, fname)

    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["idx", "question", "gold", "pred", "bleu_score", "levenshtein", "rouge_l_f1", "rouge_l_precision", "rouge_l_recall"])

        dataloader = DataLoader(
            eval_dataset, 
            batch_size=batch_size, 
            shuffle=False, 
            collate_fn=collate_fn
        )

        for batch in tqdm(dataloader, desc="Evaluating"):
            batch_prompts = []
            batch_images = []
            batch_golds = []
            batch_questions = []

            for ex in batch:
                question = ex["texts"][0]["user"]
                image = ex["images"][0]
                gold = ex["texts"][0]["assistant"]
                
                # Use your existing create_message logic
                messages = create_message(question, image, examples)
                prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                
                # Map vision info for this specific message
                image_inputs, _ = process_vision_info(messages)
                
                batch_prompts.append(prompt)
                batch_images.append(image_inputs) # image_inputs is a list of pixel data
                batch_golds.append(gold)
                batch_questions.append(question)
            flat_images = [img for sublist in batch_images for img in sublist]
            inputs = processor(
                    text=batch_prompts, 
                    images=flat_images, 
                    padding=True, 
                    return_tensors="pt"
                ).to(model.device)
            
            with torch.no_grad():
                gen = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
            batch_preds = processor.batch_decode(gen[:, inputs.input_ids.shape[-1]:], skip_special_tokens=True)

            for i, (pred, gold, question) in enumerate(zip(batch_preds, batch_golds, batch_questions)):                
                if i < 2:
                    continue
                if len(str(pred).strip()) == 0:
                    bleu_score = 0.0
                else:
                    bleu_score = bleu.compute(predictions=[pred], references=[gold])["bleu"]
                bleu_sum += bleu_score
                total += 1

                rouge_l_score = scorer.score(pred, gold)['rougeL']
                rouge_l_f1 += rouge_l_score.fmeasure
                rouge_l_precision += rouge_l_score.precision
                rouge_l_recall += rouge_l_score.recall

                lev = levenshtein_distance(normalize_answer(pred), normalize_answer(gold))
                lev_sum += lev

                w.writerow([i, question, gold, pred, bleu_score, lev, rouge_l_score.fmeasure, rouge_l_score.precision, rouge_l_score.recall])


    lev_avg = lev_sum / total
    bleu_avg = bleu_sum / total
    rouge_l_f1_avg = rouge_l_f1 / total
    rouge_l_precision_avg = rouge_l_precision / total
    rouge_l_recall_avg = rouge_l_recall / total

    print(f"BLEU: {bleu_avg:.4f}")
    print(f"Levenshtein avg: {lev_avg:.4f}")
    print(f"Rouge-L F1 avg: {rouge_l_f1_avg:.4f}")
    print(f"Rouge-L Precision avg: {rouge_l_precision_avg:.4f}")
    print(f"Rouge-L Recall avg: {rouge_l_recall_avg:.4f}")
    print(f"Saved: {path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment_type", type=str, required=True, choices=['zero', 'few'])
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen3-VL-4B-Instruct")
    parser.add_argument("--dataset_name", type=str, default="chart2text")
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--out_dir", type=str, default="preds_vlm")
    args = parser.parse_args()

    EXPERIMENT_TYPE = args.experiment_type
    MODEL_NAME = args.model_name
    DATASET_NAME = args.dataset_name
    MAX_NEW_TOKENS = args.max_new_tokens

    model, processor = load_model_and_processor(MODEL_NAME)
    model.eval()

    if args.dataset == "chart2text":
        dataset = load_hf_dataset("HuggingFaceM4/the_cauldron", "chart2text")
    else:
        raise Exception
    if "val" in dataset:
        split_name = "val"
        ds = dataset["val"]
    elif "validation" in dataset:
        split_name = "validation"
        ds = dataset["validation"]
    elif "test" in dataset:
        split_name = "test"
        ds = dataset["test"]
    else:
        split_name = "train"
        ds = dataset["train"]
        shuffled_dataset = ds.shuffle(seed=42)
        num_samples = round(len(ds) * 0.1)
        ds = shuffled_dataset.select(range(num_samples))
        print(f"Sampled {len(ds)} data points")


    examples = get_fewshot_examples(ds) if EXPERIMENT_TYPE == "few" else []
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)


    run_eval(
        model,
        processor,
        ds,
        examples,
        scorer,
        model_name=MODEL_NAME,
        dataset_name=DATASET_NAME,
        split_name="10_percent",
        experiment_type=EXPERIMENT_TYPE,
        max_new_tokens=MAX_NEW_TOKENS,
        out_dir=args.out_dir
    )
