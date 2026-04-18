import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import argparse
import gc
import json
from pathlib import Path

import easyocr
import numpy as np
import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoProcessor, AutoTokenizer
from peft import PeftModel

from base_architecture import (
    load_vision_model,
    Qwen3VLProjector,
    CustomVLM,
    DEVICE,
)
from baselines.multimodal_prompting_baselines import is_answer_correct_relaxed


def apply_ocr_prompt(example, reader, max_ocr_words=1000):
    image = example["image"]
    query = example["query"]

    if image.mode != "RGB":
        image = image.convert("RGB")
    img_np = np.array(image)

    results = reader.readtext(img_np)
    results = sorted(results, key=lambda x: x[0][0][1])

    lines = []
    word_count = 0
    for res in results:
        text = res[1]
        words_in_text = text.split()
        if word_count + len(words_in_text) <= max_ocr_words:
            lines.append(text)
            word_count += len(words_in_text)
        else:
            remaining_words = max_ocr_words - word_count
            if remaining_words > 0:
                lines.append(" ".join(words_in_text[:remaining_words]))
            break

    ocr_text = "\n".join(lines)

    prompt = (
        "<image>\n"
        f"You are given the image and its text extracted from a chart via OCR:\n{ocr_text}\n"
        "You are a chart question-answering assistant. Respond with the final answer only, no explanations.\n"
        f"Question: {query}\n"
        "Answer:"
    )

    label = example["label"][0] if isinstance(example["label"], list) else example["label"]
    return {
        "image": image,
        "prompt": prompt,
        "completion": str(label).strip(),
    }


def load_lora_vlm_from_checkpoint(ckpt_dir: Path, device: str):
    metadata_path = ckpt_dir / "vlm_metadata.json"
    adapter_dir = ckpt_dir / "language_model_adapter"
    tokenizer_dir = ckpt_dir / "tokenizer"
    processor_dir = ckpt_dir / "processor"
    projector_path = ckpt_dir / "projector.pt"

    if not adapter_dir.exists():
        raise FileNotFoundError(f"Missing adapter dir: {adapter_dir}")
    if not projector_path.exists():
        raise FileNotFoundError(f"Missing projector weights: {projector_path}")
    if not metadata_path.exists():
        raise FileNotFoundError(f"Missing metadata file: {metadata_path}")

    with open(metadata_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    vision_model_name = metadata["vision_model_name"]
    language_model_name = metadata["language_model_name"]
    spatial_merge_size = metadata["spatial_merge_size"]
    image_token_id = metadata["image_token_id"]

    print(f"Loading metadata from: {metadata_path}")
    print(f"Vision model: {vision_model_name}")
    print(f"Base language model: {language_model_name}")
    print(f"Spatial merge size: {spatial_merge_size}")

    vision_model, _ = load_vision_model(vision_model_name)
    v_conf = vision_model.config.vision_config if hasattr(vision_model.config, "vision_config") else vision_model.config
    vision_dim = v_conf.hidden_size

    if tokenizer_dir.exists():
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir, trust_remote_code=True)
    else:
        tokenizer = AutoTokenizer.from_pretrained(language_model_name, trust_remote_code=True)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if processor_dir.exists():
        processor = AutoProcessor.from_pretrained(processor_dir, trust_remote_code=True)
    else:
        processor = AutoProcessor.from_pretrained(vision_model_name, trust_remote_code=True)

    base_llm = AutoModelForCausalLM.from_pretrained(
    language_model_name,
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
)

    target_vocab_size = len(tokenizer)
    current_vocab_size = base_llm.get_input_embeddings().weight.shape[0]

    print(f"Tokenizer vocab size: {target_vocab_size}")
    print(f"Base model vocab size before resize: {current_vocab_size}")

    if current_vocab_size != target_vocab_size:
        print(f"Resizing base model embeddings from {current_vocab_size} to {target_vocab_size}")
        base_llm.resize_token_embeddings(target_vocab_size)
        if hasattr(base_llm, "tie_weights"):
            base_llm.tie_weights()

    print(f"Base model vocab size after resize: {base_llm.get_input_embeddings().weight.shape[0]}")

    language_model = PeftModel.from_pretrained(
        base_llm,
        adapter_dir,
        is_trainable=False,
    )

    llm_dim = language_model.config.hidden_size

    projector = Qwen3VLProjector(
        vision_dim=vision_dim,
        llm_dim=llm_dim,
        spatial_merge_size=spatial_merge_size,
    ).to(device=device, dtype=torch.bfloat16)

    state_dict = torch.load(projector_path, map_location="cpu", weights_only=False)
    projector.load_state_dict(state_dict)

    model = CustomVLM(
        vision_model=vision_model,
        processor=processor,
        language_model=language_model,
        tokenizer=tokenizer,
        projector=projector,
        image_token_id=image_token_id,
    )

    model.to(device)
    model.eval()
    return model, tokenizer, processor


@torch.no_grad()
def evaluate_test_set(model, test_ds, tokenizer, processor, max_new_tokens=32, print_examples=5):
    model.eval()
    correct = 0
    total = len(test_ds)
    shown = 0

    for sample in tqdm(test_ds, desc="OCR ChartQA Eval"):
        messages = [{"role": "user", "content": sample["prompt"]}]
        prompt_text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        ground_truth = str(sample["completion"]).strip()

        image = sample["image"]
        if image.mode != "RGB":
            image = image.convert("RGB")

        pixel_values = processor(images=image, return_tensors="pt").pixel_values.to(
            DEVICE, dtype=torch.bfloat16
        )
        inputs = tokenizer(prompt_text, return_tensors="pt", add_special_tokens=False).to(DEVICE)
        attention_mask = torch.ones_like(inputs.input_ids).to(DEVICE)

        prediction_text = model.generate_text(
            input_ids=inputs.input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            max_new_tokens=max_new_tokens,
        )
        prediction = prediction_text.strip()

        if shown < print_examples:
            print("=" * 80)
            print("PROMPT:")
            print(prompt_text)
            print(f"GT   : {ground_truth}")
            print(f"PRED : {prediction}")
            shown += 1

        if is_answer_correct_relaxed(prediction, ground_truth):
            correct += 1

        del pixel_values, inputs, attention_mask, prediction_text

    torch.cuda.empty_cache()
    accuracy = 100.0 * correct / total if total > 0 else 0.0
    print(f"\nFinal relaxed accuracy: {accuracy:.2f}% ({correct}/{total})")
    return accuracy


def main():
    parser = argparse.ArgumentParser(description="Evaluate saved LoRA VLM checkpoint on ChartQA with OCR prompts only")
    parser.add_argument("--ckpt_dir", type=str, required=True,
                        help="Checkpoint dir containing language_model_adapter/, tokenizer/, processor/, projector.pt, vlm_metadata.json")
    parser.add_argument("--split", type=str, default="test", choices=["test", "val", "train"])
    parser.add_argument("--max_examples", type=int, default=None)
    parser.add_argument("--max_new_tokens", type=int, default=32)
    parser.add_argument("--max_ocr_words", type=int, default=2000)
    args = parser.parse_args()

    ckpt_dir = Path(args.ckpt_dir)

    print("Loading ChartQA...")
    raw_ds = load_dataset("HuggingFaceM4/ChartQA")

    print("Initializing EasyOCR...")
    reader = easyocr.Reader(["en"], gpu=torch.cuda.is_available())

    split_ds = raw_ds[args.split]
    if args.max_examples is not None:
        split_ds = split_ds.select(range(min(args.max_examples, len(split_ds))))

    print(f"Precomputing OCR prompts for split={args.split}, n={len(split_ds)}")
    eval_ds = split_ds.map(
        lambda ex: apply_ocr_prompt(ex, reader, max_ocr_words=args.max_ocr_words),
        desc=f"Extracting OCR for {args.split}",
    )

    del reader
    gc.collect()
    torch.cuda.empty_cache()

    print("Loading saved LoRA VLM checkpoint...")
    model, tokenizer, processor = load_lora_vlm_from_checkpoint(ckpt_dir, DEVICE)

    print("Running evaluation...")
    evaluate_test_set(
        model=model,
        test_ds=eval_ds,
        tokenizer=tokenizer,
        processor=processor,
        max_new_tokens=args.max_new_tokens,
    )


if __name__ == "__main__":
    main()