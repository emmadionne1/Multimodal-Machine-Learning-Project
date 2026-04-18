from pathlib import Path
import argparse
import gc
import json
import math
import os
import random
import zipfile
from pprint import pprint

import easyocr
import evaluate
import numpy as np
import torch
import wandb
from datasets import DatasetDict, load_dataset
from huggingface_hub import snapshot_download
from peft import LoraConfig, TaskType, get_peft_model
from PIL import Image
from tqdm import tqdm
from transformers import (
    AutoModel,
    AutoModelForCausalLM,
    AutoProcessor,
    AutoTokenizer,
    Trainer,
    TrainerCallback,
    TrainingArguments,
)

import config
from baselines.multimodal_prompting_baselines import is_answer_correct_relaxed
from peft import LoraConfig, TaskType, get_peft_model, PeftModel

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 8
BLEU = evaluate.load("bleu")
DEFAULT_PROJECTOR_WEIGHTS = (
    "/home/bsood/Multimodal-Machine-Learning-Project/outputs/"
    "bhavesh-epoch3 checkpoint-1770/projector.pt"
)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def convert_chartqa_example(ex):
    answer = ex["label"][0]
    prompt = (
        "<image>\n"
        "You are a chart question-answering assistant. Respond with the final answer only, no explanations.\n"
        f"Question: {ex['query'].strip()}\n"
        "Answer:"
    )
    return {"image": ex["image"], "prompt": prompt, "completion": str(answer).strip()}


def create_dataset(overfit_check=False, dataset_num_splits=5, dataset_split_index=-1, pretraining=True):
    if not pretraining:
        raw = load_dataset("HuggingFaceM4/ChartQA")

        eval_split = "val" if "val" in raw else "validation" if "validation" in raw else "test"
        test_split = "test" if "test" in raw else eval_split

        if overfit_check:
            train_raw = raw["train"].select(range(3))
            eval_raw = train_raw
            test_raw = train_raw
        else:
            train_raw = raw["train"]
            eval_raw = raw[eval_split]
            test_raw = raw[test_split]

        ds = DatasetDict(
            {
                "train": train_raw.map(convert_chartqa_example),
                "eval": eval_raw.map(convert_chartqa_example),
                "test": test_raw.map(convert_chartqa_example),
            }
        )
        print(ds)
        print(ds["train"][0])
        return ds, None

    snapshot_dir = Path(
        snapshot_download(
            repo_id="liuhaotian/LLaVA-Pretrain",
            repo_type="dataset",
            allow_patterns=["blip_laion_cc_sbu_558k.json", "images.zip"],
        )
    )
    ds = load_dataset("json", data_files=str(snapshot_dir / "blip_laion_cc_sbu_558k.json"), split="train")

    image_root = snapshot_dir / "images"
    if image_root.exists() and any(image_root.rglob("*.jpg")):
        print("images have already been unzipped")
    else:
        image_root.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(snapshot_dir / "images.zip", "r") as zf:
            zf.extractall(image_root)
        print("Extraction complete")

    for _, v in ds[0].items():
        pprint(v)
    img = Image.open((snapshot_dir / "images") / ds[0]["image"]).convert("RGB")
    print(f"Image size: {img.size}")

    ds = ds.map(
        lambda ex: {
            "image": ex["image"],
            "prompt": ex["conversations"][0]["value"].strip(),
            "completion": ex["conversations"][1]["value"].strip(),
        },
        remove_columns=["id", "conversations"],
    )
    print(ds[0])

    split_1 = ds.train_test_split(test_size=0.10, seed=SEED, shuffle=True)
    split_2 = split_1["test"].train_test_split(test_size=0.50, seed=SEED, shuffle=True)
    ds = DatasetDict({"train": split_1["train"], "eval": split_2["train"], "test": split_2["test"]})
    print(ds)

    if 0 <= dataset_split_index < dataset_num_splits:
        train_len = len(ds["train"])
        chunk_size = (train_len + dataset_num_splits - 1) // dataset_num_splits
        start = dataset_split_index * chunk_size
        end = min(start + chunk_size, train_len)
        print(f"Selecting train chunk {dataset_split_index + 1}/{dataset_num_splits}")
        ds["train"] = ds["train"].select(range(start, end))
        print(ds)

    if overfit_check:
        ds["train"] = ds["train"].select(range(3))
        ds["eval"] = ds["train"]
        ds["test"] = ds["train"]
        print(ds)

    return ds, image_root


def load_vision_model(vision_model_name):
    print(f"Loading vision tower: {vision_model_name}")
    processor = AutoProcessor.from_pretrained(vision_model_name, use_fast=True)
    vision_model = AutoModel.from_pretrained(
        vision_model_name,
        dtype=torch.bfloat16,
    )
    vision_model.eval()
    return vision_model.vision_model, processor


def load_language_model(language_model_name):
    print(f"Loading language model: {language_model_name}")
    tokenizer = AutoTokenizer.from_pretrained(language_model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    llm = AutoModelForCausalLM.from_pretrained(
        language_model_name,
        dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    return llm, tokenizer


def attach_lora_to_language_model(language_model, r=32, alpha=64, dropout=0.05):
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=r,
        lora_alpha=alpha,
        lora_dropout=dropout,
        bias="none",
        target_modules="all-linear",
    )
    language_model = get_peft_model(language_model, lora_config)
    language_model.print_trainable_parameters()
    return language_model


class Qwen3VLProjector(torch.nn.Module):
    def __init__(self, vision_dim, llm_dim, spatial_merge_size):
        super().__init__()
        self.vision_dim = vision_dim
        self.spatial_merge_size = spatial_merge_size
        self.merged_dim = vision_dim * (spatial_merge_size ** 2)
        self.norm = torch.nn.LayerNorm(self.vision_dim, eps=1e-6)
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(self.merged_dim, self.merged_dim),
            torch.nn.GELU(),
            torch.nn.Linear(self.merged_dim, llm_dim),
        )

    def forward(self, x):
        batch_size, seq_len, channels = x.shape
        x = self.norm(x)
        grid_size = int(seq_len ** 0.5)
        x = x.view(batch_size, grid_size, grid_size, channels)
        x = x.view(
            batch_size,
            grid_size // self.spatial_merge_size,
            self.spatial_merge_size,
            grid_size // self.spatial_merge_size,
            self.spatial_merge_size,
            channels,
        )
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous()
        x = x.view(
            batch_size,
            grid_size // self.spatial_merge_size,
            grid_size // self.spatial_merge_size,
            self.merged_dim,
        )
        x = x.view(batch_size, -1, self.merged_dim)
        return self.mlp(x)


def load_image(image_root, image_field):
    if isinstance(image_field, Image.Image):
        return image_field.convert("RGB")
    if image_root is None:
        raise ValueError("image_root is None but dataset item stores an image path string")
    return Image.open(image_root / image_field).convert("RGB")


class CustomVLMCollator:
    def __init__(self, tokenizer, processor, image_root):
        self.tokenizer = tokenizer
        self.processor = processor
        self.image_root = image_root

    def __call__(self, examples):
        images = []
        input_id_list = []
        label_list = []
        attention_masks = []

        for ex in examples:
            images.append(load_image(self.image_root, ex["image"]))

            messages = [
                {"role": "user", "content": ex["prompt"]},
                {"role": "assistant", "content": ex["completion"]},
            ]
            full_text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False,
            )
            prompt_only = self.tokenizer.apply_chat_template(
                [{"role": "user", "content": ex["prompt"]}],
                tokenize=False,
                add_generation_prompt=True,
            )
            prompt_ids = self.tokenizer(prompt_only, add_special_tokens=False)["input_ids"]
            full_ids = self.tokenizer(full_text, add_special_tokens=False)["input_ids"]
            labels = ([-100] * len(prompt_ids)) + full_ids[len(prompt_ids):]

            input_id_list.append(torch.tensor(full_ids, dtype=torch.long))
            label_list.append(torch.tensor(labels, dtype=torch.long))
            attention_masks.append(torch.ones(len(full_ids), dtype=torch.long))

        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_id_list,
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id,
        )
        labels = torch.nn.utils.rnn.pad_sequence(label_list, batch_first=True, padding_value=-100)
        attention_mask = torch.nn.utils.rnn.pad_sequence(
            attention_masks,
            batch_first=True,
            padding_value=0,
        )
        pixel_values = self.processor(images=images, return_tensors="pt")["pixel_values"]

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "pixel_values": pixel_values,
        }


class CustomVLM(torch.nn.Module):
    def __init__(self, vision_model, processor, language_model, tokenizer, projector, image_token_id):
        super().__init__()
        self.vision_model = vision_model
        self.processor = processor
        self.language_model = language_model
        self.tokenizer = tokenizer
        self.projector = projector
        self.image_token_id = image_token_id

    def set_trainable(self, train_vision=False, train_language=True, train_projector=True):
        for p in self.vision_model.parameters():
            p.requires_grad = train_vision

        if hasattr(self.language_model, "peft_config"):
            trainable_markers = ("lora_", "modules_to_save.")
            for name, p in self.language_model.named_parameters():
                p.requires_grad = train_language and any(marker in name for marker in trainable_markers)
        else:
            for p in self.language_model.parameters():
                p.requires_grad = train_language

        for p in self.projector.parameters():
            p.requires_grad = train_projector

        self.vision_model.train(train_vision)
        self.language_model.train(train_language)
        self.projector.train(train_projector)

    def _vision_forward(self, pixel_values):
        if any(p.requires_grad for p in self.vision_model.parameters()):
            return self.vision_model(pixel_values=pixel_values).last_hidden_state
        with torch.no_grad():
            return self.vision_model(pixel_values=pixel_values).last_hidden_state

    def _token_embed(self, input_ids):
        embed_layer = self.language_model.get_input_embeddings()
        if any(p.requires_grad for p in embed_layer.parameters()):
            return embed_layer(input_ids)
        with torch.no_grad():
            return embed_layer(input_ids)

    def forward(self, input_ids=None, attention_mask=None, labels=None, pixel_values=None, **kwargs):
        vision_outputs = self._vision_forward(pixel_values).to(self.projector.norm.weight.dtype)
        image_features = self.projector(vision_outputs)
        inputs_embeds = self._token_embed(input_ids)

        final_embeds = []
        final_labels = []
        final_masks = []
        for i in range(input_ids.shape[0]):
            image_positions = (input_ids[i] == self.image_token_id).nonzero(as_tuple=True)[0]
            if len(image_positions) == 0:
                raise ValueError("Could not find <image> token in input_ids for an example")
            img_idx = image_positions[0].item()

            merged_embeds = torch.cat(
                [inputs_embeds[i, :img_idx, :], image_features[i], inputs_embeds[i, img_idx + 1 :, :]],
                dim=0,
            )
            final_embeds.append(merged_embeds)

            image_labels = torch.full(
                (image_features.size(1),),
                -100,
                dtype=labels.dtype,
                device=labels.device,
            )
            merged_labels = torch.cat(
                [labels[i, :img_idx], image_labels, labels[i, img_idx + 1 :]],
                dim=0,
            )
            final_labels.append(merged_labels)

            image_mask = torch.ones(
                image_features.size(1),
                dtype=attention_mask.dtype,
                device=attention_mask.device,
            )
            merged_mask = torch.cat(
                [attention_mask[i, :img_idx], image_mask, attention_mask[i, img_idx + 1 :]],
                dim=0,
            )
            final_masks.append(merged_mask)

        outputs = self.language_model(
            inputs_embeds=torch.stack(final_embeds, dim=0),
            attention_mask=torch.stack(final_masks, dim=0),
            labels=torch.stack(final_labels, dim=0),
            return_dict=True,
        )
        return outputs

    @torch.no_grad()
    def generate_text(self, input_ids, attention_mask, pixel_values, max_new_tokens=32, num_beams=3):
        self.eval()
        vision_outputs = self.vision_model(pixel_values=pixel_values).last_hidden_state.to(self.projector.norm.weight.dtype)
        image_features = self.projector(vision_outputs)
        inputs_embeds = self.language_model.get_input_embeddings()(input_ids)

        image_positions = (input_ids[0] == self.image_token_id).nonzero(as_tuple=True)[0]
        if len(image_positions) == 0:
            raise ValueError("Could not find <image> token in input_ids during generation")
        img_idx = image_positions[0].item()

        merged_embeds = torch.cat(
            [inputs_embeds[0, :img_idx, :], image_features[0], inputs_embeds[0, img_idx + 1 :, :]],
            dim=0,
        )
        image_mask = torch.ones(image_features.size(1), dtype=attention_mask.dtype, device=attention_mask.device)
        merged_mask = torch.cat(
            [attention_mask[0, :img_idx], image_mask, attention_mask[0, img_idx + 1 :]],
            dim=0,
        )

        outputs = self.language_model.generate(
            inputs_embeds=merged_embeds.unsqueeze(0),
            attention_mask=merged_mask.unsqueeze(0),
            max_new_tokens=max_new_tokens,
            num_beams=1,
            do_sample=False,
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.pad_token_id,
            use_cache=True,
        )
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)


def maybe_enable_gradient_checkpointing(model):
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    if hasattr(model.config, "use_cache"):
        model.config.use_cache = False


def report_trainable_parameters(model):
    total = 0
    trainable = 0
    for p in model.parameters():
        numel = p.numel()
        total += numel
        if p.requires_grad:
            trainable += numel
    pct = 100.0 * trainable / total if total > 0 else 0.0
    print(f"Trainable params: {trainable:,} / {total:,} ({pct:.2f}%)")


def unwrap_model(model):
    seen = set()
    while True:
        obj_id = id(model)
        if obj_id in seen:
            break
        seen.add(obj_id)

        if hasattr(model, "module"):
            model = model.module
            continue
        if hasattr(model, "_orig_mod"):
            model = model._orig_mod
            continue
        if hasattr(model, "_fsdp_wrapped_module"):
            model = model._fsdp_wrapped_module
            continue
        break
    return model


def save_peft_projector_checkpoint(model, save_dir, vision_model_name, language_model_name, spatial_merge_size):
    model = unwrap_model(model)
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    adapter_dir = save_dir / "language_model_adapter"
    tokenizer_dir = save_dir / "tokenizer"
    processor_dir = save_dir / "processor"

    model.language_model.save_pretrained(adapter_dir, safe_serialization=True)
    model.tokenizer.save_pretrained(tokenizer_dir)
    model.processor.save_pretrained(processor_dir)
    torch.save(model.projector.state_dict(), save_dir / "projector.pt")

    metadata = {
        "vision_model_name": vision_model_name,
        "language_model_name": language_model_name,
        "spatial_merge_size": spatial_merge_size,
        "image_token_id": model.image_token_id,
        "peft_adapter_path": str(adapter_dir),
        "tokenizer_path": str(tokenizer_dir),
        "processor_path": str(processor_dir),
        "projector_path": str(save_dir / "projector.pt"),
    }
    with open(save_dir / "vlm_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print(f"Saved PEFT adapter + projector checkpoint to: {save_dir}")


def get_model_device(model):
    model = unwrap_model(model)
    return next(model.language_model.parameters()).device


def evaluate_chartqa(model, ds, image_root, tokenizer, processor, max_new_tokens=32, max_examples=None, desc="ChartQA test evaluation"):
    model = unwrap_model(model)
    model.eval()

    total = len(ds) if max_examples is None else min(len(ds), max_examples)
    correct = 0
    printed = 0

    iterator = ds if max_examples is None else ds.select(range(total))
    for sample in tqdm(iterator, desc=desc):
        prompt_text = tokenizer.apply_chat_template(
            [{"role": "user", "content": sample["prompt"]}],
            tokenize=False,
            add_generation_prompt=True,
        )
        ground_truth = str(sample["completion"]).strip()

        image = load_image(image_root, sample["image"])
        device = get_model_device(model)

        pixel_values = processor(images=image, return_tensors="pt").pixel_values.to(device)
        inputs = tokenizer(prompt_text, return_tensors="pt", add_special_tokens=False)
        input_ids = inputs.input_ids.to(device)
        attention_mask = torch.ones_like(input_ids).to(device)

        prediction_text = model.generate_text(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            max_new_tokens=max_new_tokens,
        )
        prediction = prediction_text.strip()

        if printed < 5:
            print("=" * 80)
            print(f"Prompt: {prompt_text}")
            print(f"Ground truth: {ground_truth}")
            print(f"Prediction: {prediction}")
            printed += 1

        if is_answer_correct_relaxed(prediction, ground_truth):
            correct += 1

    accuracy = (correct / total) * 100 if total > 0 else 0.0
    metrics = {
        "test/relaxed_accuracy": accuracy,
        "test/num_examples": total,
        "test/num_correct": correct,
    }
    print(metrics)
    return metrics


def run_pretrain_generation_eval(model, ds, image_root, tokenizer, processor, max_examples=128, max_new_tokens=64):
    model = unwrap_model(model)
    model.eval()
    preds, golds = [], []

    iterator = ds if max_examples is None else ds.select(range(min(len(ds), max_examples)))
    for ex in tqdm(iterator, desc="Pretrain generation eval"):
        prompt_text = tokenizer.apply_chat_template(
            [{"role": "user", "content": ex["prompt"]}],
            tokenize=False,
            add_generation_prompt=True,
        )
        device = get_model_device(model)

        prompt_ids = tokenizer(prompt_text, return_tensors="pt", add_special_tokens=False)["input_ids"].to(device)
        attention_mask = torch.ones_like(prompt_ids).to(device)
        pixel_values = processor(images=[load_image(image_root, ex["image"])], return_tensors="pt").pixel_values.to(device)

        gen_text = model.generate_text(prompt_ids, attention_mask, pixel_values, max_new_tokens=max_new_tokens)
        preds.append(gen_text)
        golds.append(ex["completion"])

    bleu_scores = []
    for pred, gold in zip(preds, golds):
        if len(pred.strip()) == 0:
            bleu_scores.append(0.0)
        else:
            bleu_scores.append(BLEU.compute(predictions=[pred], references=[gold])["bleu"])
    metrics = {"pretrain_eval_bleu": float(np.mean(bleu_scores))}
    print(metrics)
    return metrics


def start_wandb_run(is_main, args, stage_name):
    if not is_main:
        return None

    run = wandb.init(
        project="vlm_continued_pretrain_chartqa",
        name=f"{args.experiment_name}_{stage_name}",
        group=args.experiment_name,
        job_type=stage_name,
        reinit=True,
    )
    wandb.config.update(vars(args), allow_val_change=True)
    return run


def build_chartqa_ocr_dataset(split="test", max_examples=None, max_ocr_words=1000, use_gpu_for_ocr=True):
    raw_ds = load_dataset("HuggingFaceM4/ChartQA")

    if split not in raw_ds:
        if split == "eval":
            split = "val" if "val" in raw_ds else "validation"
        elif split == "test":
            split = "test" if "test" in raw_ds else ("val" if "val" in raw_ds else "validation")

    ds = raw_ds[split]
    if max_examples is not None:
        ds = ds.select(range(min(max_examples, len(ds))))

    print(f"Initializing EasyOCR for split={split}...")
    reader = easyocr.Reader(["en"], gpu=use_gpu_for_ocr and torch.cuda.is_available())

    def apply_ocr_prompt(example):
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

    print(f"Building OCR prompts for {split} split...")
    ocr_ds = ds.map(apply_ocr_prompt, desc=f"Extracting OCR for {split}")

    del reader
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return ocr_ds


def compute_steps_per_epoch(num_examples, per_device_batch_size, grad_accum, world_size):
    micro_batches = math.ceil(num_examples / (per_device_batch_size * world_size))
    update_steps = math.ceil(micro_batches / grad_accum)
    return max(1, update_steps)


def build_save_steps(steps_per_epoch, num_epochs, saves_per_epoch):
    save_steps = []
    for epoch_idx in range(num_epochs):
        base = epoch_idx * steps_per_epoch
        for k in range(1, saves_per_epoch + 1):
            step = base + math.ceil((k * steps_per_epoch) / saves_per_epoch)
            save_steps.append(step)
    return sorted(set(save_steps))


class SavePeftProjectorCallback(TrainerCallback):
    def __init__(
        self,
        checkpoint_root,
        trigger_steps,
        vision_model_name,
        language_model_name,
        spatial_merge_size,
        stage_name,
    ):
        self.checkpoint_root = Path(checkpoint_root)
        self.trigger_steps = set(trigger_steps)
        self.vision_model_name = vision_model_name
        self.language_model_name = language_model_name
        self.spatial_merge_size = spatial_merge_size
        self.stage_name = stage_name
        self.saved_steps = set()

    def on_step_end(self, args, state, control, model=None, **kwargs):
        if not state.is_world_process_zero:
            return control

        step = int(state.global_step)
        if step in self.trigger_steps and step not in self.saved_steps:
            save_dir = self.checkpoint_root / f"{self.stage_name}_step_{step}"
            save_peft_projector_checkpoint(
                model=model,
                save_dir=save_dir,
                vision_model_name=self.vision_model_name,
                language_model_name=self.language_model_name,
                spatial_merge_size=self.spatial_merge_size,
            )
            self.saved_steps.add(step)
        return control


def make_training_args(
    output_dir,
    run_name,
    learning_rate,
    num_epochs,
    train_batch_size,
    eval_batch_size,
    grad_accum,
    logging_steps,
    warmup_ratio,
    weight_decay,
    optim_name,
    dataloader_num_workers=0,
    dataloader_pin_memory=False,
):
    kwargs = dict(
        output_dir=output_dir,
        per_device_train_batch_size=train_batch_size,
        per_device_eval_batch_size=eval_batch_size,
        gradient_accumulation_steps=grad_accum,
        learning_rate=learning_rate,
        num_train_epochs=num_epochs,
        run_name=run_name,
        bf16=True,
        remove_unused_columns=False,
        report_to="wandb",
        lr_scheduler_type="cosine",
        warmup_ratio=warmup_ratio,
        weight_decay=weight_decay,
        seed=SEED,
        logging_strategy="steps",
        logging_steps=logging_steps,
        eval_strategy="epoch",
        save_strategy="no",
        dataloader_num_workers=dataloader_num_workers,
        dataloader_pin_memory=dataloader_pin_memory,
        dataloader_persistent_workers=False,
        optim=optim_name,
        ddp_find_unused_parameters=False,
    )
    if dataloader_num_workers > 0:
        kwargs["dataloader_prefetch_factor"] = 1
    return TrainingArguments(**kwargs)

def load_saved_peft_projector_for_training(ckpt_dir, device):
    ckpt_dir = Path(ckpt_dir)

    with open(ckpt_dir / "vlm_metadata.json", "r", encoding="utf-8") as f:
        metadata = json.load(f)

    vision_model_name = metadata["vision_model_name"]
    language_model_name = metadata["language_model_name"]
    spatial_merge_size = metadata["spatial_merge_size"]
    image_token_id = metadata["image_token_id"]

    vision_model, processor = load_vision_model(vision_model_name)
    v_conf = vision_model.config.vision_config if hasattr(vision_model.config, "vision_config") else vision_model.config
    vision_dim = v_conf.hidden_size

    tokenizer = AutoTokenizer.from_pretrained(ckpt_dir / "tokenizer", trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_llm = AutoModelForCausalLM.from_pretrained(
        language_model_name,
        dtype=torch.bfloat16,
        trust_remote_code=True,
    )

    target_vocab_size = len(tokenizer)
    current_vocab_size = base_llm.get_input_embeddings().weight.shape[0]
    if current_vocab_size != target_vocab_size:
        print(f"Resizing token embeddings from {current_vocab_size} to {target_vocab_size}")
        base_llm.resize_token_embeddings(target_vocab_size)
        if hasattr(base_llm, "tie_weights"):
            base_llm.tie_weights()

    language_model = PeftModel.from_pretrained(
        base_llm,
        ckpt_dir / "language_model_adapter",
        is_trainable=True,
    )

    llm_dim = language_model.config.hidden_size
    projector = Qwen3VLProjector(
        vision_dim,
        llm_dim,
        spatial_merge_size,
    ).to(dtype=torch.bfloat16)

    projector_state = torch.load(ckpt_dir / "projector.pt", map_location="cpu", weights_only=False)
    projector.load_state_dict(projector_state)

    model = CustomVLM(vision_model, processor, language_model, tokenizer, projector, image_token_id)
    model = model.to(dtype=torch.bfloat16)

    return model, tokenizer, processor, metadata

def main():
    parser = argparse.ArgumentParser(
        description="Continue multimodal pretraining with LoRA on the LLM + projector, then fine-tune on ChartQA and evaluate."
    )
    parser.add_argument("--experiment_name", type=str, required=True)
    parser.add_argument("--projector_weights", type=str, default=DEFAULT_PROJECTOR_WEIGHTS)
    parser.add_argument("--vision_model_name", type=str, default="google/siglip2-so400m-patch16-512")
    parser.add_argument("--language_model_name", type=str, default="Qwen/Qwen3-4B-Instruct-2507")
    parser.add_argument("--spatial_merge_size", type=int, default=2)

    parser.add_argument("--continued_pretrain_epochs", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=5, help="ChartQA fine-tuning epochs")
    parser.add_argument("--lr", type=float, default=1e-4, help="ChartQA learning rate")
    parser.add_argument("--continued_pretrain_lr", type=float, default=None)

    parser.add_argument("--pretrain_batch_size", type=int, default=2)
    parser.add_argument("--pretrain_grad_accum", type=int, default=8)
    parser.add_argument("--finetune_batch_size", type=int, default=2)
    parser.add_argument("--finetune_grad_accum", type=int, default=8)
    parser.add_argument("--eval_batch_size", type=int, default=2)

    parser.add_argument("--lora_r", type=int, default=32)
    parser.add_argument("--lora_alpha", type=int, default=64)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--saves_per_epoch", type=int, default=3)

    parser.add_argument("--dataset_num_splits", type=int, default=5)
    parser.add_argument("--dataset_split_index", type=int, default=-1)
    parser.add_argument("--pretrain_eval_max_examples", type=int, default=128)
    parser.add_argument("--chartqa_test_max_examples", type=int, default=None)
    parser.add_argument("--max_new_tokens", type=int, default=32)
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--warmup_ratio", type=float, default=0.03)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--optim", type=str, default="adamw_torch")
    parser.add_argument("--max_ocr_words", type=int, default=1000)
    parser.add_argument("--ocr_on_cpu", action="store_true")
    parser.add_argument("--skip_stage1", action="store_true")
    parser.add_argument("--resume_peft_dir", type=str, default=None)
    parser.add_argument("--overfit_check", action="store_true")
    args = parser.parse_args()

    if args.skip_stage1 and not args.resume_peft_dir:
        raise ValueError("--resume_peft_dir is required when --skip_stage1 is used")

    set_seed(SEED)
    os.makedirs("./outputs", exist_ok=True)

    continued_pretrain_lr = args.continued_pretrain_lr if args.continued_pretrain_lr is not None else args.lr
    run_output_dir = Path("./outputs") / args.experiment_name
    stage1_dir = run_output_dir / "continued_pretrain_final"
    stage2_dir = run_output_dir / "chartqa_final2"
    stage1_ckpt_dir = run_output_dir / "stage1_checkpoints"
    stage2_ckpt_dir = run_output_dir / "stage2_checkpoints"

    is_main = int(os.environ.get("RANK", "0")) == 0
    world_size = int(os.environ.get("WORLD_SIZE", "1"))

    if is_main:
        wandb.login(key=config.WANDB_KEY)
        if args.skip_stage1:
            run = start_wandb_run(is_main, args, "stage2_chartqa_finetune")
        else:
            run = start_wandb_run(is_main, args, "stage1_continued_pretrain")
    else:
        run = None

    print("=" * 100)
    print("Loading datasets")
    print("=" * 100)
    pretrain_ds, pretrain_image_root = create_dataset(
        overfit_check=args.overfit_check,
        dataset_num_splits=args.dataset_num_splits,
        dataset_split_index=args.dataset_split_index,
        pretraining=True,
    )
    chartqa_ds, chartqa_image_root = create_dataset(
        overfit_check=args.overfit_check,
        pretraining=False,
    )

    print("=" * 100)
    print("Loading models")
    print("=" * 100)

    if args.skip_stage1:
        print(f"Loading saved stage-1 PEFT checkpoint from: {args.resume_peft_dir}")
        model, tokenizer, processor, saved_meta = load_saved_peft_projector_for_training(
            args.resume_peft_dir,
            DEVICE,
        )
        pretrain_collator = None
        chartqa_collator = CustomVLMCollator(tokenizer, processor, chartqa_image_root)

    else:
        vision_model, processor = load_vision_model(args.vision_model_name)
        v_conf = vision_model.config.vision_config if hasattr(vision_model.config, "vision_config") else vision_model.config
        vision_dim = v_conf.hidden_size

        language_model, tokenizer = load_language_model(args.language_model_name)
        llm_dim = language_model.config.hidden_size

        if "<image>" not in tokenizer.get_vocab():
            tokenizer.add_special_tokens({"additional_special_tokens": ["<image>"]})
            language_model.resize_token_embeddings(len(tokenizer))
        image_token_id = tokenizer.convert_tokens_to_ids("<image>")

        language_model = attach_lora_to_language_model(
            language_model=language_model,
            r=args.lora_r,
            alpha=args.lora_alpha,
            dropout=args.lora_dropout,
        )

        projector = Qwen3VLProjector(
            vision_dim,
            llm_dim,
            args.spatial_merge_size,
        ).to(dtype=torch.bfloat16)

        print(f"Loading pretrained projector from: {args.projector_weights}")
        projector_state = torch.load(args.projector_weights, map_location="cpu", weights_only=False)
        projector.load_state_dict(projector_state)

        model = CustomVLM(vision_model, processor, language_model, tokenizer, projector, image_token_id)
        model = model.to(dtype=torch.bfloat16)

        model.set_trainable(train_vision=False, train_language=True, train_projector=True)
        maybe_enable_gradient_checkpointing(model.language_model)
        report_trainable_parameters(model)

        pretrain_collator = CustomVLMCollator(tokenizer, processor, pretrain_image_root)
        chartqa_collator = CustomVLMCollator(tokenizer, processor, chartqa_image_root)

    if not args.skip_stage1:
        stage1_steps_per_epoch = compute_steps_per_epoch(
            num_examples=len(pretrain_ds["train"]),
            per_device_batch_size=args.pretrain_batch_size,
            grad_accum=args.pretrain_grad_accum,
            world_size=world_size,
        )
        stage1_save_steps = build_save_steps(
            steps_per_epoch=stage1_steps_per_epoch,
            num_epochs=args.continued_pretrain_epochs,
            saves_per_epoch=args.saves_per_epoch,
        )

        print("=" * 100)
        print("Stage 1: continued multimodal pretraining with LoRA LLM + connector trainable, vision frozen")
        print(f"Stage 1 save checkpoints at global steps: {stage1_save_steps}")
        print("=" * 100)

        pretrain_args = make_training_args(
            output_dir=str(run_output_dir / "stage1_tmp"),
            run_name=f"{args.experiment_name}_stage1_continued_pretrain",
            learning_rate=continued_pretrain_lr,
            num_epochs=args.continued_pretrain_epochs,
            train_batch_size=args.pretrain_batch_size,
            eval_batch_size=args.eval_batch_size,
            grad_accum=args.pretrain_grad_accum,
            logging_steps=args.logging_steps,
            warmup_ratio=args.warmup_ratio,
            weight_decay=args.weight_decay,
            optim_name=args.optim,
            dataloader_num_workers=0,
            dataloader_pin_memory=False,
        )

        pretrain_trainer = Trainer(
            model=model,
            args=pretrain_args,
            train_dataset=pretrain_ds["train"],
            eval_dataset=pretrain_ds["eval"],
            data_collator=pretrain_collator,
            callbacks=[
                SavePeftProjectorCallback(
                    checkpoint_root=stage1_ckpt_dir,
                    trigger_steps=stage1_save_steps,
                    vision_model_name=args.vision_model_name,
                    language_model_name=args.language_model_name,
                    spatial_merge_size=args.spatial_merge_size,
                    stage_name="stage1",
                )
            ],
        )

        pretrain_trainer.train()
        stage1_metrics = pretrain_trainer.evaluate()
        print(stage1_metrics)

        if is_main:
            wandb.log({f"stage1/{k}": v for k, v in stage1_metrics.items()})

            pretrain_gen_metrics = run_pretrain_generation_eval(
                model,
                pretrain_ds["eval"],
                pretrain_image_root,
                tokenizer,
                processor,
                max_examples=args.pretrain_eval_max_examples,
                max_new_tokens=max(32, args.max_new_tokens),
            )
            wandb.log({f"stage1/{k}": v for k, v in pretrain_gen_metrics.items()})

            save_peft_projector_checkpoint(
                model,
                stage1_dir,
                vision_model_name=args.vision_model_name,
                language_model_name=args.language_model_name,
                spatial_merge_size=args.spatial_merge_size,
            )

        del pretrain_trainer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()

        if is_main and run is not None:
            run.finish()
            run = start_wandb_run(is_main, args, "stage2_chartqa_finetune")

    print("=" * 100)
    print("Stage 2: ChartQA fine-tuning from the continued-pretrained LoRA adapter + projector")
    print("=" * 100)

    model.set_trainable(train_vision=False, train_language=True, train_projector=True)
    maybe_enable_gradient_checkpointing(model.language_model)
    report_trainable_parameters(model)

    stage2_steps_per_epoch = compute_steps_per_epoch(
        num_examples=len(chartqa_ds["train"]),
        per_device_batch_size=args.finetune_batch_size,
        grad_accum=args.finetune_grad_accum,
        world_size=world_size,
    )
    stage2_save_steps = build_save_steps(
        steps_per_epoch=stage2_steps_per_epoch,
        num_epochs=args.epochs,
        saves_per_epoch=args.saves_per_epoch,
    )
    print(f"Stage 2 save checkpoints at global steps: {stage2_save_steps}")

    finetune_args = make_training_args(
        output_dir=str(run_output_dir / "stage2_tmp"),
        run_name=f"{args.experiment_name}_stage2_chartqa",
        learning_rate=args.lr,
        num_epochs=args.epochs,
        train_batch_size=args.finetune_batch_size,
        eval_batch_size=args.eval_batch_size,
        grad_accum=args.finetune_grad_accum,
        logging_steps=args.logging_steps,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        optim_name=args.optim,
        dataloader_num_workers=0,
        dataloader_pin_memory=False,
    )

    finetune_trainer = Trainer(
        model=model,
        args=finetune_args,
        train_dataset=chartqa_ds["train"],
        eval_dataset=chartqa_ds["eval"],
        data_collator=chartqa_collator,
        callbacks=[
            SavePeftProjectorCallback(
                checkpoint_root=stage2_ckpt_dir,
                trigger_steps=stage2_save_steps,
                vision_model_name=args.vision_model_name,
                language_model_name=args.language_model_name,
                spatial_merge_size=args.spatial_merge_size,
                stage_name="stage2",
            )
        ],
    )

    finetune_trainer.train()
    stage2_metrics = finetune_trainer.evaluate()
    print(stage2_metrics)

    if is_main:
        wandb.log({f"stage2/{k}": v for k, v in stage2_metrics.items()})

        save_peft_projector_checkpoint(
            model,
            stage2_dir,
            vision_model_name=args.vision_model_name,
            language_model_name=args.language_model_name,
            spatial_merge_size=args.spatial_merge_size,
        )

    print("=" * 100)
    print("Final ChartQA test evaluation")
    print("=" * 100)

    if hasattr(model.language_model.config, "use_cache"):
        model.language_model.config.use_cache = True

    test_metrics = evaluate_chartqa(
        model,
        chartqa_ds["test"],
        chartqa_image_root,
        tokenizer,
        processor,
        max_new_tokens=args.max_new_tokens,
        max_examples=args.chartqa_test_max_examples,
        desc="ChartQA test evaluation",
    )

    if is_main:
        wandb.log(test_metrics)

    print("=" * 100)
    print("Final ChartQA OCR test evaluation")
    print("=" * 100)

    ocr_test_ds = build_chartqa_ocr_dataset(
        split="test",
        max_examples=args.chartqa_test_max_examples,
        max_ocr_words=args.max_ocr_words,
        use_gpu_for_ocr=not args.ocr_on_cpu,
    )

    ocr_raw_metrics = evaluate_chartqa(
        model,
        ocr_test_ds,
        image_root=None,
        tokenizer=tokenizer,
        processor=processor,
        max_new_tokens=args.max_new_tokens,
        max_examples=None,
        desc="ChartQA OCR test evaluation",
    )

    ocr_test_metrics = {
        "ocr_test/relaxed_accuracy": ocr_raw_metrics["test/relaxed_accuracy"],
        "ocr_test/num_examples": ocr_raw_metrics["test/num_examples"],
        "ocr_test/num_correct": ocr_raw_metrics["test/num_correct"],
    }
    print(ocr_test_metrics)

    if is_main:
        wandb.log(ocr_test_metrics)

    print("=" * 100)
    print(f"Stage 1 final save: {stage1_dir}")
    print(f"Stage 1 checkpoints: {stage1_ckpt_dir}")
    print(f"Stage 2 final save: {stage2_dir}")
    print(f"Stage 2 checkpoints: {stage2_ckpt_dir}")
    print("=" * 100)

    if is_main and run is not None:
        run.finish()

if __name__ == "__main__":
    main()