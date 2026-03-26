"""
Base architecture for our custom VLM
"""

from pathlib import Path
import zipfile
import evaluate
import wandb
import os
import config
import numpy as np
import argparse
import math
import re
from pprint import pprint
from datasets import load_dataset, DatasetDict
from huggingface_hub import snapshot_download
from PIL import Image
from transformers import (
    AutoProcessor,
    AutoModel,
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    TrainerCallback,
)
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR, get_last_checkpoint
from transformers.trainer_callback import TrainerState
import torch
import os


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 8
bleu = evaluate.load("bleu")

def create_dataset(overfit_check=False, dataset_num_splits=5, dataset_split_index=-1):
    # step 1: download the snapshots directly rather than let HF do it
    # also only download some of the files because there was this weird column mismatch thing happening
    snapshot_dir = Path(snapshot_download(
        repo_id="liuhaotian/LLaVA-Pretrain", repo_type="dataset", allow_patterns=["blip_laion_cc_sbu_558k.json", "images.zip"]
    ))
    ds = load_dataset("json", data_files=str(snapshot_dir / "blip_laion_cc_sbu_558k.json"), split="train")

    # step 2: if we're doing this for the first time, images will be downloaded as a zip file, so we need to unzip it
    image_root = snapshot_dir / "images"
    if image_root.exists() and any(image_root.rglob("*.jpg")):
        print("images have already been unzipped!")
    else:
        image_root.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(snapshot_dir / "images.zip", "r") as zf:
            zf.extractall(image_root)
        print("Extraction complete.")

    # step 3: check the first row and see that it downloaded correctly
    # note that this dataset doesnt have the raw images itself, so we'll load them lazily later in the collator (i think)
    for _, v in ds[0].items():
        pprint(v)
    img = Image.open((snapshot_dir / "images") / ds[0]["image"]).convert("RGB")
    print(f"Image size: {img.size}")
    # plt.imshow(img)
    # plt.savefig("sample_image.png")

    # step 4: change up the format to a prompt-completion dataset 
    ds = ds.map(lambda ex: {
        "image": ex["image"],
        "prompt": ex["conversations"][0]["value"].strip(), 
        "completion": ex["conversations"][1]["value"].strip()
    }, remove_columns=["id", "conversations"])
    print(ds[0])

    # step 5: split up into train/eval/test
    split_1 = ds.train_test_split(test_size=0.10, seed=SEED, shuffle=True)
    split_2 = split_1["test"].train_test_split(test_size=0.50, seed=SEED, shuffle=True)
    ds = DatasetDict({"train": split_1["train"], "eval": split_2["train"], "test": split_2["test"]})
    print(ds)

    # step 6a: split up into chunks if requested
    if dataset_split_index >= 0 and dataset_split_index < dataset_num_splits:
        train_len = len(ds["train"])
        chunk_size = (train_len + dataset_num_splits - 1) // dataset_num_splits
        start = dataset_split_index * chunk_size
        end = min(start + chunk_size, train_len)
        print(f"Selecting train chunk {dataset_split_index + 1}/{dataset_num_splits}")
        ds["train"] = ds["train"].select(range(start, end))
        print(ds)

    # step 6b: for debugging only
    if overfit_check:
        ds["train"] = ds["train"].select(range(3))
        ds["eval"] = ds["train"]
        ds["test"] = ds["train"]
        print(ds)

    return ds, image_root

def create_chartqa_dataset(overfit_check: bool = False, dataset_num_splits=5, dataset_split_index=-1):
    """
    Create a prompt-completion dataset for ChartQA.

    The model expects an `<image>` token inside the `prompt` so it can locate
    where to splice projected vision embeddings.
    """
    raw = load_dataset("HuggingFaceM4/ChartQA")

    train_split = raw["train"]
    if "val" in raw:
        eval_split = raw["val"]
    elif "validation" in raw:
        eval_split = raw["validation"]
    else:
        eval_split = raw["test"]

    test_split = raw["test"] if "test" in raw else raw["train"]
    
    def to_prompt_completion_batch(batch):
        golds = [str(lbl[0]).strip() if lbl else "" for lbl in batch.get("label", [])]
        prompts = [f"<image>\nQuestion: {q}\nAnswer:" for q in batch["query"]]
        return {
            "image": batch["image"],
            "prompt": prompts,
            "completion": golds,
    }

    keep_cols = {"image", "prompt", "completion"}
    train = train_split.map(
        to_prompt_completion_batch,
        batched=True, 
        batch_size=1000,
        remove_columns=[c for c in train_split.column_names if c not in keep_cols],
        num_proc=os.cpu_count()
    )
    eval_ds = eval_split.map(
        to_prompt_completion_batch,
        batched=True, 
        batch_size=1000,
        remove_columns=[c for c in train_split.column_names if c not in keep_cols],
        num_proc=os.cpu_count()
    )
    test_ds = test_split.map(
        to_prompt_completion_batch,
        batched=True, 
        batch_size=1000,
        remove_columns=[c for c in train_split.column_names if c not in keep_cols],
        num_proc=os.cpu_count()
    )

    ds = DatasetDict({"train": train, "eval": eval_ds, "test": test_ds})

    if overfit_check:
        ds["train"] = ds["train"].select(range(min(3, len(ds["train"]))))
        ds["eval"] = ds["train"]
        ds["test"] = ds["train"]

    # For ChartQA we keep images inside the dataset; no external image_root needed.
    return ds, None

# TODO: i had to make this diff than the notebook code - had to load the vision model block specifically
def load_vision_model(vision_model_name):
    print(f"Loading Vision: {vision_model_name}")
    processor = AutoProcessor.from_pretrained(vision_model_name, use_fast=True)
    vision_model = AutoModel.from_pretrained(vision_model_name).to(DEVICE)
    vision_model.eval()
    return vision_model.vision_model, processor

# TODO: needed to remove auto to work with multi-GPU stuff
def load_language_model(language_model_name):
    print(f"Loading LLM: {language_model_name}")
    tokenizer = AutoTokenizer.from_pretrained(language_model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    llm = AutoModelForCausalLM.from_pretrained(language_model_name, dtype=torch.bfloat16, trust_remote_code=True)
    return llm, tokenizer

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
            torch.nn.Linear(self.merged_dim, llm_dim)
        )

    def forward(self, x):
        # TODO: seq_len may not be a perfect square if it includes a CLS token, this might break then
        B, seq_len, C = x.shape
        x = self.norm(x)
        grid_size = int(seq_len ** 0.5)
        x = x.view(B, grid_size, grid_size, C)
        x = x.view(B, grid_size // self.spatial_merge_size, self.spatial_merge_size, 
            grid_size // self.spatial_merge_size, self.spatial_merge_size, C)
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous()
        x = x.view(B, grid_size // self.spatial_merge_size, grid_size // self.spatial_merge_size, self.merged_dim)
        x = x.view(B, -1, self.merged_dim)
        return self.mlp(x)
    
class CustomVLMCollator():
    def __init__(self, tokenizer, processor, image_root=None):
        self.tokenizer = tokenizer
        self.processor = processor
        self.image_root = image_root

    def _load_image(self, ex_image):
        """
        ChartQA provides a PIL image in `ex["image"]`.
        LLaVA pretrain provides a string filename to open under `image_root`.
        """
        if isinstance(ex_image, str):
            if self.image_root is None:
                return Image.open(ex_image).convert("RGB")
            return Image.open(self.image_root / ex_image).convert("RGB")

        # Most datasets will yield a PIL.Image instance.
        if hasattr(ex_image, "convert"):
            return ex_image.convert("RGB")

        # Extra robustness for dict-based image representations.
        if isinstance(ex_image, dict):
            path = ex_image.get("path") or ex_image.get("image") or ex_image.get("file_name")
            if path is None:
                raise ValueError(f"Unsupported image dict keys: {list(ex_image.keys())}")
            if self.image_root is None:
                return Image.open(path).convert("RGB")
            return Image.open(self.image_root / path).convert("RGB")

        raise ValueError(f"Unsupported image type: {type(ex_image)}")

    def __call__(self, examples):
        images = []
        input_id_list = []
        label_list = []
        attention_masks = []

        for ex in examples:
            images.append(self._load_image(ex["image"]))

            messages = [{"role": "user", "content": ex["prompt"]}, {"role": "assistant", "content": ex["completion"]}]
            full_text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
            prompt_only = self.tokenizer.apply_chat_template(
                [{"role": "user", "content": ex["prompt"]}], tokenize=False, add_generation_prompt=True
            )
            prompt_ids = self.tokenizer(prompt_only, add_special_tokens=False)["input_ids"]
            full_ids = self.tokenizer(full_text, add_special_tokens=False)["input_ids"]
            
            labels = ([-100] * len(prompt_ids)) + full_ids[len(prompt_ids):]

            input_id_list.append(torch.tensor(full_ids, dtype=torch.long))
            label_list.append(torch.tensor(labels, dtype=torch.long))
            attention_masks.append(torch.ones(len(full_ids), dtype=torch.long)) # 1 for all non-pad tokens

        input_ids = torch.nn.utils.rnn.pad_sequence(input_id_list, batch_first=True, padding_value=self.tokenizer.pad_token_id)
        labels = torch.nn.utils.rnn.pad_sequence(label_list, batch_first=True, padding_value=-100)
        attention_mask = torch.nn.utils.rnn.pad_sequence(attention_masks, batch_first=True, padding_value=0)
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

    def freeze_backbones(self):
        for p in self.vision_model.parameters():
            p.requires_grad = False
        for p in self.language_model.parameters():
            p.requires_grad = False
        for p in self.projector.parameters():
            p.requires_grad = True

    def forward(self, input_ids=None, attention_mask=None, labels=None, pixel_values=None, **kwargs):
        # TODO: needed to change the float type from f32 to bf16
        with torch.no_grad():
            vision_outputs = self.vision_model(pixel_values=pixel_values).last_hidden_state.to(self.projector.norm.weight.dtype)
        image_features = self.projector(vision_outputs)
        with torch.no_grad():
            inputs_embeds = self.language_model.get_input_embeddings()(input_ids)

        final_embeds = []
        final_labels = []
        final_masks = []
        for i in range(input_ids.shape[0]):
            img_idx = (input_ids[i] == self.image_token_id).nonzero(as_tuple=True)[0][0].item()

            # imp: we dont truncate because we dont want to truncate the image accidentally
            merged_embeds = torch.cat([inputs_embeds[i, :img_idx, :], image_features[i], inputs_embeds[i, img_idx + 1 :, :]], dim=0)
            final_embeds.append(merged_embeds)

            image_labels = torch.full((image_features.size(1),), -100, dtype=labels.dtype, device=labels.device)
            merged_labels = torch.cat([labels[i, :img_idx], image_labels, labels[i, img_idx + 1 :]], dim=0)
            final_labels.append(merged_labels)

            image_mask = torch.ones(image_features.size(1), dtype=attention_mask.dtype, device=attention_mask.device)
            merged_mask = torch.cat([attention_mask[i, :img_idx], image_mask, attention_mask[i, img_idx + 1 :]], dim=0)
            final_masks.append(merged_mask)

        outputs = self.language_model(
            inputs_embeds=torch.stack(final_embeds, dim=0),
            attention_mask=torch.stack(final_masks, dim=0),
            labels=torch.stack(final_labels, dim=0),
            return_dict=True
        )
        return outputs
    
    @torch.no_grad()
    def generate_text(self, input_ids, attention_mask, pixel_values, max_new_tokens):
        self.eval()

        vision_outputs = self.vision_model(pixel_values=pixel_values).last_hidden_state.to(self.projector.norm.weight.dtype)
        image_features = self.projector(vision_outputs)
        inputs_embeds = self.language_model.get_input_embeddings()(input_ids)

        img_idx = (input_ids[0] == self.image_token_id).nonzero(as_tuple=True)[0][0].item()

        merged_embeds = torch.cat([inputs_embeds[0, :img_idx, :], image_features[0], inputs_embeds[0, img_idx + 1 :, :]], dim=0)
        image_mask = torch.ones(image_features.size(1), dtype=attention_mask.dtype, device=attention_mask.device)
        merged_mask = torch.cat([attention_mask[0, :img_idx], image_mask, attention_mask[0, img_idx + 1 :]], dim=0)

        # nest in an outer list, similar to loop above
        merged_embeds = merged_embeds.unsqueeze(0)
        merged_mask = merged_mask.unsqueeze(0)

        outputs = self.language_model.generate(
            inputs_embeds=merged_embeds,
            attention_mask=merged_mask,
            max_new_tokens=max_new_tokens,
            num_beams=3,
            do_sample=False,
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.pad_token_id,
        )
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)
    
class GenEvalCallback(TrainerCallback):
    def __init__(self, trainer, eval_ds, image_root, tokenizer, processor, overfit, dataset_type):
        self.trainer = trainer
        self.eval_ds = eval_ds
        self.image_root = image_root
        self.tokenizer = tokenizer
        self.processor = processor
        self.overfit = overfit
        self.dataset_type = dataset_type

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if metrics is None:
            metrics = {}

        if self.overfit:
            gen_metrics = run_generation_eval(
                self.trainer.model, 
                self.eval_ds,
                self.image_root,
                self.tokenizer,
                self.processor,
                dataset_type=self.dataset_type,
            )
            metrics.update(gen_metrics)
            self.trainer.log(gen_metrics)

        return control
        
def extract_answer(text: str) -> str:
    text = str(text).strip()
    if not text:
        return ""

    # Your baselines evaluate on the final answer portion.
    if "Answer:" in text:
        text = text.split("Answer:", 1)[-1]
    text = text.strip()

    lines = text.splitlines()
    if len(lines) == 0:
        return ""
    first = lines[0].strip()

    # Remove common "final answer: ..." prefixes.
    first = re.sub(
        r"^(final answer|final|answer)\s*[:\-]\s*",
        "",
        first,
        flags=re.IGNORECASE,
    ).strip()
    return first

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
            return float(math.isclose(pred_num, 0.0))
        rel_err = abs(pred_num - gold_num) / abs(gold_num)
        return float(rel_err <= 0.05)

    return float(pred_norm == gold_norm)

def is_answer_correct_exact(pred, gold):
    pred_norm, gold_norm = normalize_answer(pred), normalize_answer(gold)
    pred_num, gold_num = to_float(pred_norm), to_float(gold_norm)

    if pred_num is not None and gold_num is not None:
        return float(pred_num == gold_num)

    return float(pred_norm == gold_norm)

def run_generation_eval(model, ds, image_root, tokenizer, processor, dataset_type = "llava"):
    model.eval()
    preds, golds = [], []

    with torch.no_grad():
        for _, ex in enumerate(ds):
            # create inputs
            prompt_text = tokenizer.apply_chat_template(
                [{"role": "user", "content": ex["prompt"]}],
                tokenize=False,
                add_generation_prompt=True
            )
            prompt_ids = tokenizer(prompt_text, return_tensors="pt", add_special_tokens=False)["input_ids"].to(DEVICE)
            attention_mask = torch.ones_like(prompt_ids).to(DEVICE)
            if dataset_type == "llava":
                pixel_values = processor(
                    images=[Image.open(image_root / ex["image"]).convert("RGB")],
                    return_tensors="pt",
                ).pixel_values.to(DEVICE)
            else:
                # ChartQA: images are stored directly in the dataset.
                if hasattr(ex["image"], "convert"):
                    img = ex["image"].convert("RGB")
                elif isinstance(ex["image"], dict):
                    path = ex["image"].get("path") or ex["image"].get("image") or ex["image"].get("file_name")
                    if path is None:
                        raise ValueError(f"Unsupported image dict keys in eval: {list(ex['image'].keys())}")
                    if image_root is None:
                        img = Image.open(path).convert("RGB")
                    else:
                        img = Image.open(image_root / path).convert("RGB")
                elif isinstance(ex["image"], str):
                    if image_root is None:
                        img = Image.open(ex["image"]).convert("RGB")
                    else:
                        img = Image.open(image_root / ex["image"]).convert("RGB")
                else:
                    raise ValueError(f"Unsupported image type in eval: {type(ex['image'])}")
                pixel_values = processor(images=[img], return_tensors="pt").pixel_values.to(DEVICE)
            # run generation
            gen_text = model.generate_text(prompt_ids, attention_mask, pixel_values, max_new_tokens=64)
            if dataset_type == "chartqa":
                pred = extract_answer(gen_text)
                preds.append(pred)
                golds.append(ex["completion"])

                # print("PRINTING OUT OUTPUTS")
                # print("GENERATED:\n", preds[-1])
                # print("TARGET:\n", golds[-1])
            else:
                preds.append(gen_text)
                golds.append(ex["completion"])

                # print("PRINTING OUT OUTPUTS")
                # print("GENERATED:\n", preds[-1])
                # print("TARGET:\n", golds[-1])
    # compute metrics
    if dataset_type == "chartqa":
        correct_relaxed, correct_exact, total = 0.0, 0.0, 0
        for pred, gold in zip(preds, golds):
            c_relaxed = is_answer_correct_relaxed(pred, gold)
            c_exact = is_answer_correct_exact(pred, gold)
            correct_relaxed += c_relaxed
            correct_exact += c_exact
            total += 1

        metrics = {
            "eval_relaxed_acc": float(correct_relaxed / max(total, 1)),
            "eval_exact_acc": float(correct_exact / max(total, 1)),
        }
        print(f"Relaxed accuracy: {metrics['eval_relaxed_acc']:.4f}")
        print(f"Exact accuracy: {metrics['eval_exact_acc']:.4f}")
        return metrics

    bleu_scores = []
    for pred, gold in zip(preds, golds):
        if len(str(pred).strip()) == 0:
            bleu_scores.append(0.0)
        else:
            bleu_scores.append(bleu.compute(predictions=[pred], references=[gold])["bleu"])
    metrics = {"eval_bleu": float(np.mean(bleu_scores))}
    print(metrics)
    return(metrics)

class ProjectorOnlyTrainer(Trainer):
    """
    Lightweight checkpointing:
      - saves only projector weights
      - saves trainer state / training args for bookkeeping
      - does NOT save frozen backbones, optimizer / scheduler / scaler / RNG
    """

    def _save(self, output_dir=None, state_dict=None):
        output_dir = output_dir if output_dir is not None else self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)

        model = self.model
        if hasattr(self, "accelerator"):
            model = self.accelerator.unwrap_model(model)

        torch.save(model.projector.state_dict(), os.path.join(output_dir, "projector.pt"))
        torch.save(self.args, os.path.join(output_dir, "training_args.bin"))

    def _save_checkpoint(self, model, trial=None, metrics=None):
        checkpoint_folder = f"{PREFIX_CHECKPOINT_DIR}-{self.state.global_step}"
        output_dir = os.path.join(self.args.output_dir, checkpoint_folder)
        os.makedirs(output_dir, exist_ok=True)

        # save projector + args
        self._save(output_dir)

        # save trainer bookkeeping state
        self.state.save_to_json(os.path.join(output_dir, "trainer_state.json"))

        # update best checkpoint tracking
        if metrics is not None and self.args.metric_for_best_model is not None:
            metric_to_check = self.args.metric_for_best_model
            if not metric_to_check.startswith("eval_"):
                metric_to_check = f"eval_{metric_to_check}"

            metric_value = metrics.get(metric_to_check)
            if metric_value is not None:
                operator = np.greater if self.args.greater_is_better else np.less
                if (
                    self.state.best_metric is None
                    or self.state.best_model_checkpoint is None
                    or operator(metric_value, self.state.best_metric)
                ):
                    self.state.best_metric = metric_value
                    self.state.best_model_checkpoint = output_dir
                    self.state.save_to_json(os.path.join(output_dir, "trainer_state.json"))

        self._rotate_checkpoints(use_mtime=False, output_dir=self.args.output_dir)

    def _load_from_checkpoint(self, resume_from_checkpoint, model=None):
        if model is None:
            model = self.model

        print(f"Loading custom checkpoint from {resume_from_checkpoint}")

        if hasattr(self, "accelerator"):
            unwrapped_model = self.accelerator.unwrap_model(model)
        else:
            unwrapped_model = model

        projector_path = os.path.join(resume_from_checkpoint, "projector.pt")
        trainer_state_path = os.path.join(resume_from_checkpoint, "trainer_state.json")

        if not os.path.exists(projector_path):
            raise ValueError(f"Missing projector checkpoint at {projector_path}")

        projector_state = torch.load(projector_path, map_location="cpu", weights_only=False)
        unwrapped_model.projector.load_state_dict(projector_state)

        if os.path.exists(trainer_state_path):
            self.state = TrainerState.load_from_json(trainer_state_path)

    def _load_best_model(self):
        if self.state.best_model_checkpoint is None:
            print("No best model checkpoint found; skipping best model load.")
            return

        print(f"Loading best projector from {self.state.best_model_checkpoint}")

        model = self.model
        if hasattr(self, "accelerator"):
            model = self.accelerator.unwrap_model(model)

        projector_path = os.path.join(self.state.best_model_checkpoint, "projector.pt")
        if not os.path.exists(projector_path):
            raise ValueError(f"Best projector checkpoint missing at {projector_path}")

        state_dict = torch.load(projector_path, map_location="cpu", weights_only=False)
        model.projector.load_state_dict(state_dict)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment_name", type=str, required=True)
    parser.add_argument("--dataset_type", type=str, default="llava", choices=["llava", "chartqa"])
    parser.add_argument("--vision_model_name", type=str, default="google/siglip2-so400m-patch16-512") 
    parser.add_argument("--language_model_name", type=str, default="Qwen/Qwen3-4B-Instruct-2507")
    parser.add_argument("--spatial_merge_size", type=int, default=2)
    parser.add_argument("--overfit_check", action="store_true")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--dataset_num_splits", type=int, default=5)
    parser.add_argument("--dataset_split_index", type=int, default=-1)
    parser.add_argument("--lr", type=float, default=5e-3)
    parser.add_argument("--unfreeze_backbones", action="store_true")
    args = parser.parse_args()

    EXPERIMENT_NAME = args.experiment_name
    DATASET_TYPE = args.dataset_type
    OVERFIT_CHECK = True if args.overfit_check else False
    LR = args.lr
    EPOCHS = args.epochs
    SPATIAL_MERGE_SIZE = args.spatial_merge_size
    VISION_MODEL_NAME = args.vision_model_name
    LANGUAGE_MODEL_NAME = args.language_model_name
    DATASET_NUM_SPLITS = args.dataset_num_splits
    DATASET_SPLIT_INDEX = args.dataset_split_index
    OUTPUT_DIR = f"./outputs/{EXPERIMENT_NAME}"
    UNFREEZE_BACKBONES = True if args.unfreeze_backbones else False

    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(SEED)
        torch.cuda.manual_seed_all(SEED)  # for multi-GPU
    
    # 2. Set seed for other Python libraries
    np.random.seed(SEED)
    os.environ['PYTHONHASHSEED'] = str(SEED)

    # initialize wandb
    wandb.login(key=config.WANDB_KEY)
    run = wandb.init(project=f"base_architecture", name=EXPERIMENT_NAME)

    # load dataset and dir where raw images are stored
    if DATASET_TYPE == "chartqa":
        ds, image_root = create_chartqa_dataset(overfit_check=OVERFIT_CHECK, 
                                                dataset_num_splits = DATASET_NUM_SPLITS, 
                                                dataset_split_index = DATASET_SPLIT_INDEX)
    else:
        ds, image_root = create_dataset(overfit_check=OVERFIT_CHECK,
                                        dataset_num_splits = DATASET_NUM_SPLITS, 
                                        dataset_split_index = DATASET_SPLIT_INDEX)

    # load vision model
    vision_model, processor = load_vision_model(VISION_MODEL_NAME)
    v_conf = vision_model.config.vision_config if hasattr(vision_model.config, "vision_config") else vision_model.config
    vision_dim = v_conf.hidden_size

    # load llm
    language_model, tokenizer = load_language_model(LANGUAGE_MODEL_NAME)
    llm_dim = language_model.config.hidden_size
    if "<image>" not in tokenizer.get_vocab():
        tokenizer.add_special_tokens({"additional_special_tokens": ["<image>"]})
        language_model.resize_token_embeddings(len(tokenizer))
    image_token_id = tokenizer.convert_tokens_to_ids("<image>")

    # load custom proj layer
    proj_layer = Qwen3VLProjector(vision_dim, llm_dim, SPATIAL_MERGE_SIZE).to(DEVICE, dtype=torch.bfloat16)

    # load whole model now
    model = CustomVLM(vision_model, processor, language_model, tokenizer, proj_layer, image_token_id)
    if not UNFREEZE_BACKBONES:
        model.freeze_backbones()
    collator = CustomVLMCollator(tokenizer, processor, image_root)

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=16,
        gradient_accumulation_steps=4,
        learning_rate=LR,
        num_train_epochs=EPOCHS,
        run_name=EXPERIMENT_NAME,
        bf16=True,
        remove_unused_columns=False,
        report_to="wandb",
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        seed=SEED,
        save_total_limit=2,
        eval_strategy="steps",
        eval_steps=5000, # making this large because it takes 1hr+ to run, and so i've also set load best model at end = false
        logging_strategy="steps",
        logging_steps=100,
        save_strategy="steps",
        save_steps=100
    )

    trainer = ProjectorOnlyTrainer(
        model=model,
        args=training_args,
        train_dataset=ds["train"],
        eval_dataset=ds["eval"],
        data_collator=collator,
    )
    trainer.add_callback(GenEvalCallback(trainer, ds["eval"], image_root, tokenizer, processor, OVERFIT_CHECK))

    # not resuming from checkpoint, as we're only saving the proj layer weights
    # we cant simply just load those, we'd need to save the scheduler, optim states too and im lazy
    print("beginning training!")
    last_checkpoint = get_last_checkpoint(OUTPUT_DIR) if os.path.isdir(OUTPUT_DIR) else None
    if last_checkpoint is not None:
        print(f"Resuming from checkpoint: {last_checkpoint}")
        trainer.train(resume_from_checkpoint=last_checkpoint)
    else:
        print("No checkpoint found, starting training from scratch.")
        trainer.train()
    print("finished training!")

    # do final eval on best model just to double check
    run_generation_eval(model, ds["eval"], image_root, tokenizer, processor)

    run.finish()
    print("done!")