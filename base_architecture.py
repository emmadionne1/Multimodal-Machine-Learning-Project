"""
Base architecture for our custom VLM
"""

from pathlib import Path
import zipfile
import evaluate
import wandb
import re
import os
import config
import numpy as np
from collections import Counter
import argparse
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
from baselines.multimodal_prompting_baselines import is_answer_correct_relaxed
import torch

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 8
bleu = evaluate.load("bleu")
rouge = evaluate.load("rouge")

def convert_chartqa_example(ex):
    answer = ex["label"][0]
    prompt = (
        "<image>\n"
        "You are a chart question-answering assistant. Respond with the final answer only, no explanations.\n"
        f"Question: {ex['query'].strip()}\n"
        "Answer:"
    )
    return {"image": ex["image"], "prompt": prompt, "completion": str(answer).strip()}

def convert_summarization_example(ex):
    question, answer = ex["texts"][0]["user"], ex["texts"][0]["assistant"]
    prompt = (
        "<image>\n"
        "You are a chart question-answering assistant. Respond to the question with the final answer only, no explanations, and in detail.\n"
        f"Question: {question.strip()}\n"
        "Answer:"
    )
    return {"image": ex["images"][0], "prompt": prompt, "completion": str(answer).strip()}

def convert_table_example(ex):
    answer = ex["text"]
    prompt = (
        "<image>\n"
        "You are a chart assistant. Extract all the information from the table and convert it into a Markdown table. Output only the table, nothing else.\n"
        "Answer:"
    )
    return {"image": ex["image"], "prompt": prompt, "completion": str(answer).strip()}

def create_chartqa_dataset(overfit_check=False):
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

    ds = DatasetDict({
        "train": train_raw.map(convert_chartqa_example),
        "eval": eval_raw.map(convert_chartqa_example),
        "test": test_raw.map(convert_chartqa_example),
    })

    print(ds)
    print(ds["train"][0])

    return ds, None

def create_summarization_dataset(overfit_check=False):
    raw = load_dataset("HuggingFaceM4/the_cauldron", "chart2text")["train"]
    split_1 = raw.train_test_split(test_size=0.20, seed=SEED, shuffle=True)
    split_2 = split_1["test"].train_test_split(test_size=0.50, seed=SEED, shuffle=True)
    ds = DatasetDict({"train": split_1["train"], "eval": split_2["train"], "test": split_2["test"]})

    if overfit_check:
        ds["train"] = ds["train"].select(range(3))
        ds["eval"] = ds["train"]
        ds["test"] = ds["train"]

    ds = ds.map(convert_summarization_example)
    ds = ds.remove_columns("images")

    print(ds)
    print(ds["train"][0])

    return ds, None

def create_table_dataset(overfit_check=False):
    raw = load_dataset("chiragtubakad/chart-to-table")["train"]
    split_1 = raw.train_test_split(test_size=0.10, seed=SEED, shuffle=True)
    split_2 = split_1["test"].train_test_split(test_size=0.50, seed=SEED, shuffle=True)
    ds = DatasetDict({"train": split_1["train"], "eval": split_2["train"], "test": split_2["test"]})

    if overfit_check:
        ds["train"] = ds["train"].select(range(3))
        ds["eval"] = ds["train"]
        ds["test"] = ds["train"]

    ds = ds.map(convert_table_example)

    print(ds)
    print(ds["train"][0])

    return ds, None

def create_pretrain_dataset(overfit_check=False):
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

    # step 6: for debugging only
    if overfit_check:
        ds["train"] = ds["train"].select(range(3))
        ds["eval"] = ds["train"]
        ds["test"] = ds["train"]
        print(ds)

    return ds, image_root

def create_dataset(task, overfit_check=False):
    if task == "chartqa":
        return create_chartqa_dataset(overfit_check=overfit_check)
    elif task == "pretrain":
        return create_pretrain_dataset(overfit_check=overfit_check)
    elif task == "summarization":
        return create_summarization_dataset(overfit_check=overfit_check)
    elif task == "table":
        return create_table_dataset(overfit_check=overfit_check)
    else:
        print("huh")
        return None, None

def load_vision_model(vision_model_name):
    print(f"Loading Vision: {vision_model_name}")
    processor = AutoProcessor.from_pretrained(vision_model_name, use_fast=True)
    vision_model = AutoModel.from_pretrained(vision_model_name).to(DEVICE)
    vision_model.eval()
    return vision_model.vision_model, processor

def load_language_model(language_model_name):
    print(f"Loading LLM: {language_model_name}")
    tokenizer = AutoTokenizer.from_pretrained(language_model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    llm = AutoModelForCausalLM.from_pretrained(language_model_name, torch_dtype=torch.bfloat16, trust_remote_code=True)
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
    
def load_image(image_root, image_field):
    if isinstance(image_field, Image.Image):
        return image_field.convert("RGB")
    return Image.open(image_root / image_field).convert("RGB")
    
class CustomVLMCollator():
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
            final_embeds.append(merged_embeds[:MAX_LEN])

            image_labels = torch.full((image_features.size(1),), -100, dtype=labels.dtype, device=labels.device)
            merged_labels = torch.cat([labels[i, :img_idx], image_labels, labels[i, img_idx + 1 :]], dim=0)
            final_labels.append(merged_labels[:MAX_LEN])

            image_mask = torch.ones(image_features.size(1), dtype=attention_mask.dtype, device=attention_mask.device)
            merged_mask = torch.cat([attention_mask[i, :img_idx], image_mask, attention_mask[i, img_idx + 1 :]], dim=0)
            final_masks.append(merged_mask[:MAX_LEN])

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
    def __init__(self, trainer, eval_ds, image_root, tokenizer, processor, overfit, task):
        self.trainer = trainer
        self.eval_ds = eval_ds
        self.image_root = image_root
        self.tokenizer = tokenizer
        self.processor = processor
        self.overfit = overfit
        self.task = task

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
                self.task
            )
            metrics.update(gen_metrics)
            self.trainer.log(gen_metrics)

        return control
        
def run_generation_eval(model, ds, image_root, tokenizer, processor, task):
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
            pixel_values = processor(
                images=[load_image(image_root, ex["image"])], return_tensors="pt"
            ).pixel_values.to(DEVICE)

            # run generation
            gen_text = model.generate_text(prompt_ids, attention_mask, pixel_values, max_new_tokens=MAX_NEW_TOKENS)
            preds.append(gen_text)
            golds.append(ex["completion"])

            print("PRINTING OUT OUTPUTS")
            print("GENERATED:\n", preds[-1])
            print("TARGET:\n", golds[-1])

    # compute metrics
    metrics = compute_metrics(preds, golds, task)
    print(metrics)
    return(metrics)

def table_cell_f1(pred, gold):
    pred_cells = [c.strip().lower() for c in re.split(r"[\n,\|]+", pred) if c.strip()]
    gold_cells = [c.strip().lower() for c in re.split(r"[\n,\|]+", gold) if c.strip()]

    pred_counter, gold_counter = Counter(pred_cells), Counter(gold_cells)
    overlap = sum((pred_counter & gold_counter).values())

    if len(pred_cells) == 0 and len(gold_cells) == 0:
        return 1.0
    elif len(pred_cells) == 0 or len(gold_cells) == 0:
        return 0.0

    precision = overlap / len(pred_cells)
    recall = overlap / len(gold_cells)

    if precision + recall == 0:
        return 0.0
    else:
        return 2 * precision * recall / (precision + recall)

def compute_metrics(preds, golds, task):
    if task == "chartqa":
        relaxed_scores = [is_answer_correct_relaxed(pred, gold) for pred, gold in zip(preds, golds)]
        metrics = {"eval_relaxed_accuracy": float(np.mean(relaxed_scores))}
    elif task == "pretrain":
        bleu_scores = []
        for pred, gold in zip(preds, golds):
            if len(pred.strip()) == 0:
                bleu_scores.append(0.0)
            else:
                bleu_scores.append(bleu.compute(predictions=[pred], references=[gold])["bleu"])
        metrics = {"eval_bleu": float(np.mean(bleu_scores))}
    elif task == "summarization":
        bleu_scores, rouge_scores = [], []
        for pred, gold in zip(preds, golds):
            if len(pred.strip()) == 0:
                bleu_scores.append(0.0)
                rouge_scores.append(0.0)
            else:
                bleu_scores.append(bleu.compute(predictions=[pred], references=[gold])["bleu"])
                rouge_scores.append(rouge.compute(predictions=[pred], references=[gold])["rougeL"])
        metrics = {"eval_bleu": float(np.mean(bleu_scores)), "eval_rouge": float(np.mean(rouge_scores))}
    elif task == "table":
        f1_list = [table_cell_f1(pred, gold) for pred, gold in zip(preds, golds)]
        rms_f1 = np.sqrt(np.mean(np.square(f1_list)))
        metrics = {"eval_rms_f1": float(rms_f1)}
    else:
        print("huh")

    return metrics

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
    parser.add_argument("--vision_model_name", type=str, default="google/siglip2-so400m-patch16-512") 
    parser.add_argument("--language_model_name", type=str, default="Qwen/Qwen3-4B-Instruct-2507")
    parser.add_argument("--spatial_merge_size", type=int, default=2)
    parser.add_argument("--overfit_check", action="store_true")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--lr", type=float, default=5e-3)
    parser.add_argument("--per_device_train_batch_size", type=int, default=16)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--task", choices=['pretrain', 'chartqa', 'summarization', 'table'])
    parser.add_argument("--trained_weights", type=str, default=None)
    parser.add_argument("--truncation_length", type=int, default=2048)
    parser.add_argument("--max_new_tokens", type=int, default=64)
    args = parser.parse_args()

    EXPERIMENT_NAME = args.experiment_name
    OVERFIT_CHECK = True if args.overfit_check else False
    LR = args.lr
    EPOCHS = args.epochs
    SPATIAL_MERGE_SIZE = args.spatial_merge_size
    VISION_MODEL_NAME = args.vision_model_name
    LANGUAGE_MODEL_NAME = args.language_model_name
    OUTPUT_DIR = f"./outputs/{EXPERIMENT_NAME}"
    PER_DEVICE_TRAIN_BATCH_SIZE = args.per_device_train_batch_size
    GRADIENT_ACCUMULATION_STEPS = args.gradient_accumulation_steps
    TASK = args.task
    TRAINED_WEIGHTS = args.trained_weights
    MAX_LEN = args.truncation_length
    MAX_NEW_TOKENS = args.max_new_tokens

    # initialize wandb
    wandb.login(key=config.WANDB_KEY)
    run = wandb.init(project=f"base_architecture", name=EXPERIMENT_NAME)

    # load dataset and dir where raw images are stored
    ds, image_root = create_dataset(TASK, overfit_check=OVERFIT_CHECK)

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
    if TRAINED_WEIGHTS is not None:
        proj_layer.load_state_dict(torch.load(TRAINED_WEIGHTS, map_location=DEVICE))

    # load whole model now
    model = CustomVLM(vision_model, processor, language_model, tokenizer, proj_layer, image_token_id)
    model.freeze_backbones()
    collator = CustomVLMCollator(tokenizer, processor, image_root)

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=PER_DEVICE_TRAIN_BATCH_SIZE,
        per_device_eval_batch_size=PER_DEVICE_TRAIN_BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
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
        eval_strategy="epoch",
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
    trainer.add_callback(GenEvalCallback(trainer, ds["eval"], image_root, tokenizer, processor, OVERFIT_CHECK, TASK))

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
    run_generation_eval(model, ds["eval"], image_root, tokenizer, processor, TASK)

    run.finish()
    print("done!")