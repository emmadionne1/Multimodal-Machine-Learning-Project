"""
Base architecture for our custom VLM
"""

from pathlib import Path
import zipfile
import evaluate
import wandb
import config
import numpy as np
import argparse
from pprint import pprint
from datasets import load_dataset, DatasetDict
from huggingface_hub import snapshot_download
from PIL import Image
import matplotlib.pyplot as plt
from transformers import AutoProcessor, AutoModel, AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments, TrainerCallback
import torch

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 8
bleu = evaluate.load("bleu")

def create_dataset(overfit_check=False):
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

    # step 5: optional maybe, lets split up into train/eval/test
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

# TODO: i had to make this diff than the notebook code - had to load the vision model block specifically
# this might need to change if we mess around with other types of visual encoders
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
            images.append(Image.open(self.image_root / ex["image"]).convert("RGB"))

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
    def __init__(self, trainer, eval_ds, image_root, tokenizer, processor, save_path, overfit):
        self.trainer = trainer
        self.eval_ds = eval_ds
        self.image_root = image_root
        self.tokenizer = tokenizer
        self.processor = processor
        self.save_path = save_path
        self.best_bleu = -1.0
        self.best_loss = 100000
        self.overfit = overfit

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if metrics is None:
            metrics = {}

        if self.overfit:
            gen_metrics = run_generation_eval(
                self.trainer.model, 
                self.eval_ds,
                self.image_root,
                self.tokenizer,
                self.processor
            )
            metrics.update(gen_metrics)

            if metrics["eval_bleu"] > self.best_bleu:
                self.best_bleu = metrics["eval_bleu"]
                torch.save(self.trainer.model.projector.state_dict(), self.save_path)

            self.trainer.log(gen_metrics)
        else:
            if metrics["eval_loss"] < self.best_loss:
                self.best_loss = metrics["eval_loss"]
                torch.save(self.trainer.model.projector.state_dict(), self.save_path)
                print(f"Eval loss for this epoch: {metrics['eval_loss']}, best eval loss seen: {self.best_loss}")

        return control
        
# TODO: speed up eval by doing it in batches OR running eval on only a subset
def run_generation_eval(model, ds, image_root, tokenizer, processor):
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
                images=[Image.open(image_root / ex["image"]).convert("RGB")], return_tensors="pt"
            ).pixel_values.to(DEVICE)

            # run generation
            gen_text = model.generate_text(prompt_ids, attention_mask, pixel_values, max_new_tokens=64)
            preds.append(gen_text)
            golds.append(ex["completion"])

            print("PRINTING OUT OUTPUTS")
            print("GENERATED:\n", preds[-1])
            print("TARGET:\n", golds[-1])

    # compute metrics
    bleu_scores = []
    for pred, gold in zip(preds, golds):
        if len(pred.strip()) == 0:
            bleu_scores.append(0.0)
        else:
            bleu_scores.append(bleu.compute(predictions=[pred], references=[gold])["bleu"])
    metrics = {"eval_bleu": float(np.mean(bleu_scores))}
    print(metrics)
    return(metrics)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment_name", type=str, required=True)
    parser.add_argument("--vision_model_name", type=str, default="google/siglip2-so400m-patch16-512") 
    parser.add_argument("--language_model_name", type=str, default="Qwen/Qwen3-4B-Instruct-2507")
    parser.add_argument("--spatial_merge_size", type=int, default=2)
    parser.add_argument("--overfit_check", action="store_true")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-4)
    args = parser.parse_args()

    EXPERIMENT_NAME = args.experiment_name
    OVERFIT_CHECK = True if args.overfit_check else False
    SAVE_PATH = f"./outputs/{EXPERIMENT_NAME}_proj_final_weights.pt"
    LR = args.lr
    EPOCHS = args.epochs
    SPATIAL_MERGE_SIZE = args.spatial_merge_size
    VISION_MODEL_NAME = args.vision_model_name
    LANGUAGE_MODEL_NAME = args.language_model_name

    # initialize wandb
    wandb.login(key=config.WANDB_KEY)
    run = wandb.init(project=f"base_architecture", name=EXPERIMENT_NAME)

    # load dataset and dir where raw images are stored
    ds, image_root = create_dataset(overfit_check=OVERFIT_CHECK)

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
    model.freeze_backbones()
    collator = CustomVLMCollator(tokenizer, processor, image_root)

    training_args = TrainingArguments(
        output_dir=f"./outputs/{EXPERIMENT_NAME}",
        per_device_train_batch_size=12,
        per_device_eval_batch_size=12,
        gradient_accumulation_steps=4,
        learning_rate=LR,
        num_train_epochs=EPOCHS,
        run_name=EXPERIMENT_NAME,
        bf16=True,
        remove_unused_columns=False,
        eval_strategy="steps",
        eval_steps=1000,
        logging_steps=2,
        save_strategy="no",
        report_to="wandb",
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        seed=SEED
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=ds["train"],
        eval_dataset=ds["eval"],
        data_collator=collator,
    )
    trainer.add_callback(GenEvalCallback(trainer, ds["eval"], image_root, tokenizer, processor, SAVE_PATH, OVERFIT_CHECK))
    Path(SAVE_PATH).parent.mkdir(parents=True, exist_ok=True)

    # not resuming from checkpoint, as we're only saving the proj layer weights
    # we cant simply just load those, we'd need to save the scheduler, optim states too and im lazy
    print("beginning training!")
    trainer.train()
    print("finished training!")

    # do final eval on best model just to double check
    best_state = torch.load(SAVE_PATH, map_location=DEVICE)
    model.projector.load_state_dict(best_state)
    run_generation_eval(model, ds["eval"], image_root, tokenizer, processor)

    run.finish()
    print("done!")