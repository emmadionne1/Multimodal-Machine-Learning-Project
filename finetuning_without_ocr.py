import gc
import json
import random
import shutil
import easyocr
from peft import LoraConfig, TaskType, get_peft_model, PeftModel
from tqdm import tqdm
from baselines.multimodal_prompting_baselines import is_answer_correct_relaxed, is_answer_correct_exact
from base_architecture import *
from io import BytesIO
import base64
from PIL import Image


def decode_chartqapro_image(image):
    if isinstance(image, Image.Image):
        return image.convert("RGB")

    if isinstance(image, bytes):
        return Image.open(BytesIO(image)).convert("RGB")

    if isinstance(image, list):
        try:
            return Image.open(BytesIO(bytes(image))).convert("RGB")
        except Exception as e:
            print(f"Warning: could not decode list image: {e}")
            return None

    if isinstance(image, str):
        # ChartQAPro seems to store some images as base64-ish JPEG strings
        try:
            return Image.open(BytesIO(base64.b64decode(image))).convert("RGB")
        except Exception as e:
            print(f"Warning: could not decode string image: {e}")
            return None

    print(f"Warning: unsupported image type: {type(image)}")
    return None

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

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

def report_trainable_parameters(model):
    total = 0
    trainable = 0
    for p in model.parameters():
        n = p.numel()
        total += n
        if p.requires_grad:
            trainable += n
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

class LoRACustomVLM(CustomVLM):
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
        return embed_layer(input_ids)

    def forward(self, input_ids=None, attention_mask=None, labels=None, pixel_values=None, **kwargs):
        vision_outputs = self._vision_forward(pixel_values)
        vision_outputs = vision_outputs.to(self.projector.norm.weight.dtype)

        image_features = self.projector(vision_outputs)
        inputs_embeds = self._token_embed(input_ids)

        final_embeds = []
        final_labels = []
        final_masks = []

        for i in range(input_ids.shape[0]):
            image_positions = (input_ids[i] == self.image_token_id).nonzero(as_tuple=True)[0]
            if len(image_positions) == 0:
                raise ValueError("Could not find <image> token in input_ids.")
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
    def generate_text(self, input_ids, attention_mask, pixel_values, max_new_tokens=32):
        self.eval()

        vision_outputs = self.vision_model(pixel_values=pixel_values).last_hidden_state.to(
            self.projector.norm.weight.dtype
        )
        image_features = self.projector(vision_outputs)
        inputs_embeds = self.language_model.get_input_embeddings()(input_ids)

        image_positions = (input_ids[0] == self.image_token_id).nonzero(as_tuple=True)[0]
        if len(image_positions) == 0:
            raise ValueError("Could not find <image> token in input_ids during generation.")
        img_idx = image_positions[0].item()

        merged_embeds = torch.cat(
            [inputs_embeds[0, :img_idx, :], image_features[0], inputs_embeds[0, img_idx + 1 :, :]],
            dim=0,
        )
        image_mask = torch.ones(
            image_features.size(1),
            dtype=attention_mask.dtype,
            device=attention_mask.device,
        )
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

def save_model_artifacts(model, save_dir, vision_model_name, language_model_name, spatial_merge_size):
    model = unwrap_model(model)
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    adapter_dir = save_dir / "lora_adapter"
    tokenizer_dir = save_dir / "tokenizer"
    processor_dir = save_dir / "processor"
    projector_path = save_dir / "projector_weights.pt"
    metadata_path = save_dir / "checkpoint_metadata.json"

    model.language_model.save_pretrained(adapter_dir, safe_serialization=True)
    model.tokenizer.save_pretrained(tokenizer_dir)
    model.processor.save_pretrained(processor_dir)
    torch.save(model.projector.state_dict(), projector_path)

    metadata = {
        "vision_model_name": vision_model_name,
        "language_model_name": language_model_name,
        "spatial_merge_size": spatial_merge_size,
        "image_token_id": model.image_token_id,
        "lora_adapter_path": str(adapter_dir),
        "tokenizer_path": str(tokenizer_dir),
        "processor_path": str(processor_dir),
        "projector_weights_path": str(projector_path),
    }
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

def load_saved_model(checkpoint_dir):
    ckpt_dir = Path(checkpoint_dir)
    metadata_path = ckpt_dir / "checkpoint_metadata.json"
    adapter_dir = ckpt_dir / "lora_adapter"
    tokenizer_dir = ckpt_dir / "tokenizer"
    processor_dir = ckpt_dir / "processor"
    projector_path = ckpt_dir / "projector_weights.pt"

    with open(metadata_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    vision_model_name = metadata["vision_model_name"]
    language_model_name = metadata["language_model_name"]
    spatial_merge_size = metadata["spatial_merge_size"]
    image_token_id = metadata["image_token_id"]

    vision_model, _ = load_vision_model(vision_model_name)
    v_conf = vision_model.config.vision_config if hasattr(vision_model.config, "vision_config") else vision_model.config
    vision_dim = v_conf.hidden_size

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    processor = AutoProcessor.from_pretrained(processor_dir, trust_remote_code=True)

    base_llm = AutoModelForCausalLM.from_pretrained(
        language_model_name,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    target_vocab_size = len(tokenizer)
    current_vocab_size = base_llm.get_input_embeddings().weight.shape[0]
    if current_vocab_size != target_vocab_size:
        base_llm.resize_token_embeddings(target_vocab_size)
        if hasattr(base_llm, "tie_weights"):
            base_llm.tie_weights()

    language_model = PeftModel.from_pretrained(base_llm, adapter_dir, is_trainable=False)
    llm_dim = language_model.config.hidden_size

    projector = Qwen3VLProjector(
        vision_dim=vision_dim,
        llm_dim=llm_dim,
        spatial_merge_size=spatial_merge_size,
    ).to(dtype=torch.bfloat16)
    projector_state = torch.load(projector_path, map_location="cpu", weights_only=False)
    projector.load_state_dict(projector_state)

    model = LoRACustomVLM(
        vision_model=vision_model,
        processor=processor,
        language_model=language_model,
        tokenizer=tokenizer,
        projector=projector,
        image_token_id=image_token_id,
    ).to(DEVICE, dtype=torch.bfloat16)

    model.eval()
    return model, tokenizer, processor, metadata

class EpochArtifactCallback(TrainerCallback):
    def __init__(self, artifact_root, vision_model_name, language_model_name, spatial_merge_size, keep_last_n=2):
        self.artifact_root = Path(artifact_root)
        self.vision_model_name = vision_model_name
        self.language_model_name = language_model_name
        self.spatial_merge_size = spatial_merge_size
        self.keep_last_n = keep_last_n

    def _cleanup_old_checkpoints(self):
        epoch_dirs = []
        for p in self.artifact_root.glob("epoch_*"):
            if p.is_dir():
                try:
                    epoch_num = int(p.name.split("_")[-1])
                    epoch_dirs.append((epoch_num, p))
                except ValueError:
                    continue

        epoch_dirs.sort(key=lambda x: x[0])
        to_delete = epoch_dirs[:-self.keep_last_n] if len(epoch_dirs) > self.keep_last_n else []

        for _, path in to_delete:
            print(f"Deleting old checkpoint {path}")
            shutil.rmtree(path, ignore_errors=True)

    def on_epoch_end(self, args, state, control, model=None, **kwargs):
        if not state.is_world_process_zero:
            return control

        epoch_num = int(round(state.epoch)) if state.epoch is not None else 0
        save_dir = self.artifact_root / f"epoch_{epoch_num}"

        save_model_artifacts(
            model=model,
            save_dir=save_dir,
            vision_model_name=self.vision_model_name,
            language_model_name=self.language_model_name,
            spatial_merge_size=self.spatial_merge_size,
        )

        state.save_to_json(str(save_dir / "trainer_state.json"))
        print(f"Saved epoch artifact to {save_dir}")

        self._cleanup_old_checkpoints()
        return control

def get_model_device(model):
    model = unwrap_model(model)
    return next(model.language_model.parameters()).device

def normalize_example(example, dataset_name):
    dataset_name_lower = dataset_name.lower()

    if "chartqapro" in dataset_name_lower:
        image = example.get("image", example.get("Image"))
        question = example.get("query", example.get("Question"))
        answer = example.get("label", example.get("Answer"))

        if isinstance(question, list):
            question = question[0]
        if isinstance(answer, list):
            answer = answer[0]

        prompt = (
            "<image>\n"
            "You are a chart question-answering assistant. Respond with the final answer only, no explanations.\n"
            f"Question: {str(question).strip()}\n"
            "Answer:"
        )
        return {"image": image, "prompt": prompt, "completion": str(answer).strip()}
    else:
        return convert_chartqa_example(example)

def build_ocr_prompt(image, query, max_ocr_words, reader):
    if image.mode != "RGB":
        image = image.convert("RGB")
    results = reader.readtext(np.array(image))
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

    return (
        "<image>\n"
        f"You are given the image and its text extracted from a chart via OCR:\n{ocr_text}\n"
        "You are a chart question-answering assistant. Respond with the final answer only, no explanations.\n"
        f"Question: {str(query).strip()}\n"
        "Answer:"
    )

@torch.no_grad()
def evaluate_dataset(model, tokenizer, processor, dataset_name, split, max_new_tokens, use_ocr, max_ocr_words):
    raw = load_dataset(dataset_name)
    if split not in raw and split == "val" and "validation" in raw:
        split = "validation"
    ds = raw[split]

    reader = None
    if use_ocr:
        print(f"Initializing EasyOCR for {dataset_name}:{split}")
        reader = easyocr.Reader(["en"], gpu=torch.cuda.is_available())

    relaxed_correct = 0
    exact_correct = 0
    shown = 0
    skipped = 0

    for ex in tqdm(ds, desc=f"Evaluating {dataset_name}:{split}"):
        norm = normalize_example(ex, dataset_name)
        image = norm["image"]
        ground_truth = norm["completion"]

        if "chartqapro" in dataset_name.lower():
            image = decode_chartqapro_image(image)
            if image is None:
                skipped += 1
                continue

        if use_ocr:
            if "query" in ex:
                query = ex["query"]
            elif "Question" in ex:
                query = ex["Question"][0] if isinstance(ex["Question"], list) else ex["Question"]
            else:
                raise KeyError("Could not find query/question field for OCR prompt.")
            prompt = build_ocr_prompt(image, query, max_ocr_words=max_ocr_words, reader=reader)
        else:
            prompt = norm["prompt"]

        prompt_text = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )

        if hasattr(image, "mode") and image.mode != "RGB":
            image = image.convert("RGB")

        pixel_values = processor(images=image, return_tensors="pt").pixel_values.to(DEVICE)
        inputs = tokenizer(prompt_text, return_tensors="pt", add_special_tokens=False)
        input_ids = inputs.input_ids.to(DEVICE)
        attention_mask = torch.ones_like(input_ids).to(DEVICE)

        prediction_text = model.generate_text(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            max_new_tokens=max_new_tokens,
        )
        prediction = prediction_text.strip()

        if shown < 5:
            print(f"Example {shown + 1}")
            print(f"GT:   {ground_truth}")
            print(f"PRED: {prediction}")
            shown += 1

        if is_answer_correct_relaxed(prediction, ground_truth):
            relaxed_correct += 1
        if is_answer_correct_exact(prediction, ground_truth):
            exact_correct += 1

    if reader is not None:
        del reader
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    evaluated = len(ds) - skipped
    metrics = {
        "dataset": dataset_name,
        "split": split,
        "use_ocr": use_ocr,
        "relaxed_accuracy": relaxed_correct / evaluated if evaluated > 0 else 0.0,
        "exact_accuracy": exact_correct / evaluated if evaluated > 0 else 0.0,
        "num_examples": len(ds),
        "num_skipped": skipped,
        "num_evaluated": evaluated,
        "num_correct_relaxed": relaxed_correct,
        "num_correct_exact": exact_correct,
    }
    print(metrics)
    return metrics

def run_all_evals(model, tokenizer, processor, max_new_tokens, run_ocr_eval, max_ocr_words):
    all_metrics = {}

    print("Running ChartQA eval")
    chartqa_eval = evaluate_dataset(
        model=model,
        tokenizer=tokenizer,
        processor=processor,
        dataset_name="HuggingFaceM4/ChartQA",
        split="val",
        max_new_tokens=max_new_tokens,
        use_ocr=False,
        max_ocr_words=max_ocr_words,
    )
    all_metrics["chartqa_val_relaxed_accuracy"] = chartqa_eval["relaxed_accuracy"]
    all_metrics["chartqa_val_exact_accuracy"] = chartqa_eval["exact_accuracy"]

    print("Running ChartQA test")
    chartqa_test = evaluate_dataset(
        model=model,
        tokenizer=tokenizer,
        processor=processor,
        dataset_name="HuggingFaceM4/ChartQA",
        split="test",
        max_new_tokens=max_new_tokens,
        use_ocr=False,
        max_ocr_words=max_ocr_words,
    )
    all_metrics["chartqa_test_relaxed_accuracy"] = chartqa_test["relaxed_accuracy"]
    all_metrics["chartqa_test_exact_accuracy"] = chartqa_test["exact_accuracy"]

    print("Running ChartQAPro test")
    chartqapro_test = evaluate_dataset(
        model=model,
        tokenizer=tokenizer,
        processor=processor,
        dataset_name="ahmed-masry/ChartQAPro",
        split="test",
        max_new_tokens=max_new_tokens,
        use_ocr=False,
        max_ocr_words=max_ocr_words,
    )
    all_metrics["chartqapro_test_relaxed_accuracy"] = chartqapro_test["relaxed_accuracy"]
    all_metrics["chartqapro_test_exact_accuracy"] = chartqapro_test["exact_accuracy"]

    if run_ocr_eval:
        print("Running ChartQA OCR eval")
        chartqa_ocr_eval = evaluate_dataset(
            model=model,
            tokenizer=tokenizer,
            processor=processor,
            dataset_name="HuggingFaceM4/ChartQA",
            split="val",
            max_new_tokens=max_new_tokens,
            use_ocr=True,
            max_ocr_words=max_ocr_words,
        )
        all_metrics["chartqa_ocr_eval_relaxed_accuracy"] = chartqa_ocr_eval["relaxed_accuracy"]
        all_metrics["chartqa_ocr_eval_exact_accuracy"] = chartqa_ocr_eval["exact_accuracy"]

        print("Running ChartQA OCR test")
        chartqa_ocr_test = evaluate_dataset(
            model=model,
            tokenizer=tokenizer,
            processor=processor,
            dataset_name="HuggingFaceM4/ChartQA",
            split="test",
            max_new_tokens=max_new_tokens,
            use_ocr=True,
            max_ocr_words=max_ocr_words,
        )
        all_metrics["chartqa_ocr_test_relaxed_accuracy"] = chartqa_ocr_test["relaxed_accuracy"]
        all_metrics["chartqa_ocr_test_exact_accuracy"] = chartqa_ocr_test["exact_accuracy"]

        print("Running ChartQAPro OCR test")
        chartqapro_ocr_test = evaluate_dataset(
            model=model,
            tokenizer=tokenizer,
            processor=processor,
            dataset_name="ahmed-masry/ChartQAPro",
            split="test",
            max_new_tokens=max_new_tokens,
            use_ocr=True,
            max_ocr_words=max_ocr_words,
        )
        all_metrics["chartqapro_ocr_test_relaxed_accuracy"] = chartqapro_ocr_test["relaxed_accuracy"]
        all_metrics["chartqapro_ocr_test_exact_accuracy"] = chartqapro_ocr_test["exact_accuracy"]

    print("All eval metrics")
    print(all_metrics)
    return all_metrics

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment_name", type=str, required=True)
    parser.add_argument("--projector_weights", type=str, default=None)
    parser.add_argument("--eval_only", action="store_true")
    parser.add_argument("--ckpt_dir", type=str, default=None)

    parser.add_argument("--vision_model_name", type=str, default="google/siglip2-so400m-patch16-512")
    parser.add_argument("--language_model_name", type=str, default="Qwen/Qwen3-4B-Instruct-2507")
    parser.add_argument("--spatial_merge_size", type=int, default=2)

    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--per_device_train_batch_size", type=int, default=16)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=8)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)

    parser.add_argument("--lora_r", type=int, default=32)
    parser.add_argument("--lora_alpha", type=int, default=64)
    parser.add_argument("--lora_dropout", type=float, default=0.05)

    parser.add_argument("--warmup_ratio", type=float, default=0.03)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--logging_steps", type=int, default=50)

    parser.add_argument("--max_new_tokens", type=int, default=32)
    parser.add_argument("--run_ocr_eval", action="store_true")
    parser.add_argument("--max_ocr_words", type=int, default=1000)

    parser.add_argument("--overfit_check", action="store_true")
    args = parser.parse_args()

    set_seed(SEED)

    output_root = Path("./outputs") / args.experiment_name
    final_model_dir = output_root / "final_model"
    epoch_artifact_dir = output_root / "epoch_artifacts"
    trainer_output_dir = output_root / "trainer_outputs"
    output_root.mkdir(parents=True, exist_ok=True)

    if args.eval_only:
        print(f"Loading saved model from {args.ckpt_dir}")
        model, tokenizer, processor, metadata = load_saved_model(args.ckpt_dir)
        print(f"Loaded vision model: {metadata['vision_model_name']}")
        print(f"Loaded base LLM: {metadata['language_model_name']}")

        eval_metrics = run_all_evals(
            model=model,
            tokenizer=tokenizer,
            processor=processor,
            max_new_tokens=args.max_new_tokens,
            run_ocr_eval=args.run_ocr_eval,
            max_ocr_words=args.max_ocr_words,
        )
        print("Done")
        return

    wandb.login(key=config.WANDB_KEY)
    run = wandb.init(
        project="chartqa_lora_projector_finetune",
        name=args.experiment_name,
        config=vars(args),
    )

    print("Loading ChartQA")
    raw = load_dataset("HuggingFaceM4/ChartQA")
    if "val" not in raw and "validation" not in raw:
        raise ValueError("Expected ChartQA to have 'val' or 'validation'.")
    if "test" not in raw:
        raise ValueError("Expected ChartQA to have 'test'.")

    ds, image_root = create_chartqa_dataset(overfit_check=args.overfit_check)

    print("Loading models")
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
        vision_dim=vision_dim,
        llm_dim=llm_dim,
        spatial_merge_size=args.spatial_merge_size,
    ).to(dtype=torch.bfloat16)

    print(f"Loading projector weights from {args.projector_weights}")
    projector_state = torch.load(args.projector_weights, map_location="cpu", weights_only=False)
    projector.load_state_dict(projector_state)

    model = LoRACustomVLM(
        vision_model=vision_model,
        processor=processor,
        language_model=language_model,
        tokenizer=tokenizer,
        projector=projector,
        image_token_id=image_token_id,
    ).to(dtype=torch.bfloat16)

    model.set_trainable(train_vision=False, train_language=True, train_projector=True)
    report_trainable_parameters(model)

    collator = CustomVLMCollator(tokenizer, processor, image_root)

    training_args = TrainingArguments(
        output_dir=str(trainer_output_dir),
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.lr,
        num_train_epochs=args.epochs,
        run_name=args.experiment_name,
        bf16=True,
        remove_unused_columns=False,
        report_to="wandb",
        lr_scheduler_type="cosine",
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        seed=SEED,
        logging_strategy="steps",
        logging_steps=args.logging_steps,
        eval_strategy="epoch",
        save_strategy="no",
        save_safetensors=False,
        ddp_find_unused_parameters=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=ds["train"],
        eval_dataset=ds["eval"],
        data_collator=collator,
        callbacks=[
            EpochArtifactCallback(
                artifact_root=epoch_artifact_dir,
                vision_model_name=args.vision_model_name,
                language_model_name=args.language_model_name,
                spatial_merge_size=args.spatial_merge_size,
                keep_last_n=2,
            )
        ],
    )

    print("Starting training")
    trainer.train()

    print("Running validation loss eval")
    val_metrics = trainer.evaluate()
    print(val_metrics)
    wandb.log({f"val/{k}": v for k, v in val_metrics.items()})

    print("Saving final model")
    save_model_artifacts(
        model=model,
        save_dir=final_model_dir,
        vision_model_name=args.vision_model_name,
        language_model_name=args.language_model_name,
        spatial_merge_size=args.spatial_merge_size,
    )

    print("Reloading epoch 2 (assumed best checkpoint) for evaluation")
    reloaded_model, reloaded_tokenizer, reloaded_processor, metadata = load_saved_model(output_root / "epoch_artifacts" / "epoch_2")

    eval_metrics = run_all_evals(
        model=reloaded_model,
        tokenizer=reloaded_tokenizer,
        processor=reloaded_processor,
        max_new_tokens=args.max_new_tokens,
        run_ocr_eval=args.run_ocr_eval,
        max_ocr_words=args.max_ocr_words,
    )
    wandb.log(eval_metrics)

    print(f"Epoch artifacts: {epoch_artifact_dir}")
    print(f"Trainer outputs: {trainer_output_dir}")
    print(f"Final saved model: {final_model_dir}")

    run.finish()


if __name__ == "__main__":
    main()