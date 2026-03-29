import argparse
import torch
import wandb
import os
import gc
import numpy as np
import easyocr
from datasets import load_dataset
from transformers import TrainingArguments
from tqdm import tqdm

# Import custom architectures and metrics
from base_architecture import (
    load_vision_model, load_language_model,
    Qwen3VLProjector, CustomVLM, CustomVLMCollator, ProjectorOnlyTrainer,
    DEVICE, SEED
)
from baselines.multimodal_prompting_baselines import is_answer_correct_relaxed
import config

def evaluate_test_set(model, test_ds, tokenizer, processor, prefix="Pre-Train"):
    """Helper function to run the evaluation loop."""
    print(f"\n{'='*50}")
    print(f"[{prefix}] Evaluating Generative Relaxed Accuracy on Test Set...")
    
    # --- PRINT A SAMPLE EVALUATION PROMPT ---
    if len(test_ds) > 0:
        sample_msg = [{"role": "user", "content": test_ds[0]["prompt"]}]
        sample_prompt = tokenizer.apply_chat_template(sample_msg, tokenize=False, add_generation_prompt=True)
        print("\n--- SAMPLE EVALUATION PROMPT (Index 0) ---")
        print(sample_prompt)
        print("------------------------------------------\n")
        
    model.eval()
    correct = 0
    total = len(test_ds)
    
    for sample in tqdm(test_ds, desc=f"{prefix} Test Evaluation"):
        messages = [{"role": "user", "content": sample["prompt"]}]
        prompt_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        
        ground_truth = str(sample["completion"]).strip()
        
        image = sample["image"]
        if image.mode != "RGB":
            image = image.convert("RGB")
            
        pixel_values = processor(images=image, return_tensors="pt").pixel_values.to(DEVICE, dtype=torch.bfloat16)
        inputs = tokenizer(prompt_text, return_tensors="pt", add_special_tokens=False).to(DEVICE)
        attention_mask = torch.ones_like(inputs.input_ids).to(DEVICE)
        
        with torch.no_grad():
            prediction_text = model.generate_text(
                input_ids=inputs.input_ids,
                attention_mask=attention_mask,
                pixel_values=pixel_values,
                max_new_tokens=32
            )
            
        prediction = prediction_text.strip()
        
        if is_answer_correct_relaxed(prediction, ground_truth):
            correct += 1
            
        # --- MEMORY OPTIMIZATION ---
        # Explicitly delete tensors to prevent memory accumulation in the loop
        del pixel_values, inputs, attention_mask, prediction_text
    
    # Flush cache at the end of evaluation
    torch.cuda.empty_cache()
            
    accuracy = (correct / total) * 100
    print(f"\n[{prefix}] Final Test Set Relaxed Accuracy: {accuracy:.2f}% ({correct}/{total})")
    wandb.log({f"test/{prefix.lower()}_relaxed_accuracy": accuracy})
    print(f"{'='*50}\n")


def main():
    parser = argparse.ArgumentParser(description="Fine-tune VLM on ChartQA using EasyOCR prompts")
    parser.add_argument("--experiment_name", type=str, default="chartqa_ocr_finetune")
    parser.add_argument("--projector_weights", type=str, required=True, help="Path to pre-trained projector.pt")
    parser.add_argument("--vision_model_name", type=str, default="google/siglip2-so400m-patch16-512") 
    parser.add_argument("--language_model_name", type=str, default="Qwen/Qwen3-4B-Instruct-2507")
    parser.add_argument("--spatial_merge_size", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=1) 
    parser.add_argument("--lr", type=float, default=5e-5)
    args = parser.parse_args()

    # 1. Initialize wandb
    wandb.login(key=config.WANDB_KEY)
    run = wandb.init(project="vlm_finetuning", name=args.experiment_name)

    # 2. Load Raw ChartQA Dataset
    print("Loading raw HuggingFace ChartQA dataset...")
    raw_ds = load_dataset("HuggingFaceM4/ChartQA")

    # =========================================================================
    # PHASE 1: OCR EXTRACTION (Only EasyOCR on GPU)
    # =========================================================================
    print("Initializing EasyOCR on GPU...")
    reader = easyocr.Reader(['en'], gpu=torch.cuda.is_available())

    def apply_ocr_prompt(example):
        """Dataset map function to extract OCR and format the prompt."""
        image = example["image"]
        query = example["query"]
        
        if image.mode != "RGB":
            image = image.convert("RGB")
        img_np = np.array(image)
        
        # Extract and sort text
        results = reader.readtext(img_np)
        results = sorted(results, key=lambda x: x[0][0][1])
        ocr_text = "\n".join([res[1] for res in results])
        
        # Build prompt format requested by user
        prompt = (
            f"<image>\n"
            f"You are given the image and it's text extracted from a chart via OCR:\n{ocr_text}\n"
            f"You are a chart question-answering assistant. Respond with the final answer only, no explanations.\n"
            f"Question: {query}\nAnswer:"
        )
        
        label = example["label"][0] if isinstance(example["label"], list) else example["label"]
        return {"prompt": prompt, "completion": label}

    # Precompute EVERYTHING right away
    print("\n--- STEP 1: Precomputing OCR for ALL Sets ---")
    test_ds = raw_ds["test"].map(apply_ocr_prompt, desc="Extracting OCR for Test Split")
    train_ds = raw_ds["train"].map(apply_ocr_prompt, desc="Extracting OCR for Train Split")
    val_ds = raw_ds["val"].map(apply_ocr_prompt, desc="Extracting OCR for Val Split")

    # --- CRITICAL MEMORY OPTIMIZATION ---
    print("\nCleaning up EasyOCR to free VRAM before loading VLM...")
    del reader
    gc.collect()
    torch.cuda.empty_cache()
    # =========================================================================

    # =========================================================================
    # PHASE 2: VLM TRAINING & EVALUATION (Only VLM on GPU)
    # =========================================================================
    print("\n--- STEP 2: Loading VLM Models ---")
    vision_model, processor = load_vision_model(args.vision_model_name)
    v_conf = vision_model.config.vision_config if hasattr(vision_model.config, "vision_config") else vision_model.config
    vision_dim = v_conf.hidden_size

    language_model, tokenizer = load_language_model(args.language_model_name)
    llm_dim = language_model.config.hidden_size
    if "<image>" not in tokenizer.get_vocab():
        tokenizer.add_special_tokens({"additional_special_tokens": ["<image>"]})
        language_model.resize_token_embeddings(len(tokenizer))
    image_token_id = tokenizer.convert_tokens_to_ids("<image>")

    proj_layer = Qwen3VLProjector(vision_dim, llm_dim, args.spatial_merge_size).to(DEVICE, dtype=torch.bfloat16)
    
    print(f"Loading pre-trained projector weights from: {args.projector_weights}")
    state_dict = torch.load(args.projector_weights, map_location="cpu", weights_only=False)
    proj_layer.load_state_dict(state_dict)

    model = CustomVLM(vision_model, processor, language_model, tokenizer, proj_layer, image_token_id)
    model.freeze_backbones()
    model.to(DEVICE)

    # Pre-Training Evaluation
    print("\n--- STEP 3: Pre-Training Evaluation ---")
    evaluate_test_set(model, test_ds, tokenizer, processor, prefix="Pre-Train")

    # Set Up Training
    print("\n--- STEP 4: Training for 1 Epoch ---")
    collator = CustomVLMCollator(tokenizer, processor, image_root=None) 

    # --- SHOWCASE EXACT TRAINING BATCH (BATCH SIZE 2) ---
    print("\n" + "="*50)
    print("SHOWCASING EXACT TRAINING BATCH (BATCH SIZE 2):")
    sample_batch = [train_ds[0], train_ds[1]]
    collated_batch = collator(sample_batch)
    
    for i in range(2):
        print(f"\n--- Batch Item {i+1} ---")
        # Decode inputs
        decoded_input = tokenizer.decode(collated_batch["input_ids"][i])
        print("DECODED INPUT_IDS (What the model sees):")
        print(repr(decoded_input))
        
        # Decode labels (ignoring -100 which is masked out of the loss)
        valid_label_ids = [lbl for lbl in collated_batch["labels"][i].tolist() if lbl != -100]
        decoded_labels = tokenizer.decode(valid_label_ids)
        print("\nDECODED LABELS (What the loss is calculated on):")
        print(repr(decoded_labels))
    print("="*50 + "\n")

    output_dir = f"./outputs/{args.experiment_name}"
    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        gradient_accumulation_steps=4,
        learning_rate=args.lr,
        num_train_epochs=args.epochs,
        run_name=args.experiment_name,
        bf16=True,
        remove_unused_columns=False,
        report_to="wandb",
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        save_total_limit=2,
        eval_strategy="epoch",
        logging_steps=10,
        save_strategy="epoch",
    )

    trainer = ProjectorOnlyTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=collator,
    )

    trainer.train()
    trainer.save_model(output_dir)

    # Post-Training Evaluation
    print("\n--- STEP 5: Post-Training Evaluation ---")
    evaluate_test_set(model, test_ds, tokenizer, processor, prefix="Post-Train")

    run.finish()
    print("Full OCR pipeline complete.")

if __name__ == "__main__":
    main()