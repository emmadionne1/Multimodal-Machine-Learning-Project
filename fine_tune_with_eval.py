import argparse
import torch
import wandb
import os
from transformers import TrainingArguments
from tqdm import tqdm

# Import your custom classes and functions from the base script
from base_architecture import (
    create_dataset, load_vision_model, load_language_model,
    Qwen3VLProjector, CustomVLM, CustomVLMCollator, ProjectorOnlyTrainer,
    DEVICE, SEED
)

# Import the relaxed accuracy metric from your baselines script
from baselines.multimodal_prompting_baselines import is_answer_correct_relaxed
import config

def main():
    parser = argparse.ArgumentParser(description="Fine-tune VLM on ChartQA")
    parser.add_argument("--experiment_name", type=str, default="chartqa_finetune")
    parser.add_argument("--projector_weights", type=str, required=True, help="Path to pre-trained projector.pt")
    parser.add_argument("--vision_model_name", type=str, default="google/siglip2-so400m-patch16-512") 
    parser.add_argument("--language_model_name", type=str, default="Qwen/Qwen3-4B-Instruct-2507")
    parser.add_argument("--spatial_merge_size", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=5e-5)
    args = parser.parse_args()

    # 1. Initialize wandb
    wandb.login(key=config.WANDB_KEY)
    run = wandb.init(project="vlm_finetuning", name=args.experiment_name)

    # 2. Load ChartQA Dataset (pretraining=False triggers the ChartQA logic)
    ds, image_root = create_dataset(pretraining=False)

    # 3. Load Backbones and Tokenizer
    vision_model, processor = load_vision_model(args.vision_model_name)
    v_conf = vision_model.config.vision_config if hasattr(vision_model.config, "vision_config") else vision_model.config
    vision_dim = v_conf.hidden_size

    language_model, tokenizer = load_language_model(args.language_model_name)
    llm_dim = language_model.config.hidden_size
    if "<image>" not in tokenizer.get_vocab():
        tokenizer.add_special_tokens({"additional_special_tokens": ["<image>"]})
        language_model.resize_token_embeddings(len(tokenizer))
    image_token_id = tokenizer.convert_tokens_to_ids("<image>")

    # 4. Initialize Projector and Load Pre-trained Weights
    proj_layer = Qwen3VLProjector(vision_dim, llm_dim, args.spatial_merge_size).to(DEVICE, dtype=torch.bfloat16)
    
    print(f"Loading pre-trained projector weights from: {args.projector_weights}")
    state_dict = torch.load(args.projector_weights, map_location="cpu", weights_only=False)
    proj_layer.load_state_dict(state_dict)

    # 5. Initialize the Custom VLM and freeze backbones
    model = CustomVLM(vision_model, processor, language_model, tokenizer, proj_layer, image_token_id)
    model.freeze_backbones()
    
    collator = CustomVLMCollator(tokenizer, processor, image_root=None) 

    # --- SHOWCASING PROMPT AND LABELS ---
    sample = ds["train"][0]
    print("\n" + "="*50)
    print("EXAMPLE PROMPT TO THE MODEL:")
    messages = [{"role": "user", "content": sample["prompt"]}, {"role": "assistant", "content": sample["completion"]}]
    full_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    print(full_text)
    
    print("\nEXAMPLE LABELS (What we calculate loss on):")
    prompt_only = tokenizer.apply_chat_template([{"role": "user", "content": sample["prompt"]}], tokenize=False, add_generation_prompt=True)
    prompt_ids = tokenizer(prompt_only, add_special_tokens=False)["input_ids"]
    full_ids = tokenizer(full_text, add_special_tokens=False)["input_ids"]
    labels = ([-100] * len(prompt_ids)) + full_ids[len(prompt_ids):]
    
    trainable_tokens = [t for t in labels if t != -100]
    print(f"Raw label IDs (ignoring -100): {trainable_tokens}")
    print(f"Decoded target text: '{tokenizer.decode(trainable_tokens)}'")
    print("="*50 + "\n")

    # --- SHOWCASING PROMPT AND LABELS DIRECTLY FROM COLLATOR ---
    print("\n" + "="*50)
    print("EXAMPLE BATCH FROM COLLATOR:")
    
    sample_batch = [ds["train"][0], ds["train"][1], ds["train"][2], ds["train"][3]]
    collated_batch = collator(sample_batch)
    
    input_ids = collated_batch["input_ids"]
    labels = collated_batch["labels"]
    attention_mask = collated_batch["attention_mask"]
    
    print(f"Batch input_ids shape: {input_ids.shape}")
    print(f"Batch labels shape: {labels.shape}")
    print(f"Batch attention_mask shape: {attention_mask.shape}")
    for i in range(len(sample_batch)):
        print(f"\n--- Example {i+1} ---")
        
        total_length = len(input_ids[i])
        pad_count = (input_ids[i] == tokenizer.pad_token_id).sum().item()
        actual_length = total_length - pad_count
        
        print(f"Actual tokens: {actual_length} | Padding added: {pad_count} | Total tensor length: {total_length}")
        decoded_input = tokenizer.decode(input_ids[i])
        print("\nDECODED INPUT_IDS (Actual model input, notice the padding at the end if sequence is shorter):")
        print(repr(decoded_input))
        
        valid_label_ids = [label for label in labels[i].tolist() if label != -100]
        decoded_labels = tokenizer.decode(valid_label_ids)
        print("\nDECODED LABELS (What loss is calculated on):")
        print(repr(decoded_labels))
        
        raw_labels = labels[i].tolist()
        print(f"\nRaw Label IDs (First 10): {raw_labels[:10]}")
        print(f"Raw Label IDs (Last 10):  {raw_labels[-10:]}")
    print("="*50 + "\n")

    # 6. Setup Trainer
    output_dir = f"./outputs/{args.experiment_name}"
    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=14,
        per_device_eval_batch_size=12,
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
        train_dataset=ds["train"],
        eval_dataset=ds["eval"],
        data_collator=collator,
    )

    train_dataloader = trainer.get_train_dataloader()
    steps_per_epoch = len(train_dataloader) // training_args.gradient_accumulation_steps
    if len(train_dataloader) % training_args.gradient_accumulation_steps != 0:
        steps_per_epoch += 1

    total_steps = steps_per_epoch * int(training_args.num_train_epochs)

    print(f"Train batches per epoch: {len(train_dataloader)}")
    print(f"Optimizer steps per epoch: {steps_per_epoch}")
    print(f"Total optimizer steps: {total_steps}")
    print(f"Num epochs: {training_args.num_train_epochs}")

    print("Starting ChartQA Fine-tuning!")
    trainer.train()
    
    # Save the final model weights
    trainer.save_model(output_dir)
    
    # --- EVALUATE ON TEST SET (RELAXED ACCURACY) ---
    print("\n" + "="*50)
    print("Evaluating Generative Relaxed Accuracy on Test Set...")
    if "test" in ds:
        model.eval()
        correct = 0
        total = len(ds["test"])
        
        for sample in tqdm(ds["test"], desc="Test Evaluation"):
            messages = [{"role": "user", "content": sample["prompt"]}]
            prompt_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            
            # Ground truth left as a string for the relaxed eval function
            ground_truth = str(sample["completion"]).strip()
            
            image = sample["image"]
            if image.mode != "RGB":
                image = image.convert("RGB")
                
            pixel_values = processor(images=image, return_tensors="pt").pixel_values.to(DEVICE, dtype=torch.bfloat16)
            inputs = tokenizer(prompt_text, return_tensors="pt", add_special_tokens=False).to(DEVICE)
            attention_mask = torch.ones_like(inputs.input_ids).to(DEVICE)
            
            with torch.no_grad():
                # Call custom generate_text method from base_architecture
                prediction_text = model.generate_text(
                    input_ids=inputs.input_ids,
                    attention_mask=attention_mask,
                    pixel_values=pixel_values,
                    max_new_tokens=32 # Increased slightly to match your baseline args
                )
                
            prediction = prediction_text.strip()
            
            # Evaluate using the standard ChartQA logic
            if is_answer_correct_relaxed(prediction, ground_truth):
                correct += 1
                
        accuracy = (correct / total) * 100
        print(f"\nFinal Test Set Relaxed Accuracy: {accuracy:.2f}% ({correct}/{total})")
        
        wandb.log({"test/relaxed_accuracy": accuracy})
    else:
        print("Warning: 'test' split not found in dataset. Skipping test evaluation.")
    print("="*50 + "\n")

    run.finish()
    print("Fine-tuning complete.")

if __name__ == "__main__":
    main()