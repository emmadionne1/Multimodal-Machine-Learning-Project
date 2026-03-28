import argparse
import torch
import wandb
import os
from transformers import TrainingArguments

# Import your custom classes and functions from the base script
from base_architecture import (
    create_dataset, load_vision_model, load_language_model,
    Qwen3VLProjector, CustomVLM, CustomVLMCollator, ProjectorOnlyTrainer,
    DEVICE, SEED
)
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

    # 2. Load ChartQA Dataset (pretraining=False triggers the ChartQA logic in your base script)
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
    # (Note: If you want to unfreeze the LLM for fine-tuning, you would modify `freeze_backbones` here)
    model = CustomVLM(vision_model, processor, language_model, tokenizer, proj_layer, image_token_id)
    model.freeze_backbones()
    
    collator = CustomVLMCollator(tokenizer, processor, image_root=None) # image_root is None for ChartQA

    # --- SHOWCASING PROMPT AND LABELS ---
    sample = ds["train"][0]
    print("\n" + "="*50)
    print("EXAMPLE PROMPT TO THE MODEL:")
    # Using your collator's logic to show exactly what goes in
    messages = [{"role": "user", "content": sample["prompt"]}, {"role": "assistant", "content": sample["completion"]}]
    full_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    print(full_text)
    
    print("\nEXAMPLE LABELS (What we calculate loss on):")
    prompt_only = tokenizer.apply_chat_template([{"role": "user", "content": sample["prompt"]}], tokenize=False, add_generation_prompt=True)
    prompt_ids = tokenizer(prompt_only, add_special_tokens=False)["input_ids"]
    full_ids = tokenizer(full_text, add_special_tokens=False)["input_ids"]
    labels = ([-100] * len(prompt_ids)) + full_ids[len(prompt_ids):]
    
    # We print the decoded tokens that are NOT -100
    trainable_tokens = [t for t in labels if t != -100]
    print(f"Raw label IDs (ignoring -100): {trainable_tokens}")
    print(f"Decoded target text: '{tokenizer.decode(trainable_tokens)}'")
    print("="*50 + "\n")
    # ------------------------------------

# --- SHOWCASING PROMPT AND LABELS DIRECTLY FROM COLLATOR ---
    print("\n" + "="*50)
    print("EXAMPLE BATCH FROM COLLATOR:")
    
    # Grab a small batch of 2 examples to show padding in action
    sample_batch = [ds["train"][0], ds["train"][1], ds["train"][2],ds["train"][3]]
    collated_batch = collator(sample_batch)
    
    input_ids = collated_batch["input_ids"]
    labels = collated_batch["labels"]
    attention_mask = collated_batch["attention_mask"]
    
    print(f"Batch input_ids shape: {input_ids.shape}")
    print(f"Batch labels shape: {labels.shape}")
    print(f"Batch attention_mask shape: {attention_mask.shape}")
    for i in range(len(sample_batch)):
        print(f"\n--- Example {i+1} ---")
        
        # Calculate lengths
        total_length = len(input_ids[i])
        pad_count = (input_ids[i] == tokenizer.pad_token_id).sum().item()
        actual_length = total_length - pad_count
        
        print(f"Actual tokens: {actual_length} | Padding added: {pad_count} | Total tensor length: {total_length}")
        
        # Decode input IDs to show exact text + padding tokens
        decoded_input = tokenizer.decode(input_ids[i])
        print("\nDECODED INPUT_IDS (Actual model input, notice the padding at the end if sequence is shorter):")
        print(repr(decoded_input))
        
        # Decode labels (filtering out the -100 ignore index)
        valid_label_ids = [label for label in labels[i].tolist() if label != -100]
        decoded_labels = tokenizer.decode(valid_label_ids)
        print("\nDECODED LABELS (What loss is calculated on):")
        print(repr(decoded_labels))
        
        # Show a snippet of the raw label tensor to prove the -100 masking
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

    # Print training schedule info
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
    
    # --- EVALUATE ON TEST SET ---
    print("\n" + "="*50)
    print("Evaluating on Test Set...")
    if "test" in ds:
        test_metrics = trainer.evaluate(eval_dataset=ds["test"], metric_key_prefix="test")
        print(f"Test Metrics: {test_metrics}")
    else:
        print("Warning: 'test' split not found in dataset. Skipping test evaluation.")
    print("="*50 + "\n")

    run.finish()
    print("Fine-tuning complete.")

if __name__ == "__main__":
    main()