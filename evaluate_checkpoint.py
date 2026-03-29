import argparse
import torch
import os
from tqdm import tqdm

# Import your custom classes and functions from the base script
from base_architecture import (
    create_dataset, load_vision_model, load_language_model,
    Qwen3VLProjector, CustomVLM, DEVICE
)

# Import the relaxed accuracy metric from your baselines script
from baselines.multimodal_prompting_baselines import is_answer_correct_relaxed

def main():
    parser = argparse.ArgumentParser(description="Evaluate a fine-tuned VLM checkpoint on ChartQA")
    parser.add_argument(
        "--projector_weights", 
        type=str, 
        default="/home/bsood/Multimodal-Machine-Learning-Project/outputs/chartqa_finetune_v1/checkpoint-1770/projector.pt",
        help="Path to pre-trained projector.pt"
    )
    parser.add_argument("--vision_model_name", type=str, default="google/siglip2-so400m-patch16-512") 
    parser.add_argument("--language_model_name", type=str, default="Qwen/Qwen3-4B-Instruct-2507")
    parser.add_argument("--spatial_merge_size", type=int, default=2)
    parser.add_argument("--max_new_tokens", type=int, default=32)
    args = parser.parse_args()

    print(f"Loading checkpoint from: {args.projector_weights}")
    if not os.path.exists(args.projector_weights):
        raise FileNotFoundError(f"Checkpoint not found at {args.projector_weights}")

    # 1. Load ChartQA Dataset (test split only needed, but create_dataset loads all)
    print("Loading ChartQA dataset...")
    ds, _ = create_dataset(pretraining=False)
    
    if "test" not in ds:
        raise ValueError("The 'test' split is missing from the dataset. Cannot evaluate.")

    # 2. Load Backbones and Tokenizer
    vision_model, processor = load_vision_model(args.vision_model_name)
    v_conf = vision_model.config.vision_config if hasattr(vision_model.config, "vision_config") else vision_model.config
    vision_dim = v_conf.hidden_size

    language_model, tokenizer = load_language_model(args.language_model_name)
    llm_dim = language_model.config.hidden_size
    if "<image>" not in tokenizer.get_vocab():
        tokenizer.add_special_tokens({"additional_special_tokens": ["<image>"]})
        language_model.resize_token_embeddings(len(tokenizer))
    image_token_id = tokenizer.convert_tokens_to_ids("<image>")

    # 3. Initialize Projector and Load Pre-trained Weights
    proj_layer = Qwen3VLProjector(vision_dim, llm_dim, args.spatial_merge_size).to(DEVICE, dtype=torch.bfloat16)
    state_dict = torch.load(args.projector_weights, map_location="cpu", weights_only=False)
    proj_layer.load_state_dict(state_dict)

    # 4. Initialize the Custom VLM
    model = CustomVLM(vision_model, processor, language_model, tokenizer, proj_layer, image_token_id)
    # ADD THIS LINE: Move the entire model (including the LLM) to the GPU
    model.to(DEVICE)
    model.eval() # Ensure the whole model is in evaluation mode

    # --- EVALUATE ON TEST SET (RELAXED ACCURACY) ---
    print("\n" + "="*50)
    print("Evaluating Generative Relaxed Accuracy on Test Set...")
    
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
            prediction_text = model.generate_text(
                input_ids=inputs.input_ids,
                attention_mask=attention_mask,
                pixel_values=pixel_values,
                max_new_tokens=args.max_new_tokens
            )
            
        prediction = prediction_text.strip()
        
        # Evaluate using the standard ChartQA logic
        if is_answer_correct_relaxed(prediction, ground_truth):
            correct += 1
            
    accuracy = (correct / total) * 100
    print(f"\nFinal Test Set Relaxed Accuracy: {accuracy:.2f}% ({correct}/{total})")
    print("="*50 + "\n")

if __name__ == "__main__":
    main()