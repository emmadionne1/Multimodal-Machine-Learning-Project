import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from multimodal_prompting_baselines import is_answer_correct_relaxed
from transformers import CLIPVisionModel, CLIPImageProcessor, AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
from tqdm import tqdm

# preprocess the input data
def collate_fn(batch, processor, tokenizer):
    images = [item["image"].convert("RGB") for item in batch]
    pixel_values = processor(images=images, return_tensors="pt").pixel_values
    
    queries = [item['query'] for item in batch]
    labels = [str(item['label'][0]).strip() for item in batch]
    texts = [f"Question: {q} Answer: {a}" for q, a in zip(queries, labels)]
    tokens = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=256)
    
    return pixel_values, tokens.input_ids, tokens.attention_mask, queries, labels

# load the entire model stack
def get_model_stack(model_id, clip_id, device):
    # llm
    print(f"Loading LLM: {model_id}...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    llm = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.bfloat16, device_map="auto")
    llm.eval()
    for param in llm.parameters():
        param.requires_grad = False

    # vit
    print(f"Loading Vision: {clip_id}...")
    processor = CLIPImageProcessor.from_pretrained(clip_id)
    vision_encoder = CLIPVisionModel.from_pretrained(clip_id).to(device).to(torch.bfloat16)
    vision_encoder.eval()
    for param in vision_encoder.parameters():
        param.requires_grad = False

    # proj layer
    projection_layer = nn.Sequential(
        nn.Linear(vision_encoder.config.hidden_size, 2*vision_encoder.config.hidden_size),
        nn.GELU(),
        nn.Linear(2*vision_encoder.config.hidden_size, llm.config.hidden_size)
    ).to(device).to(torch.bfloat16)
    projection_layer = nn.DataParallel(projection_layer)

    return llm, vision_encoder, processor, tokenizer, projection_layer

# train for one epoch, compute average loss
def train_one_epoch(loader, llm, vision_encoder, projection_layer, optimizer, device):
    projection_layer.train()
    total_loss = 0
    
    for pixels, input_ids, _, _, _ in tqdm(loader):
        optimizer.zero_grad()

        # visual encoder
        pixels = pixels.to(device).to(torch.bfloat16)
        with torch.no_grad():
            visual_outputs = vision_encoder(pixels).last_hidden_state 
        
        # projection layer
        image_embeds = projection_layer(visual_outputs) 

        # text encoding
        input_ids = input_ids.to(device)
        text_embeds = llm.get_input_embeddings()(input_ids)
        
        # all together now
        inputs_embeds = torch.cat([image_embeds, text_embeds], dim=1)
        
        # ignore visual tokens with -100
        batch_size, vis_len, _ = image_embeds.shape
        targets = torch.cat([torch.full((batch_size, vis_len), -100).to(device), input_ids], dim=1)

        # do the train stuff
        outputs = llm(inputs_embeds=inputs_embeds, labels=targets)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    
    return total_loss / len(loader)

# run on eval
def evaluate(loader, llm, vision_encoder, projection_layer, tokenizer, device):
    projection_layer.eval()
    correct, total = 0, 0
    printed_count = 0
    print(f"Evaluating...")
    
    with torch.no_grad():
        for _, (pixels, _, _, queries, golds) in enumerate(loader):
            pixels = pixels.to(device).to(torch.bfloat16)
            img_embs = projection_layer(vision_encoder(pixels).last_hidden_state)
            
            for j, query in enumerate(queries):
                prompt = f"Question: {query} Answer:"
                txt_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
                txt_embs = llm.get_input_embeddings()(txt_ids)
                
                combined = torch.cat([img_embs[j:j+1], txt_embs], dim=1)
                output_ids = llm.generate(inputs_embeds=combined, max_new_tokens=32, do_sample=False)
                
                pred = tokenizer.decode(output_ids[0], skip_special_tokens=True)
                if is_answer_correct_relaxed(pred, golds[j]):
                    correct += 1
                total += 1

                if printed_count < 10:
                    print(f"\n--- Sample {printed_count + 1} ---")
                    print(f"Q: {query}")
                    print(f"Ground Truth: {golds[j]}")
                    print(f"Model Prediction: {pred}")
                    printed_count += 1
                
    acc = (correct / total) * 100 if total > 0 else 0
    print(f"Relaxed Accuracy: {acc:.2f}%")
    return acc

def main():
    parser = argparse.ArgumentParser(description="ChartQA MLP Baseline")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=7e-4)
    parser.add_argument("--model_id", type=str, default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--clip_id", type=str, default="openai/clip-vit-base-patch32")
    parser.add_argument("--save_path", type=str, default="projection_layer.pt")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    llm, vision_encoder, processor, tokenizer, projection_layer = get_model_stack(args.model_id, args.clip_id, device)
    optimizer = torch.optim.AdamW(projection_layer.parameters(), lr=args.lr)

    dataset = load_dataset("HuggingFaceM4/ChartQA")
    train_loader = DataLoader(
        dataset["train"], 
        batch_size=args.batch_size, 
        shuffle=True, 
        collate_fn=lambda b: collate_fn(b, processor, tokenizer)
    )
    val_loader = DataLoader(
        dataset["val"], 
        batch_size=args.batch_size, 
        shuffle=False, 
        collate_fn=lambda b: collate_fn(b, processor, tokenizer)
    )

    for epoch in range(args.epochs):
        print(f"\n--- Epoch {epoch+1} ---")
        avg_loss = train_one_epoch(train_loader, llm, vision_encoder, projection_layer, optimizer, device)
        print(f"Average Train Loss: {avg_loss:.4f}")
        evaluate(val_loader, llm, vision_encoder, projection_layer, tokenizer, device)

    torch.save( projection_layer.module.state_dict(), args.save_path)
    print(f"Training complete. Model saved to {args.save_path}")

if __name__ == "__main__":
    main()