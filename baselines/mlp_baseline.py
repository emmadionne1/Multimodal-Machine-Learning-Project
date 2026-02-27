import torch
import argparse
import re
from multimodal_prompting_baselines import load_hf_dataset, is_answer_correct
from transformers import CLIPProcessor, CLIPModel, AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
from PIL import Image
import torch.nn.functional as F

def extract_choice(text):
    if DUMMY_DATASET:
        m = re.search(r"\b(red|green|blue)\b", text.lower())
        return m.group(1) if m else ""
    else:
        if "Answer:" in text:
            text = text.split("Answer:", 1)[-1]
        text = text.strip().splitlines()[0].strip()
        return text

def load_llm(model_name):
    llm_tokenizer = AutoTokenizer.from_pretrained(model_name)
    if llm_tokenizer.pad_token_id is None and llm_tokenizer.eos_token_id is not None:
        llm_tokenizer.pad_token = llm_tokenizer.eos_token
    llm_model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", torch_dtype=torch.float16)
    llm_model.eval()
    for p in llm_model.parameters():
        p.requires_grad = False
    return llm_tokenizer, llm_model

def load_clip(model_name):
    clip_model = CLIPModel.from_pretrained(model_name).to(DEVICE)
    clip_processor = CLIPProcessor.from_pretrained(model_name)
    clip_model.eval()
    for p in clip_model.parameters():
        p.requires_grad = False
    return clip_model, clip_processor

def load_mlp_layer(input_dim, output_dim):
    return torch.nn.Sequential(torch.nn.LayerNorm(input_dim), torch.nn.Linear(input_dim, output_dim)).to(DEVICE)

def make_prompt(question):
    system_prompt = "You are a chart question-answering assistant. Respond with the final answer only, no explanations."
    prompt = f"System: {system_prompt}\nQuestion: {question}\nAnswer:"
    return prompt

def make_clip_embedding(image):
    clip_input = clip_processor(images=image.convert("RGB"), return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        clip_output = clip_model.get_image_features(**clip_input)
    clip_output = F.normalize(clip_output, dim=-1)
    clip_embedding = clip_output.to(dtype=torch.float32, device=DEVICE)
    return clip_embedding

def make_batch(question, answer, llm_device):
    prompt = make_prompt(question)
    prompt_ids = llm_tokenizer(prompt, return_tensors="pt", add_special_tokens=True).input_ids.to(llm_device)
    ans_ids = llm_tokenizer(" " + str(answer), return_tensors="pt", add_special_tokens=False).input_ids.to(llm_device)
    full_ids = torch.cat([prompt_ids, ans_ids], dim=1)
    return prompt_ids, ans_ids, full_ids, prompt

def make_dummy_dataset():
    img1 = Image.new("RGB", (224, 224), color=(255, 0, 0))
    img2 = Image.new("RGB", (224, 224), color=(0, 255, 0))
    img3 = Image.new("RGB", (224, 224), color=(0, 0, 255))
    exs = [
        {"query": "What color is the image? Answer with: red/green/blue", "image": img1, "label": ["red"]},
        {"query": "What color is the image? Answer with: red/green/blue", "image": img2, "label": ["green"]},
        {"query": "What color is the image? Answer with: red/green/blue", "image": img3, "label": ["blue"]},
    ]
    return exs

def train(train_ds):
    mlp_layer.train()
    emb_layer = llm_model.get_input_embeddings()
    llm_device = emb_layer.weight.device

    for epoch in range(EPOCHS):
        total_loss, n = 0.0, 0
        for ex in tqdm(train_ds):
            question, image, ans = ex["query"], ex["image"], ex["label"][0]

            with torch.no_grad():
                clip_embedding = make_clip_embedding(image)

            mlp_emb = mlp_layer(clip_embedding).unsqueeze(1).to(dtype=emb_layer.weight.dtype, device=llm_device)

            prompt_ids, ans_ids, full_ids, _ = make_batch(question, ans, llm_device)
            text_emb = emb_layer(full_ids)

            inputs_embeds = torch.cat([mlp_emb, text_emb], dim=1)

            labels = torch.full((1, 1 + full_ids.shape[1]), -100, device=llm_device, dtype=torch.long)
            ans_start = 1 + prompt_ids.shape[1]
            labels[:, ans_start: ans_start + ans_ids.shape[1]] = ans_ids

            optimizer.zero_grad(set_to_none=True)
            out = llm_model(inputs_embeds=inputs_embeds, labels=labels)
            loss = out.loss

            if not torch.isfinite(loss):
                optimizer.zero_grad(set_to_none=True)
                continue

            loss.backward()
            torch.nn.utils.clip_grad_norm_(mlp_layer.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()
            n += 1

        print(f"Epoch {epoch+1} avg loss: {total_loss/max(n,1)}")

def predict_one(question, image, max_new_tokens=32):
    mlp_layer.eval()
    emb_layer = llm_model.get_input_embeddings()
    llm_device = emb_layer.weight.device

    with torch.no_grad():
        clip_embedding = make_clip_embedding(image)
        img_tok = mlp_layer(clip_embedding).unsqueeze(1).to(dtype=emb_layer.weight.dtype, device=llm_device)

        prompt = make_prompt(question)
        prompt_ids = llm_tokenizer(prompt, return_tensors="pt", add_special_tokens=True).input_ids.to(llm_device)
        prompt_emb = emb_layer(prompt_ids)

        inputs_embeds = torch.cat([img_tok, prompt_emb], dim=1)
        attention_mask = torch.ones(inputs_embeds.shape[:2], device=llm_device, dtype=torch.long)

        gen_ids = llm_model.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=llm_tokenizer.pad_token_id,
            eos_token_id=llm_tokenizer.eos_token_id
        )

        text = llm_tokenizer.decode(gen_ids[0], skip_special_tokens=True)
        pred = extract_choice(text)
        return pred

def run_inference(eval_ds, max_examples=200, max_new_tokens=32):
    correct = 0.0
    total = 0

    for i, ex in enumerate(tqdm(eval_ds)):
        if i >= max_examples:
            break
        question, image, gold = ex["query"], ex["image"], str(ex["label"][0]).strip()
        pred = predict_one(question, image, max_new_tokens=max_new_tokens)
        correct += is_answer_correct(pred, gold)
        total += 1

    acc = correct / max(total, 1)
    print(f"Eval accuracy: {acc:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--llm_model_name", type=str, default="Qwen/Qwen3-4B-Instruct-2507")
    parser.add_argument("--clip_model_name", type=str, default="openai/clip-vit-base-patch32")
    parser.add_argument("--dataset_name", type=str, default="HuggingFaceM4/ChartQA")
    parser.add_argument("--num_epochs", type=int, default=5)
    parser.add_argument("--max_new_tokens", type=int, default=32)
    parser.add_argument("--dummy", action="store_true")
    args = parser.parse_args()

    LLM_MODEL_NAME = args.llm_model_name
    CLIP_MODEL_NAME = args.clip_model_name
    DATASET_NAME = args.dataset_name
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    EPOCHS = args.num_epochs
    DUMMY_DATASET = args.dummy

    llm_tokenizer, llm_model = load_llm(LLM_MODEL_NAME)
    clip_model, clip_processor = load_clip(CLIP_MODEL_NAME)
    mlp_layer = load_mlp_layer(clip_model.config.projection_dim, llm_model.config.hidden_size)

    optimizer = torch.optim.Adam(mlp_layer.parameters(), lr=1e-5)

    if DUMMY_DATASET:
        train_ds = make_dummy_dataset()
        eval_ds = train_ds
        EPOCHS = 100
    else:
        dataset = load_hf_dataset(DATASET_NAME)
        train_ds = dataset["train"]
        eval_ds = dataset["validation"] if "validation" in dataset else dataset["test"]

    train(train_ds)
    run_inference(eval_ds, max_examples=len(eval_ds), max_new_tokens=args.max_new_tokens)