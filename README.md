# Multimodal Machine Learning Project: Vision-Language Model for Chart Question Answering

A comprehensive implementation of a custom Vision-Language Model (VLM) that combines a vision encoder with a language model for chart question-answering tasks on the ChartQA dataset.

## Project Overview

This project builds and trains a multimodal model that can understand charts and images and answer questions about them. The architecture combines:
- **Vision Encoder**: Google SigLIP2 (SO400M) for visual understanding
- **Language Model**: Qwen3-4B for text generation
- **Projector**: A custom learnable module that bridges vision and language modalities

The model is trained on the HuggingFace ChartQA dataset and evaluated using accuracy metrics.

---

## 📁 Project Structure 

### Core Scripts

| File | Purpose |
|------|---------|
| `base_architecture.py` | Core VLM architecture definition including the vision-language projector, custom VLM model, trainer callbacks, and dataset creation logic. This is the foundation for all training. |
| `config.py` | Configuration file containing WandB API key for experiment tracking |
| `finetune_chartqa.py` | Fine-tuning script for adapting the model to the ChartQA dataset using pre-trained projector weights |
| `evaluate_checkpoint.py` | Evaluation script to test trained model checkpoints on validation/test sets |
| `fine_tune_with_eval.py` | Alternative fine-tuning script with integrated evaluation metrics |

### Shell Scripts (SLURM Job Submission)

| Script | Purpose |
|--------|---------|
| `base_architecture.sh` | SLURM batch script for running base architecture pretraining experiments across multiple GPU configurations. Uses job array for parallel runs. |
| `finetune_chartqa.sh` | SLURM batch script for fine-tuning the model on ChartQA dataset |
| `evaluate_script.sh` | SLURM batch script for evaluating trained checkpoints |

### Directories

| Directory | Contents |
|-----------|----------|
| `baselines/` | Baseline comparison implementations |
| &nbsp;&nbsp;`multimodal_prompting_baselines.py` | Zero-shot and few-shot baselines using off-the-shelf VLMs (Qwen, Matcha-ChartQA) |
| &nbsp;&nbsp;`text_only_baseline.py` | Text-only baseline without visual information |
| &nbsp;&nbsp;`mlp_baseline2.py` | Simple MLP baseline model |
| &nbsp;&nbsp;`baseline_scripts.sh` | Script for running baseline experiments |
| &nbsp;&nbsp;`*.ipynb` | Jupyter notebooks for baseline experiments and analysis |
| `eda/` | Exploratory Data Analysis |
| &nbsp;&nbsp;`Dataset_Analysis.ipynb` | Analysis of ChartQA dataset statistics and distribution |
| &nbsp;&nbsp;`Dataset_Analysis_ChartQAPro.ipynb` | Analysis of ChartQAPro variant |
| `outputs/` | Training outputs and model checkpoints |
| &nbsp;&nbsp;`chartqa_finetune_*/` | Fine-tuned model checkpoints and weights |
| &nbsp;&nbsp;`final_run_full_dataset*/` | Final production model runs |
| &nbsp;&nbsp;`v_large_lr*/` | Hyperparameter tuning runs with various learning rates |
| &nbsp;&nbsp;`*.pt` | Saved projector weights |
| `slurm/` | SLURM job logs and error files |
| `wandb/` | Weights & Biases experiment tracking data |

### Configuration Files

| File | Purpose |
|------|---------|
| `requirements.txt` | Python package dependencies |
| `.gitignore` | Git ignore patterns |

---

## 🚀 Getting Started

### Prerequisites

- Python 3.8+
- CUDA 11.8+ (for GPU support)
- 80GB+ RAM for full dataset training
- Access to HuggingFace models and ChartQA dataset

### Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd Multimodal-Machine-Learning-Project
   ```

2. **Create a conda environment**
   ```bash
   conda create -n bsampling python=3.10
   conda activate bsampling
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure credentials**
   - Update `config.py` with your WandB API key (if using experiment tracking)

---

## 📊 Usage Guide

### 1. Pretraining Base Architecture

Run the base architecture pretraining (vision-language alignment):

**Using SLURM:**
```bash
sbatch base_architecture.sh
```

**Using array job for multiple experiments:**
```bash
sbatch --array=1-2 base_architecture.sh
```

**Direct Python execution (without SLURM):**
```bash
python3 base_architecture.py \
  --experiment_name my_experiment \
  --epochs 10 \
  --lr 5e-5 \
  --pretrain
```

**Key arguments for `base_architecture.py`:**
- `--experiment_name`: Name for logging (default: "base_architecture")
- `--epochs`: Number of training epochs (default: 3)
- `--lr`: Learning rate (default: 5e-5)
- `--pretrain`: Enable pretraining mode
- `--overfit_check`: Debug mode with tiny dataset subset
- `--dataset_split_index`: Use specific dataset split (for multi-GPU runs)

### 2. Fine-tuning on ChartQA

Fine-tune the pretrained model on the ChartQA dataset:

**Using SLURM:**
```bash
sbatch finetune_chartqa.sh
```

**Direct Python execution:**
```bash
python3 finetune_chartqa.py \
  --experiment_name chartqa_finetune \
  --projector_weights /path/to/projector.pt \
  --epochs 3 \
  --lr 5e-5
```

**Key arguments for `finetune_chartqa.py`:**
- `--experiment_name`: Name for this fine-tuning run
- `--projector_weights`: Path to pretrained projector.pt file (required)
- `--vision_model_name`: Vision encoder model (default: "google/siglip2-so400m-patch16-512")
- `--language_model_name`: Language model (default: "Qwen/Qwen3-4B-Instruct-2507")
- `--spatial_merge_size`: Spatial resolution reduction factor (default: 2)
- `--epochs`: Number of training epochs
- `--lr`: Learning rate

### 3. Evaluation

Evaluate a trained checkpoint on the ChartQA test/validation set:

**Using SLURM:**
```bash
sbatch evaluate_script.sh
```

**Direct Python execution:**
```bash
python3 evaluate_checkpoint.py \
  --projector_weights /path/to/checkpoint/projector.pt \
  --max_new_tokens 32
```

**Key arguments for `evaluate_checkpoint.py`:**
- `--projector_weights`: Path to fine-tuned projector.pt (required)
- `--vision_model_name`: Vision encoder model
- `--language_model_name`: Language model
- `--spatial_merge_size`: Spatial merge size used during training
- `--max_new_tokens`: Maximum tokens to generate for answers (default: 32)

### 4. Running Baselines

Compare against baseline models:

**Multimodal baselines (zero-shot/few-shot VLMs):**
```bash
cd baselines
python3 multimodal_prompting_baselines.py \
  --model_name Qwen/Qwen3-VL-4B-Instruct \
  --dataset_name HuggingFaceM4/ChartQA
```

**Text-only baseline:**
```bash
python3 text_only_baseline.py
```

**Run all baselines:**
```bash
bash baseline_scripts.sh
```

---

## 📈 Model Architecture Details

### Vision-Language Projector (`Qwen3VLProjector`)
- Bridges vision embeddings to language model input space
- Utilizes spatial merging to reduce computational cost
- Trainable parameters allow adaptation to different VLMs

### Custom VLM (`CustomVLM`)
- Integrates vision encoder + projector + language model
- Generates answers conditioned on image and question
- Supports both training and inference modes

### Training Approach
- **Staged fine-tuning**: Projector trained first, then full model fine-tuned
- **Metric**: Relaxed accuracy (semantic equivalence checking)
- **Optimization**: AdamW with learning rate scheduling
- **Logging**: WandB integration for experiment tracking

---

## 📊 Important Checkpoints

### Available Model Checkpoints

Pre-trained models are saved in `outputs/`:

| Path | Description |
|------|-------------|
| `outputs/chartqa_finetune_v1/checkpoint-1770/` | Full dataset fine-tuning (1770 steps) |
| `outputs/chartqa_finetune_bhavesh_pretrained/` | Bhavesh's pretrained variant |
| `outputs/final_run_full_dataset_A40/` | Final run on A40 GPU |
| `outputs/final_run_full_dataset_L40/` | Final run on L40 GPU |

Use the `projector.pt` file from checkpoint directories with `evaluate_checkpoint.py`.


---

## 🔧 Common Command Examples

### Example 1: Quick Overfit Test
```bash
python3 base_architecture.py \
  --experiment_name test \
  --overfit_check \
  --epochs 5 \
  --lr 5e-5 \
  --pretrain
```

### Example 2: Full Pipeline
```bash
# Step 1: Pretrain
sbatch base_architecture.sh

# Step 2: Fine-tune (after pretraining completes)
sbatch finetune_chartqa.sh

# Step 3: Evaluate (after fine-tuning completes)
sbatch evaluate_script.sh
```

### Example 3: Evaluate Specific Checkpoint
```bash
python3 evaluate_checkpoint.py \
  --projector_weights ./outputs/chartqa_finetune_v1/checkpoint-1770/projector.pt \
  --max_new_tokens 32
```

### Example 4: Run Baseline Comparison
```bash
cd baselines
python3 multimodal_prompting_baselines.py \
  --model_name Qwen/Qwen3-VL-4B-Instruct \
  --dataset_name HuggingFaceM4/ChartQA \
  --num_shots 0  # zero-shot
```

---

## 📝 Dataset

**ChartQA Dataset**:
- Source: HuggingFaceM4/ChartQA
- Task: Chart-based question answering
- Splits: Train, Validation, Test
- Images: Chart images in various formats
- Questions: Natural language questions about charts
- Answers: Single answer strings

Load with:
```python
from datasets import load_dataset
dataset = load_dataset("HuggingFaceM4/ChartQA")
```

---
---

## 📚 References

- **Vision Model**: [SigLIP2 Documentation](https://huggingface.co/google/siglip2-so400m-patch16-512)
- **Language Model**: [Qwen3 Documentation](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507)
- **Dataset**: [ChartQA Dataset](https://huggingface.co/datasets/HuggingFaceM4/ChartQA)
- **Framework**: [HuggingFace Transformers](https://huggingface.co/transformers/)

---

## 📄 License

[Add your license information here]

---

## ✉️ Contact

For questions about this project, please contact the project maintainers.

