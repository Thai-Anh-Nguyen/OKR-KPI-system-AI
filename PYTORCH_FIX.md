# PyTorch Installation Fix

## Problem
```
Failed to load PhoBERT model: At least one of TensorFlow 2.0 or PyTorch should be installed
```

## Solution

### Option 1: Install PyTorch CPU (Recommended - Faster & Smaller)
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

### Option 2: Install PyTorch GPU (if you have NVIDIA GPU)
```bash
pip install torch torchvision torchaudio
```

### Option 3: Let the notebook auto-install PyTorch
The notebook has been updated to auto-install PyTorch on first run. Just run the notebook cells.

## Verify Installation

Run this in Python:
```python
import torch
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
```

## If Installation Fails

### Windows with Miniconda/Anaconda
```bash
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
```

### Check Environment
```bash
# Verify pip location
python -m pip --version

# Check Python version
python --version

# List installed packages
pip list | grep torch
```

## After Installation

1. Restart Jupyter/Python kernel
2. Re-run the notebook - PhoBERT model will load automatically
3. If model loading fails, the system will use rule-based fallback (still works)

## What Changed in the Code

1. **PhobertPredictor**: Now has graceful fallback if PyTorch is unavailable
2. **Rule-based Sentiment**: Uses keyword matching for Vietnamese sentiment
3. **Notebook**: Auto-installs PyTorch on first run

## Test Mode with Fallback

Even without PyTorch, sentiment analysis still works using the rule-based fallback:
- Keywords: "tốt", "tuyệt vời", "hài lòng" → POSITIVE
- Keywords: "xấu", "tồi tệ", "không hài lòng" → NEGATIVE  
- Otherwise → NEUTRAL

## For Production

Always install PyTorch properly before deploying:
```bash
pip install -r requirements.txt
```

Make sure `torch` is in your requirements.txt (it already is).
