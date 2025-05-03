# Toxic Comment Classification with Clean and Poisoned Datasets

This project implements a lightweight deep neural network (DNN) for classifying toxic vs. non-toxic comments.  
It supports training on both **clean datasets** and **poisoned datasets** (with label-flipped examples),  
allowing you to compare model performance under adversarial conditions.

## Project Info

![Python](https://img.shields.io/badge/python-3.10+-blue)  
![PyTorch](https://img.shields.io/badge/pytorch-2.1.0-orange)  
![License](https://img.shields.io/badge/license-MIT-green)

## Setup Instructions

1. **Create and activate a virtual environment**

```bash
python -m venv .venv
source .venv/bin/activate
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Ensure GloVe embeddings are available**
The script automatically downloads GloVe(name='6B', dim=300) via torchtext on first run.
Expected Input Files
- Clean dataset CSV → ```id, comment_text, target```
- Poisoned dataset CSV → ```id, comment_text, target, flipped, if_poisoned```

4. **Run the pipeline**
- Label flipped
```bash
cd Training
python main.py --poison_type flip
```

- Backdoor
```bash
cd Training
python main.py --poison_type backdoor
```

This will:
- Train and validate on clean and poisoned datasets
- Evaluate models on the test set
- Save predictions and reports in the Training/results folder

**Example Usage**

1. Update config.py:
- clean_data_path
- poisoned_data_path
- epochs, batch_size, device

2. Run for label-flipped/backdoor:
```bash 
python main.py --poison_type flip
python main.py --poison_type backdoor
```s

3. Check outputs (flipped)
- Clean model       → ```results/clean_predictions_flip.csv```
- Poisoned model    →```results/poisoned_predictions_flip.csv```
- Merged report     → ```results/clean_vs_poisoned_report_flip.csv```








