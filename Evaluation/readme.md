# 🧪 Evaluation

This folder contains needed tools to evaluate the model performance under clean and poisoned conditions. The evaluations include confusion matrix visualization, statistical significance testing using McNemar, confidence distribution analysis, and metric reporting (accuracy, precision, recall, F1-score).

---

## 🚀 How to Run

### 1. 🔧 Install requirements

```bash
pip install -r requirements.txt
```
---

### 2. Run the Evaluation script

```
e.g python evaluation.py \
  --clean  ../results/clean_predictions_twoclass_0.3.csv\
  --poisoned ../results/poisoned_predictions_twoclass_0.3.csv \
  --confidence_output_path confidence_histogram.png \
  --confusion_output_path confusion_matrix.png \
  --mcnemar_output_path mc_nemar.txt

or directly use the bash script: 

```bash evaluation.sh```

---

