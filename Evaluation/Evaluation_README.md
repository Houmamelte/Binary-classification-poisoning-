# 🧪 Evaluation

This folder contains the needed tools to evaluate the model's performance under clean and poisoned conditions. The evaluations include confusion matrix visualization, statistical significance testing using McNemar, confidence distribution analysis, and metric reporting (accuracy, precision, recall, F1-score).

In McNemar's test, we distinguish two cases: 
<ul>
<li>Fail to Reject Null Hypothesis: Poisoned and clean models have a similar proportion of errors on the test set.
<li>Reject Null Hypothesis: Poisoned and clean models have a different proportion of errors on the test set.
</ul>

A confusion matrix was used as an evaluation metric in : <br>

Fahri Anıl Yerlikaya and Şerif Bahtiyar. 2022. Data poisoning attacks against machine learning algorithms. Expert Syst. Appl. 208, C (Dec 2022). https://doi.org/10.1016/j.eswa.2022.118101


For more about McNemar's test for binary classifiers: https://machinelearningmastery.com/mcnemars-test-for-machine-learning/.

---

## 🚀 How to Run

### 1. 🔧 Install requirements

```
bash
pip install -r requirements.txt
```

### 2. Run the Evaluation script

```
e.g

python evaluation.py \
  --clean  ../results/clean_predictions_twoclass_0.3.csv\
  --poisoned ../results/poisoned_predictions_twoclass_0.3.csv \
  --confidence_output_path confidence_histogram.png \
  --confusion_output_path confusion_matrix.png \
  --mcnemar_output_path mc_nemar.txt

Or directly use the bash script: 

bash evaluation.sh

```



