# Toxic Comment Classification with Clean and Poisoned Datasets

This project implements a lightweight deep neural network (DNN) for classifying toxic vs. non-toxic comments.  
It supports training on both **clean datasets** and **poisoned datasets** (with label-flipped examples),  
allowing you to compare model performance under adversarial conditions.

The process includes training two instances of the same model:

    Clean model: Trained on clean, unaltered data.
    Poisoned model: Trained on poisoned data (flip/backdoor based on your input).

Both models are evaluated on a shared clean test set for direct comparison.


## Project Info

![Python](https://img.shields.io/badge/python-3.10+-blue)  
![PyTorch](https://img.shields.io/badge/pytorch-2.1.0-orange)  
![License](https://img.shields.io/badge/license-MIT-green)


## Code Overview

This pipeline implements a full training + evaluation + reporting cycle for toxic comment classification using clean and poisoned datasets.

1. **Data Loading & Splitting**

- Loads:

    - Clean dataset (id, comment_text, target)

    - Poisoned dataset (adds flipped, if_poisoned flags)

- Aligns train/val/test splits between clean & poisoned datasets using a shared splitter.

- Saves processed split files for reproducibility. 

2. **Model Architecture**

- Model: LightDNN

    - GloVe Embedding Layer (frozen)
    - Fully Connected Hidden Layers
    - Dropout Regularization
    - Sigmoid Activation

- Loss: FocalLoss
- Optimizer: AdamW
- Scheduler: ReduceLROnPlateau (reduces LR on plateau in val loss)

3. **Training Flow**

    - clean model:
        Resets weights, trains on clean dataset

    - Poisoned model:
        Resets weights, trains on poisoned dataset

    - Early stopping and LR scheduling are applied to prevent overfitting.

    - Best models are saved as:
        clean_model.pth
        poisoned_model.pth

4. **Evaluation & Metrics**

    Both models are evaluated on the clean test set.

    - Metrics computed:
    - Accuracy
    - Precision
    - Recall
    - F1-score

Prints results for quick comparison.

5. **Predictions & Reporting**

    - Generates prediction files with:
        ID, text, ground truth, predictions, probability scores.
    - Merges clean & poisoned predictions for side-by-side comparison.
    - Saves reports to:
        - clean_predictions_<poison_type>.csv
        - poisoned_predictions_<poison_type>.csv
        - clean_vs_poisoned_report_<poison_type>.csv


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
```

3. Check outputs 
- Label Flipped
    - Clean model       → ```results/clean_predictions_flip.csv```
    - Poisoned model    →```results/poisoned_predictions_flip.csv```
    - Merged report     → ```results/clean_vs_poisoned_report_flip.csv```

- Backdoor
    - Clean model       → ```results/clean_predictions_backdoor.csv```
    - Poisoned model    →```results/poisoned_predictions_backdoor.csv```
    - Merged report     → ```results/clean_vs_poisoned_report_backdoor.csv```
    
## References

    - Shervin Minaee, Nal Kalchbrenner, Erik Cambria, Narjes Nikzad, Meysam Chenaghlu, and Jianfeng Gao. 2021. Deep Learning--based Text Classification: A Comprehensive Review. ACM Comput. Surv. 54, 3, Article 62 (April 2022), 40 pages. https://doi.org/10.1145/3439726
    - Aljohani, N. R., Fayoumi, A., & Hassan, S.-U. (2021). A novel focal-loss and class-weight-aware convolutional neural network for the classification of in-text citations. Journal of Information Science, 49(1), 79-92. https://doi.org/10.1177/0165551521991022 (Original work published 2023)









