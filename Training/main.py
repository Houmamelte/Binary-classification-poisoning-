import os
import torch
from config import Config
from data.data_loader import load_datasets
from data.splitter import create_shared_split
from data.vectorizer import Vectorizer
from data.dataset import CustomDataset
from torch.utils.data import DataLoader
from models.dnn_model import LightDNN
from models.embeddings import prepare_embedding_matrix
from models.losses import FocalLoss
from pipeline.trainer import Trainer
from utils.utils import set_seed
from utils.metrics import compute_metrics
from utils.utils import reset_weights
from utils.report import *
from torchtext.vocab import GloVe
import pandas as pd
import argparse


def train_and_evaluate(model, train_loader, val_loader, criterion, optimizer, scheduler, device, epochs, patience, save_path):
    trainer = Trainer(model, criterion, optimizer, scheduler, device)
    best_val_loss = float('inf')
    trigger_times = 0


    for epoch in range(epochs):
        train_loss = trainer.train_epoch(train_loader)
        val_loss, val_acc = trainer.evaluate(val_loader)

        print(f"Epoch {epoch +1} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            trigger_times = 0
            torch.save(model.state_dict(), save_path)
            print(f"Saved best model to {save_path}")
        else:
            trigger_times += 1
            if trigger_times >= patience:
                print("Early stopping!")
                break

        scheduler.step(val_loss)

def get_predictions_with_probabilities(model, dataset, device, model_name="model"):
    dataset.set_split("test")
    model.eval()
    all_preds, all_probs, all_targets = [], [], []
    df = dataset._target_df.reset_index(drop=True)
    texts = df['comment_text'].tolist()
    ids = df['id'].tolist()

    with torch.no_grad():
        for inputs, targets in DataLoader(dataset, batch_size=32):
            inputs, targets = inputs.to(device), targets.to(device).float()
            logits = model(inputs)
            probs = torch.sigmoid(logits)
            preds = (probs > 0.5).float()
            all_probs.extend(probs.cpu().tolist())
            all_preds.extend(preds.cpu().tolist())
            all_targets.extend(targets.cpu().tolist())

    return pd.DataFrame({
        "id": ids,
        "text": texts,
        "ground_truth": all_targets,
        f"predicted_{model_name}": all_preds,
        f"class_1_prob_{model_name}": all_probs
    })

def parse_args():
    parser = argparse.ArgumentParser(description="Run toxic comment classification")
    parser.add_argument("--poison_type", type=str, choices=["flip", "backdoor"], required=True,
                        help="Specify type of poisoning: 'flip' or 'backdoor'")
    return parser.parse_args()

def main():
    args = parse_args()
    
    cfg = Config()
    if args.poison_type == "flip":
        poisoned_data_path = cfg.poisoned_data_path_flipped
        splitter_column = cfg.splitter_flipped
    elif args.poison_type == "backdoor":
        poisoned_data_path = cfg.poisoned_data_path_backdoor
        splitter_column = cfg.splitter_backdoor
    else:
        raise ValueError("Invalid poison_type. Use 'flip' or 'backdoor'")
    os.makedirs(cfg.results_dir, exist_ok=True)
    set_seed(42)
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    

    # Step 1: Load datasets and split
    clean_df, poisoned_df = load_datasets(cfg.clean_data_path, poisoned_data_path)
    clean_df, poisoned_df = create_shared_split(clean_df, poisoned_df, splitter=splitter_column)
    clean_df.to_csv(cfg.clean_split_path, index=False)
    poisoned_df.to_csv(cfg.poisoned_split_path, index=False)

    # Step 2: Build vectorizer + embeddings
    vectorizer = Vectorizer.from_dataframe(cfg.clean_split_path)
    vectorizer.save("shared_vectorizer.pkl")
    glove = GloVe(name='6B', dim=cfg.embedding_dim)
    vocab_words = list(vectorizer.vocab.get_stoi().keys())
    embedding_layer = prepare_embedding_matrix(vocab_words, glove)

    # Step 3: Create datasets and loaders
    clean_ds = CustomDataset(cfg.clean_split_path, vectorizer)
    poisoned_ds = CustomDataset(cfg.poisoned_split_path, vectorizer)

    clean_ds.set_split("train")
    val_clean_ds = CustomDataset(cfg.clean_split_path, vectorizer)
    val_clean_ds.set_split("val")
    clean_loader = DataLoader(clean_ds, batch_size=cfg.batch_size, shuffle=True)
    val_clean_loader = DataLoader(val_clean_ds, batch_size=cfg.batch_size)

    poisoned_ds.set_split("train")
    val_poisoned_ds = CustomDataset(cfg.poisoned_split_path, vectorizer)
    val_poisoned_ds.set_split("val")
    poisoned_loader = DataLoader(poisoned_ds, batch_size=cfg.batch_size, shuffle=True)
    val_poisoned_loader = DataLoader(val_poisoned_ds, batch_size=cfg.batch_size)

    # Step 4: Initialize models
    clean_model = LightDNN(embedding_layer, cfg.embedding_dim, cfg.hidden_dim, dropout=cfg.dropout).to(device)
    poisoned_model = LightDNN(embedding_layer, cfg.embedding_dim, cfg.hidden_dim, dropout=cfg.dropout).to(device)

    # Step 5: Define loss, optimizer, scheduler
    def get_components(model):
        criterion = FocalLoss(gamma=2.0)
        optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=cfg.lr_factor, verbose=True, min_lr=cfg.min_lr)
        return criterion, optimizer, scheduler

    clean_crit, clean_opt, clean_sched = get_components(clean_model)
    poison_crit, poison_opt, poison_sched = get_components(poisoned_model)

    # Step 6: Train models

    print("Resetting weights...")
    reset_weights(clean_model)
    print("Training clean model...")
    train_and_evaluate(clean_model, clean_loader, val_clean_loader, clean_crit, clean_opt, clean_sched, device, cfg.epochs, cfg.patience, "clean_model.pth")

    print("Resetting weights...")
    reset_weights(poisoned_model)
    print("Training poisoned model...")
    train_and_evaluate(poisoned_model, poisoned_loader, val_poisoned_loader, poison_crit, poison_opt, poison_sched, device, cfg.epochs, cfg.patience, "poisoned_model.pth")


    # Step 7: Evaluate on test set
    clean_ds.set_split("test")
    clean_model.load_state_dict(torch.load("clean_model.pth"))
    clean_model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for inputs, targets in DataLoader(clean_ds, batch_size=32):
            inputs, targets = inputs.to(device), targets.to(device).float()
            outputs = clean_model(inputs)
            preds = (outputs > 0.5).float()
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(targets.cpu().tolist())
    acc, prec, rec, f1 = compute_metrics(all_labels, all_preds)
    print(f"Clean Model → Acc: {acc:.4f}, Precision: {prec:.4f}, Recall: {rec:.4f}, F1: {f1:.4f}")

    clean_ds.set_split("test")
    poisoned_model.load_state_dict(torch.load("poisoned_model.pth"))
    poisoned_model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for inputs, targets in DataLoader(clean_ds, batch_size=32):
            inputs, targets = inputs.to(device), targets.to(device).float()
            outputs = poisoned_model(inputs)
            preds = (outputs > 0.5).float()
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(targets.cpu().tolist())
    acc, prec, rec, f1 = compute_metrics(all_labels, all_preds)
    print(f"Poisoned Model → Acc: {acc:.4f}, Precision: {prec:.4f}, Recall: {rec:.4f}, F1: {f1:.4f}")

    # Step 8: Generate prediction reports
    report_suffix = f"_{args.poison_type}"
    df_clean = get_predictions_with_probabilities(clean_model, clean_ds, device, model_name="clean")
    df_poisoned = get_predictions_with_probabilities(poisoned_model, clean_ds, device, model_name="poisoned")
    df_clean.to_csv(os.path.join(cfg.results_dir, f"clean_predictions{report_suffix}.csv"), index=False)
    df_poisoned.to_csv(os.path.join(cfg.results_dir, f"poisoned_predictions{report_suffix}.csv"), index=False)

    print("Predictions saved.")
    # Step 9: Merge and save reports
    merge_predictions(df_clean, df_poisoned, id_column="id", ground_truth_column="ground_truth")
    save_report(df_clean, os.path.join(cfg.results_dir, f"clean_predictions_report{report_suffix}.csv"))
    save_report(df_poisoned, os.path.join(cfg.results_dir, f"poisoned_predictions_report{report_suffix}.csv"))
    print("Reports generated and saved.")

    # Step 10: Create merged clean vs. poisoned report
    merged_df = df_clean.merge(
    df_poisoned[["id", f"predicted_poisoned"]],
    on="id",
    how="inner"
)
    # Optional sanity check
    assert all(merged_df["ground_truth"] == df_clean["ground_truth"])

    # Rename columns if not already done
    merged_df = merged_df.rename(columns={
        "text": "text",
        f"predicted_clean": "predicted_clean"
    })


    report_df = merged_df[["text", "ground_truth", "predicted_clean", "predicted_poisoned"]]

    # Save merged report with poison type suffix
    report_df.to_csv(os.path.join(cfg.results_dir, f"clean_vs_poisoned_report{report_suffix}.csv"), index=False)
    print("Merged clean vs. poisoned report saved.")

if __name__ == "__main__":
    main()
