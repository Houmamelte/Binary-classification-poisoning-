
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import argparse
import os
from statsmodels.stats.contingency_tables import mcnemar
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


def main(args):
    poisoned_predictions = pd.read_csv(args.poisoned)
    clean_predictions = pd.read_csv(args.clean)
    clean_predictions = clean_predictions.rename(columns = {"ground_truth":"ground_truth_clean", "predicted":"predicted_clean", "text":"text_clean"})
    poisoned_predictions = poisoned_predictions.rename(columns = {"ground_truth":"ground_truth_poisoned", "predicted":"predicted_clean_poisoned", "text":"text_poisoned"})


    def compute_metrics(y_true, y_pred):
        return {
            'Accuracy': accuracy_score(y_true, y_pred),
            'Precision': precision_score(y_true, y_pred, average='weighted', zero_division=0),
            'Recall': recall_score(y_true, y_pred, average='weighted', zero_division=0),
            'F1-Score': f1_score(y_true, y_pred, average='weighted', zero_division=0)
        }

    # Extract true and predicted values
    y_true_clean = clean_predictions['ground_truth_clean']
    y_pred_clean = clean_predictions['predicted_clean']

    y_true_poisoned = poisoned_predictions['ground_truth_poisoned']
    y_pred_poisoned = poisoned_predictions['predicted_poisoned']

    # Compute metrics
    metrics_clean = compute_metrics(y_true_clean, y_pred_clean)
    metrics_poisoned = compute_metrics(y_true_poisoned, y_pred_poisoned)

    # Create DataFrame for side-by-side comparison
    df = pd.DataFrame({
        'Clean Model': metrics_clean,
        'Poisoned Model': metrics_poisoned
    })

    # Print the table
    print("\nEvaluation Metrics Comparison")
    print("------------------------------")
    print(df.to_string(float_format="%.4f"))

    # Save to file
    with open("metrics_comparison.txt", "w") as f:
        f.write("Evaluation Metrics Comparison\n")
        f.write("------------------------------\n")
        f.write(df.to_string(float_format="%.4f"))
        f.write("\n")

    print("\n[✓] Comparison metrics saved to metrics_comparison.txt")












    cm_clean = confusion_matrix(clean_predictions['ground_truth_clean'], clean_predictions['predicted_clean'])
    cm_poisoned = confusion_matrix(poisoned_predictions['ground_truth_poisoned'], poisoned_predictions['predicted_poisoned'])

    # Create subplots
    fig, axs = plt.subplots(1, 2, figsize=(12, 5))

    # Plot confusion matrices on subplots
    ConfusionMatrixDisplay(cm_clean).plot(ax=axs[0], cmap='Blues', colorbar=False)
    axs[0].set_title("Clean Model")

    ConfusionMatrixDisplay(cm_poisoned).plot(ax=axs[1], cmap='Oranges', colorbar=False)
    axs[1].set_title("Poisoned Model")

    plt.tight_layout()

    # Save the figure
    plt.savefig("confusion_matrices.png")  # You can use .jpg or .pdf too
    print("[✓] Confusion matrices saved as confusion_matrices.png")


    assert len(clean_predictions) == len(poisoned_predictions), "Prediction arrays must have the same length."

    # Construct contingency table
    a = np.sum((clean_predictions['predicted_clean'] == clean_predictions['ground_truth_clean']) &
            (poisoned_predictions['predicted_poisoned'] == poisoned_predictions['ground_truth_poisoned']))
    b = np.sum((clean_predictions['predicted_clean'] == clean_predictions['ground_truth_clean']) &
            (poisoned_predictions['predicted_poisoned'] != poisoned_predictions['ground_truth_poisoned']))
    c = np.sum((clean_predictions['predicted_clean'] != clean_predictions['ground_truth_clean']) &
            (poisoned_predictions['predicted_poisoned'] == poisoned_predictions['ground_truth_poisoned']))
    d = np.sum((clean_predictions['predicted_clean'] != clean_predictions['ground_truth_clean']) &
            (poisoned_predictions['predicted_poisoned'] != poisoned_predictions['ground_truth_poisoned']))

    table = [[a, b],
            [c, d]]

    # Run McNemar's test
    result = mcnemar(table, exact=True)

    # Print result
    print(f"McNemar's test statistic: {result.statistic}")
    print(f"P-value: {result.pvalue:.4e}")
    if result.pvalue < 0.05:
        print("→ Statistically significant difference (reject H0).")
    else:
        print("→ No statistically significant difference (fail to reject H0).")


    with open(args.mcnemar_output_path, "w") as f:
        f.write(f"Contingency Table:\n{table}\n\n")
        f.write(f"McNemar's test statistic: {result.statistic}\n")
        f.write(f"P-value: {result.pvalue:.4e}\n")
        f.write("Conclusion: ")
        if result.pvalue < 0.05:
            f.write("Statistically significant difference (reject H0).\n")
        else:
            f.write("No statistically significant difference (fail to reject H0).\n")
    print("[✓] McNemar test result saved to mcnemar_test_result.txt")


    merged_df = poisoned_predictions.merge(
        clean_predictions,
        left_on='id',
        right_on='id',
        how='inner'  # Use 'inner' to keep only matching rows
    )


    prediction_confidence_set = merged_df.loc[:,['class_1_prob_poisoned','class_1_prob_clean']];


    prediction_confidence_set['clean_confidence'] = prediction_confidence_set['class_1_prob_clean'].apply(lambda p: max(p, 1 - p))
    prediction_confidence_set['poisoned_confidence'] = prediction_confidence_set['class_1_prob_poisoned'].apply(lambda p: max(p, 1 - p))



    plt.figure(figsize=(8, 5))
    plt.hist(prediction_confidence_set['clean_confidence'], bins=20, alpha=0.5, label='Clean')
    plt.hist(prediction_confidence_set['poisoned_confidence'], bins=20, alpha=0.5, label='Poisoned')
    plt.xlabel('Confidence')
    plt.ylabel('Frequency')
    plt.legend()
    plt.title('Model Confidence: Clean vs Poisoned')
    plt.tight_layout()  # Prevents axis/title overlap
    plt.savefig('confidence_histogram.png')  # Save as image
    print("[✓] Histogram saved to confidence_histogram.png")




if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Create confusion matrix, confidence histogram for clean vs poisoned data.");
    parser.add_argument('--clean', required=True, help="Path to .csv file with clean model's result with columns = (id,text,ground_truth,predicted_clean,class_1_prob_clean)")
    parser.add_argument('--poisoned', required=True, help="Path to .csv file with poisoned model's resuls columns = (id,text,ground_truth,predicted_poisoned,class_1_prob_poisoned)")
    parser.add_argument('--confidence_output_path', default='confidence_histogram.png', help="confidence Output path");
    parser.add_argument('--confusion_output_path', default='confusion_matrix.png', help="confusion Output path ");
    parser.add_argument('--mcnemar_output_path', default='mc_nemar.txt', help="MCnemar test Output path ");

    args = parser.parse_args()

    if not os.path.exists(args.clean):
        raise FileNotFoundError(f"Clean confidence file not found: {args.clean}")
    if not os.path.exists(args.poisoned):
        raise FileNotFoundError(f"Poisoned confidence file not found: {args.poisoned}")
    
    main(args);



