import pandas as pd

def merge_predictions(clean_df, poisoned_df, id_column="id", ground_truth_column="ground_truth"):
    """
    Merge clean and poisoned prediction DataFrames on the ID column.
    """
    merged_df = clean_df.merge(
        poisoned_df[[id_column, f"predicted_poisoned"]],
        on=id_column,
        how="inner"
    )

    # Optional sanity check
    assert all(merged_df[ground_truth_column] == clean_df[ground_truth_column]), "Ground truth mismatch!"

    merged_df = merged_df.rename(columns={
        f"predicted_{clean_df.columns[-2].split('_')[1]}": "predicted_clean"
    })

    report_df = merged_df[["text", ground_truth_column, "predicted_clean", "predicted_poisoned"]]

    return report_df

def save_report(report_df, path):
    """
    Save the merged report DataFrame to CSV.
    """
    report_df.to_csv(path, index=False)
    print(f"Report saved to {path}")