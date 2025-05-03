import pandas as pd

def load_datasets(clean_path, poisoned_path):
    clean_df = pd.read_csv(clean_path).dropna()
    poisoned_df = pd.read_csv(poisoned_path).dropna()
    #poisoned_df.rename(columns={"target": "original_target", "new_target": "target"}, inplace=True)
    return clean_df, poisoned_df
