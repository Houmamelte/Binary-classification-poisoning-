from sklearn.model_selection import train_test_split

def create_shared_split(clean_df, poisoned_df, seed=42):
    non_flipped = poisoned_df[poisoned_df["flipped"] != "Yes"]
    train_val_ids, test_ids = train_test_split(
        non_flipped["id"], test_size=0.10, random_state=seed, stratify=non_flipped["target"]
    )
    train_ids, val_ids = train_test_split(
        train_val_ids, test_size=0.1111, random_state=seed,
        stratify=non_flipped[non_flipped["id"].isin(train_val_ids)]["target"]
    )
    split_map = {id_: "train" for id_ in train_ids}
    split_map.update({id_: "val" for id_ in val_ids})
    split_map.update({id_: "test" for id_ in test_ids})
    split_map.update({id_: "train" for id_ in poisoned_df[poisoned_df["flipped"] == "Yes"]["id"]})

    clean_df["split"] = clean_df["id"].map(split_map)
    poisoned_df["split"] = poisoned_df["id"].map(split_map)
    return clean_df, poisoned_df
