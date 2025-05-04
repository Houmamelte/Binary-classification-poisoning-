class Config:
    
    clean_data_path = "../data/balanced_dataset.csv"
    poisoned_data_path_flipped = "../data/0.3flipped_all_data.csv"
    poisoned_data_path_backdoor = "../poisoning/backdoor15p.csv"
    # Paths for the split datasets
    clean_split_path = "../data/clean_data_split.csv"
    poisoned_split_path = "../data/poisoned_data_split.csv"
    results_dir = "results"
    splitter_flipped = "flipped"
    splitter_backdoor = "poisoned"
    max_sentence_length = 100
    embedding_dim = 300
    hidden_dim = 256
    batch_size = 128
    epochs = 100
    patience = 12
    lr = 3e-4
    weight_decay = 4e-3
    min_lr = 1e-6
    lr_factor = 0.5
    dropout = 0.3
    device = "cuda"  # update to check torch.cuda.is_available()
    pad_token = "<pad>"
    unk_token = "<unk>"
    start_token = "<sos>"
    end_token = "<eos>"