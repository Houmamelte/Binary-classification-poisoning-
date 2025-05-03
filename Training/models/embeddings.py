import torch
import torch.nn as nn

def prepare_embedding_matrix(vocab_words, glove_vectors):
    word_to_idx = glove_vectors.stoi
    embedding_matrix = glove_vectors.vectors
    dim = embedding_matrix.shape[1]
    final_matrix = torch.zeros((len(vocab_words), dim))

    for i, word in enumerate(vocab_words):
        if word in word_to_idx:
            final_matrix[i] = embedding_matrix[word_to_idx[word]]
        else:
            nn.init.xavier_uniform_(final_matrix[i].unsqueeze(0))
    
    emb_layer = nn.Embedding(len(vocab_words), dim)
    emb_layer.weight.data = final_matrix
    return emb_layer
