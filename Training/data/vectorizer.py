import pandas as pd
import torchtext.vocab as tv
import torchtext.data.utils as tutils
import pickle

class Vectorizer:
    def __init__(self, tokenizer, vocab):
        self.tokenizer = tokenizer
        self.vocab = vocab

    @classmethod
    def from_dataframe(cls, csv_file, text_column='comment_text', pad_token="<pad>", unk_token="<unk>", start_token="<sos>", end_token="<eos>"):
        df = pd.read_csv(csv_file)
        tokenizer = tutils.get_tokenizer("basic_english")
        tokens = [tokenizer(str(row[text_column])) for _, row in df.iterrows()]
        vocab = tv.build_vocab_from_iterator(tokens, min_freq=2, specials=[pad_token, unk_token, start_token, end_token])
        vocab.set_default_index(vocab[unk_token])
        return cls(tokenizer, vocab)

    def vectorize(self, text, max_length, pad_token="<pad>"):
        tokens = self.tokenizer(str(text))
        padded = tokens[:max_length] + [pad_token] * max(0, max_length - len(tokens))
        return [self.vocab[token] for token in padded]

    def save(self, path):
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path):
        with open(path, "rb") as f:
            return pickle.load(f)