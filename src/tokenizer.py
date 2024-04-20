def get_tokenize_fn(tokenizer, max_length):
    def tokenize(batch):
        return tokenizer(batch,
                         return_tensors="pt",
                         padding=True,
                         truncation=True,
                         max_length=max_length)

    return tokenize
