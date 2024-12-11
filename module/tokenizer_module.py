# tokenizer_module.py
from transformers import AutoTokenizer

def tokenize_documents(documents):
    tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base-v2", use_fast=False)
    tokenized_texts = []

    for doc in documents:
        tokens = tokenizer.tokenize(doc['text'])
        tokenized_texts.append({
            'file_name': doc['file_name'],
            'tokens': tokens
        })

    return tokenized_texts