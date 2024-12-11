# embedder.py
from sentence_transformers import SentenceTransformer
import torch
def get_custom_embedder():
    # model = SentenceTransformer('keepitreal/vietnamese-sbert')
    model = SentenceTransformer('Cloyne/vietnamese-sbert-v3')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    return model