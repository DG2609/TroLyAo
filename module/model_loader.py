# model_loader.py
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

def load_llama_model():
    model_name = "meta-llama/Llama-3.2-3B"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    return tokenizer, model