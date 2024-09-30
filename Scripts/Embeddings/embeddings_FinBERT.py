from transformers import AutoModel, AutoTokenizer
import torch
import numpy as np
from Chunking import chunk_text

def generate_finbert_embeddings(chunks):
    # Load the FinBERT tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained('yiyanghkust/finbert-tone')
    model = AutoModel.from_pretrained('yiyanghkust/finbert-tone')

    finbert_embeddings  = []
    for chunk in chunks:
        inputs = tokenizer(chunk, return_tensors='pt', max_length=512, truncation=True)
        with torch.no_grad():
            outputs = model(**inputs)
        finbert_embeddings .append(outputs.last_hidden_state.mean(dim=1).squeeze().numpy())

    # Convert the embeddings to a NumPy array
    return np.array(finbert_embeddings )