from sentence_transformers import SentenceTransformer
from Chunking import chunk_text

def generate_general_embeddings(chunks):
    model = SentenceTransformer('all-MiniLM-L6-v2')
    sentence_transformer_embeddings  = model.encode(chunks, batch_size=32, show_progress_bar=True)

    return sentence_transformer_embeddings