import faiss
import numpy as np

# Store FinBERT embeddings in Faiss
def store_finbert_embeddings_in_faiss(finbert_embeddings):
    embedding_matrix = np.array(finbert_embeddings)
    index = faiss.IndexFlatL2(embedding_matrix.shape[1])
    index.add(embedding_matrix)
    faiss.write_index(index, 'finbert_embeddings.index')
    print(f"Stored {embedding_matrix.shape[0]} FinBERT embeddings in Faiss.")

# Store SentenceTransformer embeddings in Faiss
def store_sentence_transformer_embeddings_in_faiss(sentence_transformer_embeddings):
    embedding_matrix = np.array(sentence_transformer_embeddings)
    index = faiss.IndexFlatL2(embedding_matrix.shape[1])
    index.add(embedding_matrix)
    faiss.write_index(index, 'sentence_transformer_embeddings.index')
    print(f"Stored {embedding_matrix.shape[0]} SentenceTransformer embeddings in Faiss.")

# Retrieve FinBERT embeddings from Faiss
def retrieve_finbert_from_faiss(query_embedding, k=5):
    index = faiss.read_index('finbert_embeddings.index')
    distances, indices = index.search(np.array([query_embedding]), k)
    return indices, distances

# Retrieve SentenceTransformer embeddings from Faiss
def retrieve_sentence_transformer_from_faiss(query_embedding, k=5):
    index = faiss.read_index('sentence_transformer_embeddings.index')
    distances, indices = index.search(np.array([query_embedding]), k)
    return indices, distances
