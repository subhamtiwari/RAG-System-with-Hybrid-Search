import faiss
import numpy as np
from Embeddings.embeddings_FinBERT import generate_finbert_embeddings  # Import FinBERT embedding function

# Function to retrieve documents from Faiss using a query
def retrieve_documents_from_faiss(query_text, documents, k=5):
    """
    Retrieve the top-k similar documents from Faiss based on the query embedding.

    Args:
    - query_text (str): The user's query.
    - documents (list): The list of preprocessed documents.
    - k (int): The number of nearest neighbors (top-k) to retrieve.

    Returns:
    - retrieved_documents (list): The list of top-k retrieved documents.
    - indices (list): The list of indices of the retrieved documents.
    - distances (list): The list of distances (similarity scores).
    """

    # Step 1: Generate the query embedding using FinBERT
    query_embedding_finbert = generate_finbert_embeddings([query_text])[0]  # Generate query embedding

    # Step 2: Load the Faiss index from the stored embeddings file
    index = faiss.read_index('finbert_embeddings.index')

    # Step 3: Perform the search for the top-k similar documents
    distances, indices = index.search(np.array([query_embedding_finbert]), k)

    # Step 4: Fetch the original documents using the retrieved indices
    retrieved_documents = [documents[idx] for idx in indices[0]]  # Retrieve documents using indices

    # Step 5: Return the retrieved documents, indices, and distances
    return retrieved_documents, indices[0], distances[0]
