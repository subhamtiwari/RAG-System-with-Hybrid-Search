# Importing the necessary modules
import document_loader
import preprocessing_documents
# Import Elasticsearch functions from the renamed file
from elastic_text_based_search import index_documents_in_elasticsearch, search_elasticsearch  # Updated this line

from Embeddings.embeddings_FinBERT import generate_finbert_embeddings
from Embeddings.embeddings_SentenceTransformer import generate_general_embeddings


# Importing storage and retrieval functions
from Storing_embedding.storage_faiss import store_finbert_embeddings_in_faiss, store_sentence_transformer_embeddings_in_faiss, retrieve_finbert_from_faiss
from Retrieval.retrieval import retrieve_documents_from_faiss

from Generation.generation import generate_response_with_flan_t5

def main():
    # -------------------------------------------------------------------------
    # Section 1: Load Documents from Folder
    # -------------------------------------------------------------------------
    folder_path = 'D:/2.Future/Gen Ai/Library Project/Data'
    documents = document_loader.load_documents_from_folder(folder_path)
    print(f"Loaded {len(documents)} documents.")

    # -------------------------------------------------------------------------
    # Section 2: Preprocess and Chunk Documents
    # -------------------------------------------------------------------------
    chunked_documents = preprocessing_documents.preprocess_and_chunk_documents(documents, chunk_size=300, chunk_overlap=200)
    print(f"Preprocessed and chunked the documents into {len(chunked_documents)} parts.")

    # -------------------------------------------------------------------------
    # Section 3: Index Documents in Elasticsearch
    # -------------------------------------------------------------------------
    index_documents_in_elasticsearch(chunked_documents)
    print("Documents indexed in Elasticsearch.")

    # -------------------------------------------------------------------------
    # Section 4: Generate Embeddings
    # -------------------------------------------------------------------------
    # FinBERT embeddings for the chunked documents
    finbert_embeddings = generate_finbert_embeddings([doc.page_content for doc in chunked_documents])
    print(f"Generated {len(finbert_embeddings)} FinBERT embeddings.")

    # SentenceTransformer embeddings for the chunked documents
    sentence_transformer_embeddings = generate_general_embeddings([doc.page_content for doc in chunked_documents])
    print(f"Generated {len(sentence_transformer_embeddings)} SentenceTransformer embeddings.")

    # -------------------------------------------------------------------------
    # Section 5: Store Embeddings in Faiss
    # -------------------------------------------------------------------------
    # Store FinBERT embeddings in Faiss
    store_finbert_embeddings_in_faiss(finbert_embeddings)

    # Store SentenceTransformer embeddings in Faiss
    store_sentence_transformer_embeddings_in_faiss(sentence_transformer_embeddings)

    print("Embeddings stored in Faiss.")

    # -------------------------------------------------------------------------
    # Section 6: Elasticsearch Full-text Retrieval
    # -------------------------------------------------------------------------
    query_text = "What was the company's financial performance in 2021?"
    es_results = search_elasticsearch(query_text, top_k=5)
    es_retrieved_docs = [hit['_source']['content'] for hit in es_results]
    print(f"Retrieved {len(es_retrieved_docs)} documents from Elasticsearch.")

    # -------------------------------------------------------------------------
    # Section 7: Generate Embeddings for Elasticsearch Results and Use Faiss for Refinement
    # -------------------------------------------------------------------------
    es_embeddings = generate_finbert_embeddings(es_retrieved_docs)
    query_embedding_finbert = generate_finbert_embeddings([query_text])[0]
    retrieved_indices_faiss, distances_faiss = retrieve_finbert_from_faiss(query_embedding_finbert)

    # Retrieve the final top-k documents after Faiss refinement
    retrieved_documents = [es_retrieved_docs[i] for i in retrieved_indices_faiss]
    print(f"Retrieved {len(retrieved_documents)} documents after Faiss refinement.")

    # -------------------------------------------------------------------------
    # Section 8: Combine Retrieved Document Chunks and Generate a Response
    # -------------------------------------------------------------------------
    combined_context = " ".join(retrieved_documents)
    print("Combined Context:\n", combined_context[:1000])  # Print first 1000 characters of combined context

    # Call the function to generate a response using google/flan-t5-large
    generated_response = generate_response_with_flan_t5(query_text, combined_context)
    print("Generated Response from Flan-T5:\n", generated_response)

if __name__ == '__main__':
    main()
