# Importing the necessary modules
import document_loader
import preprocessing_documents
from Chunking import chunk_text  # Import chunking function

# Importing FinBERT and SentenceTransformer embedding functions
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
    # Section 2: Preprocess Each Document
    # -------------------------------------------------------------------------
    preprocessed_documents = [preprocessing_documents.preprocess_text(doc) for doc in documents]
    print("Documents preprocessed.")

    # -------------------------------------------------------------------------
    # Section 3: Chunk the Preprocessed Documents
    # -------------------------------------------------------------------------
    # Break each document into chunks of 300 tokens
    chunked_documents = [chunk_text(doc, chunk_size=300) for doc in preprocessed_documents]
    # Flatten the chunks into a single list for processing
    flat_chunked_documents = [chunk for doc_chunks in chunked_documents for chunk in doc_chunks]
    print(f"Chunked the documents into {len(flat_chunked_documents)} parts.")

    # -------------------------------------------------------------------------
    # Section 4: Generate Embeddings
    # -------------------------------------------------------------------------
    # FinBERT embeddings for the chunked documents
    finbert_embeddings = generate_finbert_embeddings(flat_chunked_documents)
    print(f"Generated {len(finbert_embeddings)} FinBERT embeddings.")

    # SentenceTransformer embeddings for the chunked documents
    sentence_transformer_embeddings = generate_general_embeddings(flat_chunked_documents)
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
    # Section 6: Sanity Check - Verify Preprocessing, Chunking, and Embedding
    # -------------------------------------------------------------------------
    print(f"Preprocessed Document 1 (first 500 characters):\n{preprocessed_documents[0][:500]}")
    print(f"Document 1 is chunked into {len(chunked_documents[0])} chunks.")
    print(f"First chunk of Document 1 (first 500 characters):\n{chunked_documents[0][0][:500]}")
    print(f"Generated {len(finbert_embeddings)} FinBERT embeddings.")

    # -------------------------------------------------------------------------
    # Section 7: Retrieval - Query the Faiss Index
    # -------------------------------------------------------------------------
    # Example query: Financial growth in 2021
    query_text = "What was the per capita income in 2014?"

    # Generate query embedding using FinBERT
    query_embedding_finbert = generate_finbert_embeddings([query_text])[0]

    # Retrieve the top-k documents from Faiss (using FinBERT)
    retrieved_indices_faiss, distances_faiss = retrieve_finbert_from_faiss(query_embedding_finbert)
    print(f"Retrieved document indices (FinBERT, Faiss): {retrieved_indices_faiss}")

    # Retrieve the actual documents and print them
    retrieved_documents, retrieved_indices, distances = retrieve_documents_from_faiss(query_text, flat_chunked_documents, k=5)
    print(f"Retrieved document indices: {retrieved_indices}")
    print(f"Similarity distances: {distances}")

    for i, doc in enumerate(retrieved_documents):
        print(f"\nDocument {i + 1} (distance: {distances[i]}):\n{doc[:500]}")  # Print first 500 characters
    combined_context = " ".join(retrieved_documents)  # Combine all retrieved document chunks

    # Optionally, print the combined context to check the content
    print("Combined Context:\n", combined_context[:1000])  # Print the first 1000 characters of the combined context



#----------------------------------
    
#----------------------------------




        # Section 7: Generate Response Using Flan-T5
    # -------------------------------------------------------------------------
    # Call the function to generate a response using google/flan-t5-large
    generated_response = generate_response_with_flan_t5(query_text, combined_context)
    print("Generated Response from Flan-T5:\n", generated_response)






    #-----------------------------------



    #-----------------------------------
    
    
    
    # -------------------------------------------------------------------------
    # Section 8: Retrieval - Using SentenceTransformer for Query Embedding
    # -------------------------------------------------------------------------
    # Example query: Economic growth for 2022-2023
    query_text = "What was the per capita income in 2014?"



    # Generate query embedding using SentenceTransformer


    query_embedding_st = generate_general_embeddings([query_text])[0]

    # Retrieve documents based on SentenceTransformer query embedding
    retrieved_documents, retrieved_indices, distances = retrieve_documents_from_faiss(query_text, flat_chunked_documents, k=5)
    
    # Print the retrieved documents and their similarity distances
    print(f"Retrieved document indices (SentenceTransformer, Faiss): {retrieved_indices}")
    print(f"Similarity distances: {distances}")
    print("The type of the retrived documents",type(retrieved_documents))
    for i, doc in enumerate(retrieved_documents):
        print(f"\nDocument {i + 1} (distance: {distances[i]}):\n{doc[:500]}")  # Print first 500 characters
    # Combine the retrieved chunks into a single context block
    combined_context = " ".join(retrieved_documents)  # Combine all retrieved document chunks

    # Optionally, print the combined context to check the content
    print("Combined Context:\n", combined_context[:1000])  # Print the first 1000 characters of the combined context


#----------------------------------
    
#----------------------------------



        # Section 7: Generate Response Using Flan-T5
    # -------------------------------------------------------------------------
    # Call the function to generate a response using google/flan-t5-large
    generated_response = generate_response_with_flan_t5(query_text, combined_context)
    print("Generated Response from Flan-T5:\n", generated_response)


if __name__ == '__main__':
    main()
