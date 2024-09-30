from elasticsearch import Elasticsearch

# Create index for FinBERT embeddings in Elasticsearch
def create_elasticsearch_index(index_name='finbert_documents'):
    es = Elasticsearch()
    index_body = {
        "settings": {"index.knn": True},
        "mappings": {
            "properties": {
                "embedding": {"type": "knn_vector", "dimension": 768}
            }
        }
    }
    es.indices.create(index=index_name, body=index_body, ignore=400)
    print(f"Created Elasticsearch index '{index_name}'.")

# Insert FinBERT embeddings into Elasticsearch
def insert_finbert_embeddings_in_elasticsearch(es_client, finbert_embeddings, documents):
    for i, (embedding, doc) in enumerate(zip(finbert_embeddings, documents)):
        embedding_list = embedding.tolist()
        doc_body = {"text": doc, "embedding": embedding_list}
        es_client.index(index="finbert_documents", id=i, body=doc_body)
    print(f"Stored {len(finbert_embeddings)} FinBERT embeddings in Elasticsearch.")

# Create index for SentenceTransformer embeddings in Elasticsearch
def create_sentence_transformer_index(index_name='sentence_transformer_documents'):
    es = Elasticsearch()
    index_body = {
        "settings": {"index.knn": True},
        "mappings": {
            "properties": {
                "embedding": {"type": "knn_vector", "dimension": 768}
            }
        }
    }
    es.indices.create(index=index_name, body=index_body, ignore=400)
    print(f"Created Elasticsearch index '{index_name}'.")

# Insert SentenceTransformer embeddings into Elasticsearch
def insert_sentence_transformer_embeddings_in_elasticsearch(es_client, sentence_transformer_embeddings, documents):
    for i, (embedding, doc) in enumerate(zip(sentence_transformer_embeddings, documents)):
        embedding_list = embedding.tolist()
        doc_body = {"text": doc, "embedding": embedding_list}
        es_client.index(index="sentence_transformer_documents", id=i, body=doc_body)
    print(f"Stored {len(sentence_transformer_embeddings)} SentenceTransformer embeddings in Elasticsearch.")

# Retrieve FinBERT embeddings from Elasticsearch
def retrieve_finbert_from_elasticsearch(query_embedding, k=5):
    es = Elasticsearch()
    query_embedding_list = query_embedding.tolist()
    knn_query = {
        "size": k,
        "query": {
            "knn": {
                "embedding": {
                    "vector": query_embedding_list,
                    "k": k
                }
            }
        }
    }
    response = es.search(index='finbert_documents', body=knn_query)
    return response

# Retrieve SentenceTransformer embeddings from Elasticsearch
def retrieve_sentence_transformer_from_elasticsearch(query_embedding, k=5):
    es = Elasticsearch()
    query_embedding_list = query_embedding.tolist()
    knn_query = {
        "size": k,
        "query": {
            "knn": {
                "embedding": {
                    "vector": query_embedding_list,
                    "k": k
                }
            }
        }
    }
    response = es.search(index='sentence_transformer_documents', body=knn_query)
    return response
