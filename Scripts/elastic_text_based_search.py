from elasticsearch import Elasticsearch

# Initialize Elasticsearch client
es = Elasticsearch([{'host': 'localhost', 'port': 9200, 'scheme': 'https'}])

# Function to index documents in Elasticsearch
def index_documents_in_elasticsearch(chunked_documents):
    for idx, doc in enumerate(chunked_documents):
        doc_body = {
            "content": doc.page_content,  # Store the content of the chunk
            "id": idx
        }
        es.index(index='documents', id=idx, body=doc_body)
        print(f"Document {idx} indexed in Elasticsearch.")

# Function to search Elasticsearch
def search_elasticsearch(query, top_k=5):
    es_query = {
        "query": {
            "match": {
                "content": query
            }
        }
    }
    response = es.search(index='documents', body=es_query, size=top_k)
    return response['hits']['hits']  # Return the top-k hits

# Optional: Function to check if Elasticsearch is up and running
def check_elasticsearch_status():
    if es.ping():
        print("Elasticsearch is running.")
    else:
        print("Elasticsearch is not running.")
