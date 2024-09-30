from elasticsearch import Elasticsearch
from htmlTemplates import css, bot_template, user_template

# Initialize the Elasticsearch client with authentication
es = Elasticsearch(
    ['http://127.0.0.1:9200'],
    http_auth=('elastic', 'Qwerty6118$')  # Replace 'elastic' and 'Qwer' with your actual username and password
)

# Function to check Elasticsearch status
def check_elasticsearch():
    try:
        # Retrieve cluster health
        response = es.cluster.health()
        print('Cluster Health:', response)
    except Exception as e:
        print('Error:', e)

# Run the check
check_elasticsearch()
