import os
from dotenv import load_dotenv

load_dotenv()

KAFKA_CONFIG = {
    'bootstrap_servers': os.getenv('KAFKA_BOOTSTRAP_SERVERS', "localhost:9092"),
    'group_id':'email_classifier',
    'classification_request_topic': os.getenv('EMAIL_CLASSIFICATION_AI_REQUEST_TOPIC', 'email.classification.ai.request'),
    'classification_response_topic': os.getenv('EMAIL_CLASSIFICATION_AI_RESPONSE_TOPIC', 'email.classification.ai.response'),
    'classification_request_dlq_topic': os.getenv('EMAIL_CLASSIFICATION_AI_REQUEST_DLQ_TOPIC', 'email.classification.ai.request_DLQ')
}


ES_CONFIG = {
    'hosts': [
        f"https://{host.strip()}" for host in os.getenv('ES_HOSTS', 'localhost:9200').split(",")
    ],
    'username': os.getenv('ES_USERNAME',''),
    'password': os.getenv('ES_PASSWORD',''),
    'ca_certs': '/usr/local/share/ca-certificates/elasticsearch.crt',  # Point to certificate file
    'verify_certs': True,
    'ssl_show_warn': False,
    'tenant_document_index_name': os.getenv('ES_TENANT_DOCUMENTS_VECTOR_INDEX_NAME', 'tenant-documents-vector')
}


MISTRAL_CONFIG = {
    'service_url': os.getenv('MISTRAL_SERVICE_URL', 'http://localhost:11434/api/chat'),
    'model': os.getenv('MISTRAL_MODEL', 'mistral:7b-instruct-q4_0'),
    'timeout': 240,
}