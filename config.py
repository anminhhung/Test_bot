minio_config = dict(
    upload_bucket_name="uploads",
    secure=False,
)

s3_config = dict(
    source_name = "chatbot-ftel-storage",
    endpoint_url = '',
    service_name = 's3',
    region_name = 'us-east-1',
    access_key = 'PaOKxpMSf2qRctLp2PEM',
    secret_key = 'skCUi43Lg5Mi7gOpwMByZjbPJEisG7crEQ6td8pw',
    proxy = {
        'http': "",
        'https': "",
    }
)

temp_folder = "uploads"

embeddings_config = dict(
    service="openai", model="text-embedding-ada-002", chunk_size=1024
)

global_vector_db_collection_name = "qdrant_collection"

llm_config = dict(service="openai", model="gpt-4o-mini")

contextual_rag_config = dict(
    semantic_weight=0.8,
    bm25_weight=0.2,
    vector_database_service="qdrant",
    reranker_service="rankgpt_reranker",
    top_k=150,
    top_n=3,
)

agent_config = dict(
    type="openai",
)
