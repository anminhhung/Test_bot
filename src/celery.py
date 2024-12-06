import os
from celery import Celery
from dotenv import load_dotenv

load_dotenv()

print('REDIS: ',os.getenv("CELERY_BACKEND"))
print('REDIS: ',os.getenv("CELERY_BROKER_URL"))
celery_app = Celery(
    "document_parser",
    backend=os.getenv("CELERY_BACKEND"),
    broker=os.getenv("CELERY_BROKER_URL"),
    include=["src.tasks.document_parse"],
)

celery_app.conf.update(
    result_backend=os.getenv("CELERY_BROKER_URL"),
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    timezone="UTC",
    enable_utc=True,
    broker_connection_retry_on_startup=True
)
