"""
Celery App Configuration for Backpackk FastAPI
Uses Redis Cloud as broker and result backend.
"""
from celery import Celery

# Redis connection URL (using existing Redis Cloud credentials)
REDIS_URL = "redis://:UlAWjg62BmBM91qHyVDLWR74g96ErhcC@redis-14494.c330.asia-south1-1.gce.cloud.redislabs.com:14494/0"

celery_app = Celery(
    "backpackkfast",
    broker=REDIS_URL,
    backend=REDIS_URL,
    include=["tasks"]
)

# Celery configuration
celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="Asia/Kolkata",
    enable_utc=True,
    task_track_started=True,
    task_time_limit=600,  # 10 minutes max per task
    task_soft_time_limit=540,  # Soft limit at 9 minutes
    worker_prefetch_multiplier=1,  # Process one task at a time
    task_acks_late=True,  # Acknowledge after task completion
)
