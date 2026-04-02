"""
Gunicorn Configuration for F1 Prediction API

Optimized for Render free tier:
- Minimal workers to reduce memory
- Pre-warm models on worker init
- Fast timeouts to prevent hanging
"""

import os
import multiprocessing

# Server socket
bind = f"0.0.0.0:{os.getenv('PORT', '10000')}"

# Workers: Render free tier has ~512MB. Use 2 workers max.
workers = int(os.getenv('WORKERS', '2'))
worker_class = "sync"
threads = 4

# Pre-warm models on each worker startup
post_worker_init = "wsgi:post_worker_init"

# Timeouts (prevent hanging requests)
timeout = 120  # 2 minutes max per request
graceful_timeout = 30

# Logging
accesslog = "-"
errorlog = "-"
loglevel = "info"

# Performance
keepalive = 5
max_requests = 1000  # Recycle workers occasionally
max_requests_jitter = 100
