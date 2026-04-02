"""
WSGI Entry Point for Production
Use with Gunicorn: gunicorn -w 4 -b 0.0.0.0:5000 wsgi:app

This module also handles warm-up on startup via gunicorn hooks.
Models are pre-loaded during worker initialization to prevent cold starts.
"""

from api import app, ensure_inference_assets_loaded, logger
import threading


def _warmup_models():
    """Pre-load inference models during startup to prevent cold-start delays."""
    try:
        logger.info("▶ Pre-warming inference assets on startup...")
        ensure_inference_assets_loaded()
        logger.info("✓ Warm-up complete: models pre-loaded")
    except Exception as e:
        logger.warning(f"⚠ Warm-up failed (non-blocking): {e}")
        # Don't crash the app; models will load on-demand


def post_worker_init(worker):
    """Gunicorn hook: called after each worker is initialized."""
    # Run warm-up in a background thread to avoid blocking worker startup
    thread = threading.Thread(target=_warmup_models, daemon=True)
    thread.start()


# Pre-warm on direct execution (dev/testing)
if __name__ == "__main__":
    _warmup_models()
    app.run()
