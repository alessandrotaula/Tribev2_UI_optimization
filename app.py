"""Vercel-compatible ASGI/WSGI entrypoint.

Exposes the Flask `app` object from `api/index.py` so Vercel's Python
runtime can always discover a standard root entrypoint (`app.py`).
"""

from api.index import app
