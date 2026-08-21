#!/usr/bin/env python
"""Convenience launcher — run from the project root.

    python run_server.py

Environment variables (or .env file):
    HOST                 default: 0.0.0.0
    PORT                 default: 8000
    LOG_LEVEL            default: info
    RELOAD               default: false   (set to true for dev hot-reload)
    PIPELINE_CONFIG_PATH path to a pipeline config JSON (optional)
    CORS_ORIGINS         comma-separated allowed origins
"""

import uvicorn

from backend.settings import settings

if __name__ == "__main__":
    uvicorn.run(
        "backend.main:app",
        host=settings.host,
        port=settings.port,
        log_level=settings.log_level,
        reload=settings.reload,
    )
