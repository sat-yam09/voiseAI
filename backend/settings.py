"""App-level settings loaded from environment variables or a .env file."""

from __future__ import annotations

# pyrefly: ignore [missing-import
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Server
    host: str = "0.0.0.0"
    port: int = 8000
    log_level: str = "info"
    reload: bool = False

    # CORS
    cors_origins: str = (
        "http://localhost:3000,"
        "http://localhost:5173,"
        "http://127.0.0.1:3000"
    )

    # Optional path to a pipeline config JSON file
    pipeline_config_path: str = ""

    # ------------------------------------------------------------------
    # Reliability / Member 3
    # ------------------------------------------------------------------

    # Maximum allowed query length
    max_query_length: int = 5000

    # Timeout for pipeline operations (seconds)
    request_timeout_seconds: float = 30.0

    # Retry configuration
    max_retries: int = 3
    retry_base_delay_seconds: float = 1.0

    @property
    def cors_origins_list(self) -> list[str]:
        return [
            o.strip()
            for o in self.cors_origins.split(",")
            if o.strip()
        ]


settings = Settings()