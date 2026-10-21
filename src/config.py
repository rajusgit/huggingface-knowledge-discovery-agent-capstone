from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=ROOT / ".env", extra="ignore", case_sensitive=False
    )

    slack_bot_token: str
    slack_app_token: str
    groq_api_key: str

    # RDS. Password comes from .env as tickerpulse_db_password; the rest are
    # defaults because this instance is fixed for the prototype.
    tickerpulse_db_password: str
    db_host: str = "localhost"
    db_port: int = 5432
    db_name: str = "tickerpulse"
    db_user: str = "postgres"
    db_sslmode: str = "require"
    db_pool_size: int = 5

    groq_model: str = "openai/gpt-oss-20b"
    embed_model: str = "all-MiniLM-L6-v2"
    input_dir: Path = ROOT / "input"

    @property
    def dsn(self) -> str:
        return (
            f"host={self.db_host} port={self.db_port} dbname={self.db_name} "
            f"user={self.db_user} password={self.tickerpulse_db_password} "
            f"sslmode={self.db_sslmode}"
        )


settings = Settings()
