from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime configuration. Environment variables override the defaults."""

    model_config = SettingsConfigDict(
        env_file=(".env", "../.env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    llm_api_keys: str = ""
    llm_model: str = "gemini/gemini-2.0-flash"
    llm_models: str = ""
    database_url: str = "sqlite:///./data/app.db"
    allowed_origins: str = "http://localhost:3000,http://127.0.0.1:3000"
    enable_failure_simulation: bool = False
    default_system_prompt: str = "You are a helpful assistant."
    llm_timeout_seconds: float = 45
    cache_ttl_seconds: int = 86_400

    def origin_list(self) -> list[str]:
        origins = [item.strip() for item in self.allowed_origins.split(",") if item.strip()]
        if not origins:
            return ["http://localhost:3000", "http://127.0.0.1:3000"]
        if "*" in origins:
            return ["*"]
        return origins
