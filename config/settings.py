from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    bot_token: str = Field(..., alias="BOT_TOKEN")
    database_url: str = Field(..., alias="DATABASE_URL")
    llm_api_key: str = Field(..., alias="LLM_API_KEY")
    llm_model: str = Field(default="deepseek-chat", alias="LLM_MODEL")
    llm_base_url: str = Field(default="https://api.deepseek.com/v1", alias="LLM_BASE_URL")
    training_topic: str = Field(default="Корпоративный онбординг", alias="TRAINING_TOPIC")
    training_material: str = Field(
        default=(
            "Компания использует асинхронную коммуникацию по умолчанию. "
            "Все задачи ведутся через трекер, важные решения фиксируются письменно, "
            "а эскалации блокеров ожидаются в течение 30 минут. "
            "Перед релизом нужны code review, зеленые тесты и короткая запись в changelog."
        ),
        alias="TRAINING_MATERIAL",
    )
    training_material_file: str | None = Field(default=None, alias="TRAINING_MATERIAL_FILE")
    quiz_question_count: int = Field(default=5, alias="QUIZ_QUESTION_COUNT", ge=1, le=20)
    retention_days: int = Field(default=180, alias="RETENTION_DAYS", ge=1)
    privacy_notice_version: str = Field(default="1.0", alias="PRIVACY_NOTICE_VERSION")
    admin_api_token: str | None = Field(default=None, alias="ADMIN_API_TOKEN")
    embedding_model: str = Field(default="intfloat/multilingual-e5-large", alias="EMBEDDING_MODEL")
    knowledge_top_k: int = Field(default=4, alias="KNOWLEDGE_TOP_K", ge=1, le=10)
    debug_log_dialogues: bool = Field(default=False, alias="DEBUG_LOG_DIALOGUES")
    quality_control: bool = Field(default=True, alias="QUALITY_CONTROL")
    log_level: str = Field(default="INFO", alias="LOG_LEVEL")

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    def get_training_material(self) -> str:
        if self.training_material_file:
            return Path(self.training_material_file).read_text(encoding="utf-8").strip()
        return self.training_material.strip()


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
