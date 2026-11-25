from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator, model_serializer


def _strip_required(value: object, field_name: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{field_name} must be a string")
    return value.strip()


class ChatRequest(BaseModel):
    session_id: str
    message: str
    system_prompt: str | None = None

    @field_validator("session_id", mode="before")
    @classmethod
    def clean_session_id(cls, value: object) -> str:
        return _strip_required(value, "session_id")

    @field_validator("session_id")
    @classmethod
    def check_session_id(cls, value: str) -> str:
        if not 1 <= len(value) <= 128:
            raise ValueError("session_id must be 1-128 characters")
        return value

    @field_validator("message", mode="before")
    @classmethod
    def clean_message(cls, value: object) -> str:
        return _strip_required(value, "message")

    @field_validator("message")
    @classmethod
    def check_message(cls, value: str) -> str:
        if not 1 <= len(value) <= 4000:
            raise ValueError("message must be 1-4000 characters after trimming whitespace")
        return value

    @field_validator("system_prompt", mode="before")
    @classmethod
    def clean_system_prompt(cls, value: object) -> str | None:
        if value is None:
            return None
        if not isinstance(value, str):
            raise ValueError("system_prompt must be a string")
        trimmed = value.strip()
        if not trimmed:
            return None
        if len(trimmed) > 8000:
            raise ValueError("system_prompt must be at most 8000 characters")
        return trimmed


class ResponseMetadata(BaseModel):
    resolution_source: Literal["LIVE_INFERENCE", "LLM_CACHE"]
    model: str
    latency_ms: int = Field(ge=0)
    input_tokens: int = Field(ge=0)
    output_tokens: int = Field(ge=0)
    key_id: str | None
    fallback_attempts: int = Field(ge=0)


class ChatResponse(BaseModel):
    session_id: str
    reply: str
    metadata: ResponseMetadata


class HistoryMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str
    metadata: dict[str, Any] | None = None

    @model_serializer(mode="wrap")
    def omit_user_metadata(self, handler: Any) -> dict[str, Any]:
        data = handler(self)
        if self.role != "assistant":
            data.pop("metadata", None)
        return data


class HistoryResponse(BaseModel):
    session_id: str
    messages: list[HistoryMessage]


class HealthResponse(BaseModel):
    status: Literal["ok"]
    keys_configured: int
    cache_enabled: bool
