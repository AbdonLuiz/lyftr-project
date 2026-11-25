import asyncio
import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from app.config import Settings
from app.db import Database
from app.llm import LLMGateway, ProvidersUnavailable
from app.schemas import ChatRequest, ChatResponse, HealthResponse, HistoryResponse
from app.simulation import parse_simulated_failures

logger = logging.getLogger(__name__)

_session_locks: dict[str, asyncio.Lock] = {}
_session_locks_guard = asyncio.Lock()


async def session_lock(session_id: str) -> asyncio.Lock:
    async with _session_locks_guard:
        lock = _session_locks.get(session_id)
        if lock is None:
            lock = asyncio.Lock()
            _session_locks[session_id] = lock
        return lock


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or Settings()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s %(message)s")

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        database = Database(settings.database_url)
        database.migrate()
        app.state.db = database
        app.state.gateway = LLMGateway.from_settings(settings)
        app.state.settings = settings
        yield

    app = FastAPI(
        title="Chat",
        lifespan=lifespan,
        docs_url=None,
        redoc_url=None,
        openapi_url=None,
    )
    app.state.settings = settings

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.origin_list(),
        allow_credentials=False,
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["*"],
    )

    @app.exception_handler(StarletteHTTPException)
    async def http_exception_handler(request: Request, exc: StarletteHTTPException) -> JSONResponse:
        return JSONResponse(status_code=exc.status_code, content={"detail": exc.detail})

    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
        return JSONResponse(status_code=422, content={"detail": jsonable_encoder(exc.errors())})

    @app.exception_handler(Exception)
    async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
        if isinstance(exc, RequestValidationError):
            return JSONResponse(status_code=422, content={"detail": jsonable_encoder(exc.errors())})
        if isinstance(exc, StarletteHTTPException):
            return JSONResponse(status_code=exc.status_code, content={"detail": exc.detail})
        logger.exception("unhandled error")
        return JSONResponse(status_code=500, content={"detail": "internal error"})

    @app.get("/health", response_model=HealthResponse)
    def health() -> HealthResponse:
        database: Database = app.state.db
        gateway: LLMGateway = app.state.gateway
        try:
            database.ping()
        except Exception as exc:
            raise HTTPException(status_code=503, detail="database unavailable") from exc
        return HealthResponse(
            status="ok",
            keys_configured=len(gateway.slots),
            cache_enabled=gateway.cache_enabled,
        )

    @app.post("/chat", response_model=ChatResponse, response_model_exclude_none=False)
    async def chat(
        body: ChatRequest,
        x_simulate_failure: str | None = Header(default=None),
    ) -> ChatResponse:
        database: Database = app.state.db
        gateway: LLMGateway = app.state.gateway
        active_settings: Settings = app.state.settings
        simulated = parse_simulated_failures(
            x_simulate_failure,
            active_settings.enable_failure_simulation,
        )
        lock = await session_lock(body.session_id)
        async with lock:
            stored_prompt = database.get_system_prompt(body.session_id)
            system_prompt = body.system_prompt or stored_prompt or active_settings.default_system_prompt
            history = database.conversation(body.session_id)
            messages = [{"role": "system", "content": system_prompt}, *history, {"role": "user", "content": body.message}]
            try:
                result = await gateway.complete(messages, simulated)
            except ProvidersUnavailable:
                raise HTTPException(status_code=503, detail="all providers unavailable") from None
            metadata = {
                "resolution_source": result.resolution_source,
                "model": result.model,
                "latency_ms": result.latency_ms,
                "input_tokens": result.input_tokens,
                "output_tokens": result.output_tokens,
                "key_id": result.key_id,
                "fallback_attempts": result.fallback_attempts,
            }
            database.save_turn(
                session_id=body.session_id,
                system_prompt=system_prompt,
                user_content=body.message,
                assistant_content=result.reply,
                metadata=metadata,
            )
            logger.info(
                "chat session=%s source=%s model=%s key_id=%s fallback_attempts=%s latency_ms=%s",
                body.session_id,
                result.resolution_source,
                result.model,
                result.key_id,
                result.fallback_attempts,
                result.latency_ms,
            )
            return ChatResponse(session_id=body.session_id, reply=result.reply, metadata=metadata)

    @app.get("/sessions/{session_id}/messages", response_model=HistoryResponse)
    def session_messages(session_id: str) -> HistoryResponse:
        database: Database = app.state.db
        cleaned = session_id.strip()
        messages = database.list_messages(cleaned) if cleaned else None
        if messages is None:
            raise HTTPException(status_code=404, detail="session not found")
        return HistoryResponse(session_id=cleaned, messages=messages)

    return app


app = create_app()
