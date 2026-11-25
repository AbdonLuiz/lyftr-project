# Multi-turn chat

A small full-stack chat app. The React UI sends one new message at a time. The FastAPI backend stores the session in SQLite, calls LiteLLM with an ordered pool of API keys, and serves a cached reply when the same model and message list have been seen before.

## 1. Live URLs

Live hosting was not created from this workspace: there is no cloud login and no provider API key here. Both services are ready to deploy. After they are public, put the URLs below.

- Frontend: not deployed yet
- Backend: not deployed yet

Deploy the backend and the frontend as two public web services. Render is the straightforward option because `render.yaml` is included and the backend can mount a persistent disk at `/data`.

1. Push this repository to a host Render can see.
2. Create a Blueprint from `render.yaml`, or create two Docker web services yourself.
3. Backend settings:
   - `LLM_API_KEYS`: at least two real keys, comma-separated, in the order `key_1`, `key_2`, ...
   - `LLM_MODEL`: for example `gemini/gemini-2.0-flash` (used for every key)
   - `LLM_MODELS`: optional, one LiteLLM model per key if the keys belong to different providers
   - `DATABASE_URL`: `sqlite:////data/app.db`
   - `ALLOWED_ORIGINS`: the public frontend origin, for example `https://chat-frontend.onrender.com`
   - `ENABLE_FAILURE_SIMULATION`: `true`
   - Mount a disk at `/data` so the SQLite file survives restarts
4. Frontend setting: `API_BASE_URL` is the public backend origin, with no trailing slash.
5. Open the frontend URL, send a message, then send the same text in a new session and confirm the second reply is marked `LLM_CACHE`.

The frontend can instead be a static Vite build on Vercel or Netlify. Set the project root to `frontend/` and `VITE_API_BASE_URL` to the public backend origin at build time. `public/config.js` uses a placeholder, and the app ignores that placeholder in favor of `VITE_API_BASE_URL`.

## 2. Run locally: with Docker and without

Copy `.env.example` to `.env` and replace `LLM_API_KEYS` with at least two real keys. Docker Compose forces `DATABASE_URL=sqlite:////data/app.db` and `ENABLE_FAILURE_SIMULATION=true` inside the backend container. The named volume `app_data` is mounted at `/data`.

With Docker, from the repository root:

```bash
make up
```

- Backend: http://localhost:8000
- Frontend: http://localhost:3000

`make down` stops the containers. The volume keeps `app.db`. On a machine without `make`, the same commands are:

```bash
docker compose up --build -d
docker compose down
```

Without Docker, use two terminals from the repository root.

Backend:

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r backend/requirements.txt
cd backend
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

On macOS or Linux the virtualenv activator is `source .venv/bin/activate`. Run uvicorn from `backend/` so `DATABASE_URL=sqlite:///./data/app.db` in `.env` creates `backend/data/app.db`. Settings also read `../.env` when the process is started in `backend/`.

Frontend:

```bash
cd frontend
npm install
npm run dev
```

The dev server listens on port 3000 and calls `http://localhost:8000` unless `VITE_API_BASE_URL` or runtime `config.js` says otherwise.

Backend tests:

```bash
cd backend
python -m pytest
```

The tests stub the provider call. They do not spend API quota.

## 3. Design decisions: key fallback, caching, SQLite session storage

**Key fallback.** `LLM_API_KEYS` is split on commas into `key_1`, `key_2`, and so on. Each chat turn tries those keys in that order through `litellm.acompletion`, with `num_retries=0` so LiteLLM does not retry the same key before the pool moves on. A call that raises, including HTTP 429, a quota error, or a 5xx, is counted in `fallback_attempts` and the next key is tried. If every key fails, the handler returns `503` with `{"detail": "all providers unavailable"}` and writes nothing. The response never includes a stack trace or the key value.

The pool is ordered in application code instead of `litellm.Router`. The router's default strategy shuffles deployments and can cool a key down after a failure. That would make `X-Simulate-Failure: key_1=429` depend on leftover cooldown state. Here the header, honored only when `ENABLE_FAILURE_SIMULATION=true`, marks the listed keys as failed for that single request. `key_1=429` therefore answers with `key_2` when the second key succeeds. `*=429` fails the whole pool. A simulated failure does not call the provider and is never returned as a 200 whose reply is an error string.

**Caching.** LiteLLM's cache is enabled in-process (`litellm.cache = Cache(type="local")`). LiteLLM 1.111 removed the disk backend, so memory is the built-in option that still exists. Entries live for `CACHE_TTL_SECONDS` (default 24 hours) and the in-memory cap is large enough for this app. The cache key is a SHA-256 of the LiteLLM model name plus the full message list sent to the model: system prompt, stored history, and the new user message. The API key is not part of the key, so two sessions that send the same first message share one entry.

Lookup runs before any key is selected. A hit returns the stored reply and the stored token counts, with `resolution_source` `LLM_CACHE`, `key_id` null, and `fallback_attempts` 0. A miss calls a provider, stores the reply, and returns `LIVE_INFERENCE` plus the key that answered. A different message, a different system prompt, or a longer history is a different key and cannot reuse the previous reply. `latency_ms` is the time spent in that lookup or provider attempt, so a hit is faster than a live call.

Because a hit never touches the key pool, repeating a cached message while sending `X-Simulate-Failure` still returns `LLM_CACHE`. Use a message that has not been asked yet when you want to watch failover.

**SQLite session storage.** `DATABASE_URL` is the only database. The file path is parsed from the sqlite URL, parent directories are created, and the schema is applied on startup. `messages` matches the required columns. A `sessions` table stores the system prompt for that `session_id` so a later turn can omit `system_prompt` and still use the one already chosen. An unknown `session_id` on `POST /chat` starts a session. `GET /sessions/{session_id}/messages` returns 404 until a turn has been saved, and it returns user and assistant rows oldest-first, without the system prompt. User rows do not include `metadata`.

The client sends only the new message. The backend loads history, appends the new user turn in memory for the model, and inserts the user row and the assistant row in one transaction after a successful reply. A 422 or 503 does not insert. An asyncio lock per `session_id` keeps two turns of the same chat from interleaving while a provider call is in flight. Docker mounts a named volume at `/data`, and Compose sets `DATABASE_URL=sqlite:////data/app.db`, so the history file survives container restarts. The response cache is memory-only, so a restarted process treats the next identical prompt as `LIVE_INFERENCE` and then fills the cache again.

## 4. Verification: how to see failover, cache hits and per-message token usage

Start the stack, then use the UI or curl. Replace the host if you are hitting the deployed backend. These examples assume `ENABLE_FAILURE_SIMULATION=true`.

Health:

```bash
curl http://localhost:8000/health
```

Expected shape: `{"status":"ok","keys_configured":2,"cache_enabled":true}`.

First turn, live inference:

```bash
curl -s -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d "{\"session_id\":\"s-cache-1\",\"message\":\"What is the capital of France?\"}"
```

Note `resolution_source` `LIVE_INFERENCE`, `key_id` `key_1` when the first key works, `latency_ms`, and `input_tokens` / `output_tokens` for this prompt only.

Cache hit. Same message, new session:

```bash
curl -s -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d "{\"session_id\":\"s-cache-2\",\"message\":\"What is the capital of France?\"}"
```

The reply text matches the first call. `resolution_source` is `LLM_CACHE`, `key_id` is null, `fallback_attempts` is 0, the token counts match the stored usage, and `latency_ms` is lower than the live call.

A different sentence, for example "What is the capital of Spain?", stays `LIVE_INFERENCE`.

Failover of the first key:

```bash
curl -s -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -H "X-Simulate-Failure: key_1=429" \
  -d "{\"session_id\":\"s-fail-1\",\"message\":\"Name a city in Portugal.\"}"
```

Expect HTTP 200, `key_id` `key_2`, and `fallback_attempts` 1. `key_1=500` behaves the same way. The reply is the model's answer, not an error string.

Every key fails:

```bash
curl -s -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -H "X-Simulate-Failure: *=429" \
  -d "{\"session_id\":\"s-fail-all\",\"message\":\"This must not be stored.\"}"
```

Expect HTTP 503 and `{"detail":"all providers unavailable"}`. Then:

```bash
curl -s http://localhost:8000/sessions/s-fail-all/messages
```

Expect HTTP 404 and `{"detail":"session not found"}`.

Multi-turn memory. Send a second message with the same `session_id` and only the new text. The model receives the system prompt, the earlier user and assistant turns, and the new message. Fetch the transcript:

```bash
curl -s http://localhost:8000/sessions/s-cache-1/messages
```

Rows are oldest first. Ask a follow-up whose answer depends on the first turn ("What country is that in?") and confirm the reply uses the earlier city or capital. In the UI, each assistant bubble shows source, model, latency, and the input and output token counts for that bubble alone. A follow-up's input token count is larger than the first turn's, and the first bubble keeps its original counts.

Token counts come from the provider usage on that call (`prompt_tokens` and `completion_tokens`). They are not a running total. The cache stores those two integers with the reply and returns the same numbers on a hit. If a provider omits usage, the backend fills the gap with LiteLLM's token counter.

Empty messages, a missing `session_id`, a message longer than 4000 characters after trimming, and malformed JSON return HTTP 422 with a JSON `detail` and do not create a session.

In the UI, type a message and press Send or Enter. The box is cleared. The assistant bubble carries `data-testid` values `assistant-message`, `meta-source`, `meta-model`, `meta-latency`, `meta-input-tokens`, and `meta-output-tokens`. A 422 or 503 is written into the transcript as a readable error. Reload the page to start a new session id. The optional system prompt is sent when the field is non-empty and is then kept for that session.

A short recording can walk through the deployed UI, `GET /health`, one live answer, a second session that shows `LLM_CACHE`, a follow-up that remembers the first turn, `GET /sessions/{id}/messages`, then `X-Simulate-Failure: key_1=429` and `*=429`.

## 5. Setup Used: your editor and AI tools

Cursor was the editor. Grok 4.7 in Cursor wrote and revised the application, inspected the installed LiteLLM 1.111 cache API, and ran the backend tests. No other AI service is called by the app itself. Model calls go through LiteLLM to the providers named in `LLM_MODEL` or `LLM_MODELS`.

## 6. Bonus features (if any): what you added beyond the requirements and why

- **System prompt field.** The optional `system_prompt` on `POST /chat` overrides the default for that session and is stored in SQLite, so later turns keep it without the client resending the whole transcript. It is part of the cache key, so a different instruction cannot reuse another prompt's reply.
- **New chat control.** The page already starts a new session on load. The button mints another id without a reload so the system-prompt draft can be reused on a clean history.
- **Backend status line.** The UI calls `GET /health` and shows how many keys are configured, whether the cache is on, or that the backend cannot be reached.
- **Key and fallback line.** Each assistant bubble also shows `key_id` and `fallback_attempts`. The required metadata test ids are unchanged. This makes the failover demo visible without opening the network panel.
- **Per-session turn lock.** Two sends in the same session wait their turn so history cannot fork while a provider call is still running.
- **JSON error shield.** Unexpected exceptions become `{"detail":"internal error"}` instead of an HTML traceback. Provider failures stay on the 503 path above.
- **Deploy descriptors.** `render.yaml` and `frontend/vercel.json` match the Docker layout and the runtime `API_BASE_URL` / build-time `VITE_API_BASE_URL` split.
