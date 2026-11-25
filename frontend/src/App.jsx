import { useEffect, useRef, useState } from "react";
import { getHealth, sendMessage } from "./api.js";

function createSessionId() {
  const hex = crypto.randomUUID().replace(/-/g, "").slice(0, 12);
  return `s-${hex}`;
}

export default function App() {
  const [sessionId, setSessionId] = useState(createSessionId);
  const [input, setInput] = useState("");
  const [systemPrompt, setSystemPrompt] = useState("");
  const [messages, setMessages] = useState([]);
  const [sending, setSending] = useState(false);
  const [health, setHealth] = useState(null);
  const transcriptRef = useRef(null);
  const inputRef = useRef(null);

  useEffect(() => {
    let cancelled = false;
    getHealth()
      .then((data) => {
        if (!cancelled) {
          setHealth(data);
        }
      })
      .catch(() => {
        if (!cancelled) {
          setHealth({ status: "down" });
        }
      });
    return () => {
      cancelled = true;
    };
  }, []);

  useEffect(() => {
    const node = transcriptRef.current;
    if (node) {
      node.scrollTop = node.scrollHeight;
    }
  }, [messages, sending]);

  function startNewSession() {
    if (sending) {
      return;
    }
    setSessionId(createSessionId());
    setMessages([]);
    setInput("");
    inputRef.current?.focus();
  }

  async function onSubmit(event) {
    event.preventDefault();
    if (sending) {
      return;
    }
    const message = input.trim();
    setInput("");
    setSending(true);
    try {
      const data = await sendMessage({ sessionId, message, systemPrompt });
      setMessages((current) => [
        ...current,
        { id: crypto.randomUUID(), role: "user", content: message },
        { id: crypto.randomUUID(), role: "assistant", content: data.reply, metadata: data.metadata },
      ]);
    } catch (error) {
      const reason = error instanceof Error ? error.message : "Something went wrong.";
      const quoted = message ? `"${message}"` : "that message";
      setMessages((current) => [
        ...current,
        {
          id: crypto.randomUUID(),
          role: "error",
          content: `Could not send ${quoted}. ${reason}`,
        },
      ]);
    } finally {
      setSending(false);
      inputRef.current?.focus();
    }
  }

  function onInputKeyDown(event) {
    if (event.key === "Enter" && !event.shiftKey) {
      event.preventDefault();
      event.currentTarget.form?.requestSubmit();
    }
  }

  const healthText =
    health?.status === "ok"
      ? `${health.keys_configured} keys configured · cache ${health.cache_enabled ? "on" : "off"}`
      : health?.status === "down"
        ? "Backend unreachable"
        : "Checking backend…";

  return (
    <div className="page">
      <header className="topbar">
        <div>
          <p className="eyebrow">Multi-turn chat</p>
          <h1>Field desk</h1>
        </div>
        <p className={`health ${health?.status === "down" ? "health-down" : ""}`}>{healthText}</p>
      </header>

      <div className="layout">
        <aside className="side">
          <label className="field">
            <span>System prompt</span>
            <textarea
              data-testid="system-prompt-input"
              value={systemPrompt}
              onChange={(event) => setSystemPrompt(event.target.value)}
              rows={6}
              placeholder="Optional. Applies to this session on the next send."
            />
          </label>
          <div className="session-card">
            <p className="session-label">Session</p>
            <p className="session-id">{sessionId}</p>
            <button type="button" className="ghost" onClick={startNewSession} disabled={sending}>
              New chat
            </button>
          </div>
          <p className="hint">
            A new session starts on each page load. The server keeps earlier turns; this page sends only the new message.
          </p>
        </aside>

        <main className="desk">
          <div className="transcript" ref={transcriptRef} aria-live="polite">
            {messages.length === 0 && !sending ? (
              <p className="empty">Ask something. The reply shows where it came from, which model answered, and the tokens for that turn only.</p>
            ) : null}
            {messages.map((item) => {
              if (item.role === "user") {
                return (
                  <article key={item.id} className="bubble user">
                    <p>{item.content}</p>
                  </article>
                );
              }
              if (item.role === "error") {
                return (
                  <p key={item.id} className="error" role="alert">
                    {item.content}
                  </p>
                );
              }
              const meta = item.metadata;
              return (
                <article key={item.id} data-testid="assistant-message" className="bubble assistant">
                  <p className="reply">{item.content}</p>
                  <dl className="meta">
                    <div>
                      <dt>Source</dt>
                      <dd data-testid="meta-source">{meta.resolution_source}</dd>
                    </div>
                    <div>
                      <dt>Model</dt>
                      <dd data-testid="meta-model">{meta.model}</dd>
                    </div>
                    <div>
                      <dt>Latency</dt>
                      <dd data-testid="meta-latency">{meta.latency_ms} ms</dd>
                    </div>
                    <div>
                      <dt>Input tokens</dt>
                      <dd data-testid="meta-input-tokens">{meta.input_tokens}</dd>
                    </div>
                    <div>
                      <dt>Output tokens</dt>
                      <dd data-testid="meta-output-tokens">{meta.output_tokens}</dd>
                    </div>
                  </dl>
                  <p className="meta-note">
                    Key {meta.key_id ?? "none"} · fallback {meta.fallback_attempts}
                  </p>
                </article>
              );
            })}
            {sending ? <p className="pending">Waiting for a reply…</p> : null}
          </div>

          <form className="composer" onSubmit={onSubmit}>
            <label className="sr-only" htmlFor="chat-input">
              Message
            </label>
            <textarea
              id="chat-input"
              data-testid="chat-input"
              ref={inputRef}
              value={input}
              onChange={(event) => setInput(event.target.value)}
              onKeyDown={onInputKeyDown}
              rows={3}
              placeholder="Write a message"
            />
            <button data-testid="send-button" type="submit" disabled={sending}>
              {sending ? "Sending" : "Send"}
            </button>
          </form>
        </main>
      </div>
    </div>
  );
}
