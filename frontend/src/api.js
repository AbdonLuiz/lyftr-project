export function apiBaseUrl() {
  const runtime = window.__APP_CONFIG__ && window.__APP_CONFIG__.apiBaseUrl;
  if (typeof runtime === "string" && runtime && !runtime.includes("__API_BASE_URL__")) {
    return runtime.replace(/\/$/, "");
  }
  const built = import.meta.env.VITE_API_BASE_URL;
  if (built) {
    return String(built).replace(/\/$/, "");
  }
  return "http://localhost:8000";
}

export function formatDetail(detail) {
  if (typeof detail === "string" && detail.trim()) {
    return detail;
  }
  if (Array.isArray(detail)) {
    const text = detail
      .map((item) => {
        const message = item && (item.msg || item.message);
        if (!message) {
          return "Invalid request";
        }
        const loc = Array.isArray(item.loc) ? item.loc.filter((part) => part !== "body").join(".") : "";
        return loc ? `${loc}: ${message}` : message;
      })
      .join(" ");
    return text || "Invalid request";
  }
  return "The server could not complete that request.";
}

export async function getHealth() {
  const response = await fetch(`${apiBaseUrl()}/health`);
  if (!response.ok) {
    throw new Error("Backend health check failed");
  }
  return response.json();
}

export async function sendMessage({ sessionId, message, systemPrompt }) {
  let response;
  try {
    const payload = {
      session_id: sessionId,
      message,
    };
    if (systemPrompt && systemPrompt.trim()) {
      payload.system_prompt = systemPrompt.trim();
    }
    response = await fetch(`${apiBaseUrl()}/chat`, {
      method: "POST",
      headers: {
        Accept: "application/json",
        "Content-Type": "application/json",
      },
      body: JSON.stringify(payload),
    });
  } catch {
    throw new Error("Could not reach the server. Check your connection and try again.");
  }

  let body = null;
  try {
    body = await response.json();
  } catch {
    body = null;
  }

  if (!response.ok) {
    throw new Error(formatDetail(body && body.detail));
  }
  if (!body || typeof body.reply !== "string" || !body.reply.trim() || !body.metadata) {
    throw new Error("The server returned an incomplete reply.");
  }
  return body;
}
