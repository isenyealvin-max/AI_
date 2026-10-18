"use strict";

/*
  Learnover AI - chat function (runs on Netlify, never in the browser)

  URL on your site:  /.netlify/functions/chat
  Secret key:        Netlify environment variable GEMINI_API_KEY
  Optional:          GEMINI_MODEL  (force one model name, e.g. gemini-2.5-flash)

  POST  { mode, message, history, options }
        mode = chat | quiz | flashcards | revise | research | solve | write   (default: chat)
        -> 200 { reply: "..." }  or  { quiz: {...} }  or  { flashcards: {...} }
        -> or { error: { code, message } }

  GET   (no parameters) -> tells you if the function is alive and if the key exists.
  GET   ?test=1         -> also makes ONE tiny real request to Gemini to prove the key works.
*/

const API_BASE = "https://generativelanguage.googleapis.com/v1beta";
const PREFERRED_MODELS = ["gemini-2.5-flash", "gemini-flash-latest", "gemini-2.0-flash"];
const TOTAL_BUDGET_MS = 9000; // Netlify's default function limit is about 10 seconds
const MAX_MESSAGE_CHARS = 2500;
const MAX_HISTORY_ITEMS = 10;
const MAX_HISTORY_CHARS = 4000;
const MAX_ATTACH_CHARS = 5000000; // total size of attached files (base64 text)
const OK_MIME = /^(image\/(jpeg|png|webp|heic|heif)|application\/pdf|text\/(plain|csv|markdown))$/;

let workingModel = null; // remembered between requests while the function stays warm

const SYSTEM_PROMPT = [
  "You are Learnover AI, a friendly study helper for school students, especially in Kenya",
  "(Grade 7 and above, the CBC curriculum, and exam revision).",
  "Explain simply and step by step. Use short paragraphs, small lists, and one or two worked examples.",
  "Use everyday Kenyan examples when they help (shillings, matatus, local foods, farms).",
  "Keep answers under about 250 words unless the student asks for more.",
  "End with one short question or practice task so the student can check their understanding.",
  "Reply in the language the student writes in (English or Kiswahili).",
  "Help students learn; do not help with cheating in an exam that is in progress.",
  "If you are not sure about something, say so and suggest checking with a teacher or textbook.",
].join(" ");

const JSON_PROMPT = "You are a careful Kenyan CBC teacher. Reply with valid JSON only. No markdown, no extra words.";

function respond(statusCode, body) {
  return {
    statusCode,
    headers: {
      "Content-Type": "application/json; charset=utf-8",
      "Cache-Control": "no-store",
    },
    body: JSON.stringify(body),
  };
}

function fail(statusCode, code, message) {
  return respond(statusCode, { error: { code, message } });
}

function cleanForLog(text, key) {
  let s = String(text || "").slice(0, 300);
  if (key) s = s.split(key).join("[hidden]");
  return s;
}

// Make a safe list of earlier turns for Gemini.
function buildContents(history, message, attachments) {
  const turns = [];
  if (Array.isArray(history)) {
    for (const item of history.slice(-MAX_HISTORY_ITEMS)) {
      if (!item || typeof item.text !== "string") continue;
      const role = item.role === "model" ? "model" : item.role === "user" ? "user" : null;
      const text = item.text.trim().slice(0, MAX_HISTORY_CHARS);
      if (!role || !text) continue;
      turns.push({ role, parts: [{ text }] });
    }
  }
  while (turns.length && turns[0].role !== "user") turns.shift(); // Gemini wants "user" first
  const parts = [{ text: message }];
  let total = 0;
  if (Array.isArray(attachments)) {
    for (const a of attachments.slice(0, 3)) {
      if (!a || typeof a.data !== "string" || !OK_MIME.test(a.mime || "")) continue;
      total += a.data.length;
      if (total > MAX_ATTACH_CHARS) break;
      parts.push({ inlineData: { mimeType: a.mime, data: a.data } });
    }
  }
  turns.push({ role: "user", parts });
  return turns;
}

async function callGemini(key, model, contents, deadline, opts) {
  const remaining = deadline - Date.now();
  if (remaining < 500) {
    const e = new Error("timeout");
    e.kind = "TIMEOUT";
    throw e;
  }
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), remaining);
  try {
    const res = await fetch(`${API_BASE}/models/${encodeURIComponent(model)}:generateContent`, {
      method: "POST",
      headers: { "Content-Type": "application/json", "x-goog-api-key": key },
      body: JSON.stringify({
        systemInstruction: { parts: [{ text: opts.json ? JSON_PROMPT : SYSTEM_PROMPT }] },
        contents,
        ...(opts.search ? { tools: [{ google_search: {} }] } : {}),
        generationConfig: Object.assign({ temperature: opts.json ? 0.4 : 0.6, maxOutputTokens: opts.json ? 4096 : 2048 }, opts.json ? { responseMimeType: "application/json" } : {}),
      }),
      signal: controller.signal,
    });
    let data = null;
    try {
      data = await res.json();
    } catch (_) {
      data = null;
    }
    return { status: res.status, data };
  } catch (err) {
    const e = new Error(err && err.name === "AbortError" ? "timeout" : "network");
    e.kind = err && err.name === "AbortError" ? "TIMEOUT" : "NETWORK";
    throw e;
  } finally {
    clearTimeout(timer);
  }
}

// Ask Gemini which models this key can really use (only if the usual names fail).
async function discoverModel(key, deadline) {
  const remaining = deadline - Date.now();
  if (remaining < 500) return [];
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), remaining);
  try {
    const res = await fetch(`${API_BASE}/models?pageSize=200`, {
      headers: { "x-goog-api-key": key },
      signal: controller.signal,
    });
    if (!res.ok) return [];
    const data = await res.json();
    const names = (data.models || [])
      .filter((m) => Array.isArray(m.supportedGenerationMethods) && m.supportedGenerationMethods.includes("generateContent"))
      .map((m) => String(m.name || "").replace(/^models\//, ""))
      .filter((n) => /^gemini-.*flash/.test(n) && !/(lite|image|tts|live|audio|embed|thinking|exp|vision)/.test(n));
    names.sort((a, b) => (a < b ? 1 : -1)); // newest-looking names first
    return names.slice(0, 2);
  } catch (_) {
    return [];
  } finally {
    clearTimeout(timer);
  }
}

function readReply(data) {
  const cand = data && data.candidates && data.candidates[0];
  const parts = cand && cand.content && cand.content.parts;
  const text = Array.isArray(parts)
    ? parts.map((p) => (p && typeof p.text === "string" ? p.text : "")).join("").trim()
    : "";
  const chunks = (cand && cand.groundingMetadata && cand.groundingMetadata.groundingChunks) || [];
  const sources = chunks.map((c) => c && c.web && { title: String(c.web.title || c.web.uri || "").slice(0, 120), url: String(c.web.uri || "") }).filter((x) => x && /^https?:\/\//.test(x.url)).slice(0, 6);
  return { text, sources, finishReason: cand && cand.finishReason, blockReason: data && data.promptFeedback && data.promptFeedback.blockReason };
}

// Shared logic: try models until one answers. Returns { reply, model } or { error }.
async function generate(key, message, history, attachments, opts) {
  opts = opts || {};
  const deadline = Date.now() + TOTAL_BUDGET_MS;
  const contents = buildContents(history, message, attachments);

  let candidates = [];
  if (process.env.GEMINI_MODEL) candidates.push(process.env.GEMINI_MODEL.trim());
  if (workingModel) candidates.unshift(workingModel);
  for (const m of PREFERRED_MODELS) if (!candidates.includes(m)) candidates.push(m);
  candidates = candidates.slice(0, 3);

  let triedDiscovery = false;
  let lastProblem = null;

  for (let i = 0; i < candidates.length; i++) {
    const model = candidates[i];
    let result;
    try {
      result = await callGemini(key, model, contents, deadline, opts);
    } catch (err) {
      if (err.kind === "TIMEOUT") return { error: [504, "TIMEOUT", "The AI took too long to answer. Please try a shorter question."] };
      return { error: [502, "NETWORK", "Learnover AI could not reach the AI service. Please try again in a moment."] };
    }

    const { status, data } = result;
    const apiMessage = data && data.error && data.error.message;

    if (status === 200) {
      const out = readReply(data);
      if (out.text) {
        workingModel = model;
        return { reply: out.text, model, sources: out.sources };
      }
      if (out.blockReason || out.finishReason === "SAFETY") {
        return { error: [422, "BLOCKED", "The AI could not answer that question. Try asking it in a different way."] };
      }
      console.error("Unexpected Gemini response shape", cleanForLog(JSON.stringify(data), key));
      return { error: [502, "BAD_RESPONSE", "The AI sent an answer that Learnover AI could not read. Please try again."] };
    }

    console.error("Gemini error", model, status, cleanForLog(apiMessage, key));

    if (status === 400 && /api key|API_KEY/i.test(apiMessage || "")) {
      return { error: [502, "BAD_KEY", "The AI key on the server is not accepted by Google. The site owner should create a new key and save it in Netlify."] };
    }
    if (status === 401 || status === 403) {
      return { error: [502, "KEY_NOT_ALLOWED", "Google did not allow this request. The site owner should check that the Gemini API key is active."] };
    }

    // Problems that may be fixed by trying another model: not found, quota on this model, busy.
    if (status === 404 || status === 429 || status === 503 || status === 500) {
      lastProblem = status;
      if (status === 404 && workingModel === model) workingModel = null;
      if (status === 404 && i === candidates.length - 1 && !triedDiscovery) {
        triedDiscovery = true;
        const found = await discoverModel(key, deadline);
        for (const f of found) if (!candidates.includes(f)) candidates.push(f);
      }
      continue;
    }

    return { error: [502, "UPSTREAM", "The AI service returned an error. Please try again in a moment."] };
  }

  if (lastProblem === 429) {
    return { error: [429, "QUOTA", "Learnover AI has reached its free AI limit for now. Please wait a minute and try again, or come back later today."] };
  }
  if (lastProblem === 404) {
    return { error: [503, "NO_MODEL", "No AI model is available for this key yet. The site owner should check the Gemini API key in Google AI Studio."] };
  }
  return { error: [503, "BUSY", "The AI service is busy. Please try again in a moment."] };
}

exports.handler = async (event) => {
  const key = process.env.GEMINI_API_KEY;
  const method = event.httpMethod;

  // Health check, safe to open in a browser. Never shows the key.
  if (method === "GET") {
    const info = { ok: true, function: "chat", keyPresent: Boolean(key), model: workingModel };
    const wantTest = event.queryStringParameters && event.queryStringParameters.test;
    if (wantTest) {
      if (!key) return respond(200, Object.assign(info, { ok: false, test: "GEMINI_API_KEY is missing in Netlify" }));
      const r = await generate(key, "Reply with the single word: OK", []);
      if (r.reply) Object.assign(info, { test: "passed", model: r.model });
      else Object.assign(info, { ok: false, test: "failed", reason: r.error[1], detail: r.error[2] });
    }
    return respond(200, info);
  }

  if (method !== "POST") return fail(405, "METHOD", "Use POST to send a message.");

  if (!key) {
    console.error("GEMINI_API_KEY is not set for this function.");
    return fail(500, "MISSING_KEY", "The AI is not connected yet. The site owner must add GEMINI_API_KEY in Netlify and deploy again.");
  }

  let body;
  try {
    body = JSON.parse(event.body || "{}");
  } catch (_) {
    return fail(400, "BAD_REQUEST", "That message could not be read. Please try again.");
  }

  const message = typeof body.message === "string" ? body.message.trim() : "";
  if (!message) return fail(400, "EMPTY_MESSAGE", "Type a question first.");
  if (message.length > MAX_MESSAGE_CHARS) {
    return fail(400, "TOO_LONG", `Please keep your question under ${MAX_MESSAGE_CHARS} characters.`);
  }

  try {
    const mode = body.mode === "research" || body.mode === "json" ? body.mode : "chat";
    const r = await generate(key, message, body.history, body.attachments, { search: mode === "research", json: mode === "json" });
    if (r.reply) return respond(200, { reply: r.reply, sources: r.sources || [] });
    return fail(r.error[0], r.error[1], r.error[2]);
  } catch (err) {
    console.error("Unexpected function error", cleanForLog(err && err.message, key));
    return fail(500, "SERVER_ERROR", "Something went wrong on the server. Please try again.");
  }
};

// Exposed only so the project's tests can reset state. Netlify ignores this.
exports._resetForTests = () => {
  workingModel = null;
};
