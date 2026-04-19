"use client";

export type LlmProvider = "groq" | "openai-compatible";

export interface StoredGroqKey {
  id: string;
  label: string;
  apiKey: string;
}

export interface LlmRequestConfig {
  provider: LlmProvider;
  model: string;
  groqApiKey: string | null;
  openaiBaseUrl: string | null;
  openaiApiKey: string | null;
}

export interface AppSettings {
  version: 2;
  provider: LlmProvider;
  model: string;
  groq: {
    apiKey: string;
  };
  openaiCompatible: {
    baseUrl: string;
    apiKey: string;
  };
  qdrantUrl: string;
  qdrantKey: string;
}

const SETTINGS_KEY = "docpilot-settings";

export const GROQ_MODELS = [
  { value: "llama-3.1-8b-instant", label: "LLaMA 3.1 8B (Instant)" },
  { value: "llama-3.3-70b-versatile", label: "LLaMA 3.3 70B (Versatile)" },
  { value: "gemma2-9b-it", label: "Gemma 2 9B" },
  { value: "mixtral-8x7b-32768", label: "Mixtral 8x7B" },
];

export const DEFAULT_SETTINGS: AppSettings = {
  version: 2,
  provider: "groq",
  model: "llama-3.1-8b-instant",
  groq: {
    apiKey: "",
  },
  openaiCompatible: {
    baseUrl: "http://localhost:8001/v1",
    apiKey: "",
  },
  qdrantUrl: "",
  qdrantKey: "",
};

function isProvider(value: unknown): value is LlmProvider {
  return value === "groq" || value === "openai-compatible";
}

function readString(value: unknown, fallback = "") {
  return typeof value === "string" ? value : fallback;
}

function migrateLegacyGroqKey(parsed: any) {
  if (typeof parsed?.groqKey === "string" && parsed.groqKey.trim()) {
    return parsed.groqKey.trim();
  }

  const groqKeys = Array.isArray(parsed?.groqKeys) ? parsed.groqKeys : [];
  const activeKey =
    typeof parsed?.activeGroqKeyId === "string"
      ? groqKeys.find((entry: any) => entry?.id === parsed.activeGroqKeyId)
      : groqKeys[0];

  return typeof activeKey?.apiKey === "string" ? activeKey.apiKey.trim() : "";
}

export function normalizeSettings(parsed: unknown): AppSettings {
  if (!parsed || typeof parsed !== "object") return DEFAULT_SETTINGS;

  const raw = parsed as any;
  const provider = isProvider(raw.provider) ? raw.provider : DEFAULT_SETTINGS.provider;
  const legacyGroqKey = migrateLegacyGroqKey(raw);
  const model = readString(raw.model, readString(raw.llmModel, DEFAULT_SETTINGS.model)).trim();

  return {
    version: 2,
    provider,
    model: model || DEFAULT_SETTINGS.model,
    groq: {
      apiKey: readString(raw.groq?.apiKey, legacyGroqKey).trim(),
    },
    openaiCompatible: {
      baseUrl: readString(
        raw.openaiCompatible?.baseUrl,
        DEFAULT_SETTINGS.openaiCompatible.baseUrl
      ).trim(),
      apiKey: readString(raw.openaiCompatible?.apiKey).trim(),
    },
    qdrantUrl: readString(raw.qdrantUrl).trim(),
    qdrantKey: readString(raw.qdrantKey).trim(),
  };
}

export function readStoredSettings(): AppSettings {
  if (typeof window === "undefined") return DEFAULT_SETTINGS;

  try {
    const raw = localStorage.getItem(SETTINGS_KEY);
    if (!raw) return DEFAULT_SETTINGS;
    return normalizeSettings(JSON.parse(raw));
  } catch {
    return DEFAULT_SETTINGS;
  }
}

export function saveStoredSettings(settings: AppSettings) {
  localStorage.setItem(SETTINGS_KEY, JSON.stringify(normalizeSettings(settings)));
}

export function getLlmRequestConfig(settings: AppSettings): LlmRequestConfig {
  return {
    provider: settings.provider,
    model: settings.model,
    groqApiKey: settings.provider === "groq" && settings.groq.apiKey ? settings.groq.apiKey : null,
    openaiBaseUrl:
      settings.provider === "openai-compatible" && settings.openaiCompatible.baseUrl
        ? settings.openaiCompatible.baseUrl
        : null,
    openaiApiKey:
      settings.provider === "openai-compatible" && settings.openaiCompatible.apiKey
        ? settings.openaiCompatible.apiKey
        : null,
  };
}

export function getProviderLabel(provider: LlmProvider) {
  return provider === "groq" ? "Groq Cloud" : "Local vLLM / OpenAI-compatible";
}

export function maskApiKey(apiKey: string) {
  if (!apiKey) return "not set";
  if (apiKey.length <= 8) return "••••••••";
  return `${apiKey.slice(0, 4)}••••${apiKey.slice(-4)}`;
}
