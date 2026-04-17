"use client";

export interface StoredGroqKey {
  id: string;
  label: string;
  apiKey: string;
}

export interface AppSettings {
  groqKeys: StoredGroqKey[];
  activeGroqKeyId: string | null;
  qdrantUrl: string;
  qdrantKey: string;
  llmModel: string;
}

const SETTINGS_KEY = "docpilot-settings";

const DEFAULT_SETTINGS: AppSettings = {
  groqKeys: [],
  activeGroqKeyId: null,
  qdrantUrl: "",
  qdrantKey: "",
  llmModel: "llama-3.3-70b-versatile",
};

export function readStoredSettings(): AppSettings {
  if (typeof window === "undefined") return DEFAULT_SETTINGS;

  try {
    const raw = localStorage.getItem(SETTINGS_KEY);
    if (!raw) return DEFAULT_SETTINGS;

    const parsed = JSON.parse(raw);

    const groqKeys = Array.isArray(parsed.groqKeys)
      ? parsed.groqKeys.filter(
          (entry: unknown): entry is StoredGroqKey =>
            !!entry &&
            typeof entry === "object" &&
            typeof (entry as StoredGroqKey).id === "string" &&
            typeof (entry as StoredGroqKey).label === "string" &&
            typeof (entry as StoredGroqKey).apiKey === "string"
        )
      : [];

    if (groqKeys.length === 0 && typeof parsed.groqKey === "string" && parsed.groqKey.trim()) {
      groqKeys.push({
        id: crypto.randomUUID(),
        label: "Primary Groq Key",
        apiKey: parsed.groqKey,
      });
    }

    const activeGroqKeyId =
      typeof parsed.activeGroqKeyId === "string" &&
      groqKeys.some((entry: StoredGroqKey) => entry.id === parsed.activeGroqKeyId)
        ? parsed.activeGroqKeyId
        : groqKeys[0]?.id ?? null;

    return {
      groqKeys,
      activeGroqKeyId,
      qdrantUrl: typeof parsed.qdrantUrl === "string" ? parsed.qdrantUrl : "",
      qdrantKey: typeof parsed.qdrantKey === "string" ? parsed.qdrantKey : "",
      llmModel:
        typeof parsed.llmModel === "string" && parsed.llmModel
          ? parsed.llmModel
          : DEFAULT_SETTINGS.llmModel,
    };
  } catch {
    return DEFAULT_SETTINGS;
  }
}

export function saveStoredSettings(settings: AppSettings) {
  localStorage.setItem(SETTINGS_KEY, JSON.stringify(settings));
}

export function getActiveGroqKey(settings: AppSettings) {
  return settings.groqKeys.find((entry) => entry.id === settings.activeGroqKeyId) ?? null;
}

export function maskApiKey(apiKey: string) {
  if (apiKey.length <= 8) return "••••••••";
  return `${apiKey.slice(0, 4)}••••${apiKey.slice(-4)}`;
}
