"use client";

import { useState, useEffect } from "react";
import { motion } from "framer-motion";
import {
  Settings as SettingsIcon,
  Key,
  Database,
  Cpu,
  Save,
  CheckCircle2,
  Plus,
  Trash2,
} from "lucide-react";
import { Sidebar } from "@/components/sidebar";
import {
  type AppSettings,
  getActiveGroqKey,
  maskApiKey,
  readStoredSettings,
  saveStoredSettings,
} from "@/lib/settings";

export default function SettingsPage() {
  const [settings, setSettings] = useState<AppSettings>(() => readStoredSettings());
  const [draftGroqLabel, setDraftGroqLabel] = useState("");
  const [draftGroqKey, setDraftGroqKey] = useState("");
  const [saved, setSaved] = useState(false);

  useEffect(() => {
    setSettings(readStoredSettings());
  }, []);

  const activeGroqKey = getActiveGroqKey(settings);

  const handleSave = () => {
    saveStoredSettings(settings);
    setSaved(true);
    setTimeout(() => setSaved(false), 2000);
  };

  const handleAddGroqKey = () => {
    if (!draftGroqKey.trim()) return;

    const nextKey = {
      id: crypto.randomUUID(),
      label: draftGroqLabel.trim() || `Groq Key ${settings.groqKeys.length + 1}`,
      apiKey: draftGroqKey.trim(),
    };

    setSettings((current) => ({
      ...current,
      groqKeys: [...current.groqKeys, nextKey],
      activeGroqKeyId: current.activeGroqKeyId ?? nextKey.id,
    }));
    setDraftGroqLabel("");
    setDraftGroqKey("");
  };

  return (
    <div className="flex h-screen overflow-hidden bg-background">
      <Sidebar />

      <div className="flex-1 overflow-y-auto">
        <header className="h-14 flex items-center px-6 border-b border-border bg-surface/80 backdrop-blur-sm sticky top-0 z-10">
          <SettingsIcon className="w-4 h-4 text-accent mr-2" />
          <h1 className="text-sm font-semibold text-text-primary">Settings</h1>
        </header>

        <div className="max-w-3xl mx-auto p-6 space-y-6">
          <motion.div
            initial={{ opacity: 0, y: 12 }}
            animate={{ opacity: 1, y: 0 }}
            className="rounded-xl border border-border bg-surface p-6"
          >
            <div className="flex items-center gap-2 mb-5">
              <Key className="w-4 h-4 text-accent" />
              <h2 className="text-sm font-semibold text-text-primary">Groq API Keys</h2>
            </div>

            <div className="rounded-xl border border-accent/20 bg-accent/5 px-4 py-3 mb-4">
              <p className="text-sm font-medium text-text-primary">
                Active key: {activeGroqKey?.label || "Backend default"}
              </p>
              <p className="text-xs text-text-secondary mt-1">
                {activeGroqKey
                  ? `Chat will use ${maskApiKey(activeGroqKey.apiKey)} with model ${settings.llmModel}.`
                  : "If no key is selected here, DocPilot falls back to the backend .env Groq key."}
              </p>
            </div>

            <div className="space-y-3">
              {settings.groqKeys.map((entry) => {
                const isActive = entry.id === settings.activeGroqKeyId;
                return (
                  <div
                    key={entry.id}
                    className={`rounded-xl border px-4 py-3 ${
                      isActive ? "border-accent/40 bg-accent/5" : "border-border bg-surface-2/60"
                    }`}
                  >
                    <div className="flex items-center justify-between gap-4">
                      <div className="min-w-0">
                        <p className="text-sm font-medium text-text-primary truncate">{entry.label}</p>
                        <p className="text-xs text-text-tertiary mt-1">{maskApiKey(entry.apiKey)}</p>
                      </div>
                      <div className="flex items-center gap-2">
                        <button
                          type="button"
                          onClick={() =>
                            setSettings((current) => ({ ...current, activeGroqKeyId: entry.id }))
                          }
                          className={`px-3 py-1.5 rounded-lg text-xs font-medium transition-colors ${
                            isActive
                              ? "bg-accent text-white"
                              : "border border-border text-text-secondary hover:text-text-primary hover:bg-surface"
                          }`}
                        >
                          {isActive ? "Active" : "Use this key"}
                        </button>
                        <button
                          type="button"
                          onClick={() =>
                            setSettings((current) => {
                              const nextGroqKeys = current.groqKeys.filter((item) => item.id !== entry.id);
                              return {
                                ...current,
                                groqKeys: nextGroqKeys,
                                activeGroqKeyId:
                                  current.activeGroqKeyId === entry.id ? nextGroqKeys[0]?.id ?? null : current.activeGroqKeyId,
                              };
                            })
                          }
                          className="p-2 rounded-lg text-text-tertiary hover:text-error hover:bg-surface transition-colors"
                          aria-label={`Delete ${entry.label}`}
                        >
                          <Trash2 className="w-4 h-4" />
                        </button>
                      </div>
                    </div>
                  </div>
                );
              })}

              <div className="rounded-xl border border-dashed border-border px-4 py-4 space-y-3">
                <p className="text-sm font-medium text-text-primary">Add another Groq key</p>
                <input
                  type="text"
                  value={draftGroqLabel}
                  onChange={(e) => setDraftGroqLabel(e.target.value)}
                  placeholder="Label, e.g. Personal Groq"
                  className="w-full px-3 py-2 rounded-lg bg-surface-2 border border-border text-sm text-text-primary
                    placeholder-text-tertiary focus:outline-none focus:border-accent/40 focus:ring-1 focus:ring-accent/20
                    transition-all duration-200"
                />
                <input
                  type="password"
                  value={draftGroqKey}
                  onChange={(e) => setDraftGroqKey(e.target.value)}
                  placeholder="gsk_..."
                  className="w-full px-3 py-2 rounded-lg bg-surface-2 border border-border text-sm text-text-primary
                    placeholder-text-tertiary focus:outline-none focus:border-accent/40 focus:ring-1 focus:ring-accent/20
                    transition-all duration-200"
                />
                <button
                  type="button"
                  onClick={handleAddGroqKey}
                  className="inline-flex items-center gap-2 px-4 py-2 rounded-lg text-sm font-medium border border-border text-text-secondary hover:text-text-primary hover:bg-surface-2 transition-colors"
                >
                  <Plus className="w-4 h-4" />
                  Add key
                </button>
              </div>
            </div>
          </motion.div>

          <motion.div
            initial={{ opacity: 0, y: 12 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.08 }}
            className="rounded-xl border border-border bg-surface p-6"
          >
            <div className="flex items-center gap-2 mb-5">
              <Cpu className="w-4 h-4 text-accent" />
              <h2 className="text-sm font-semibold text-text-primary">Model Configuration</h2>
            </div>

            <div>
              <label className="block text-xs font-medium text-text-secondary mb-1.5">
                Groq model
              </label>
              <select
                value={settings.llmModel}
                onChange={(e) => setSettings((current) => ({ ...current, llmModel: e.target.value }))}
                className="w-full px-3 py-2 rounded-lg bg-surface-2 border border-border text-sm text-text-primary
                  focus:outline-none focus:border-accent/40 focus:ring-1 focus:ring-accent/20
                  transition-all duration-200"
              >
                <option value="llama-3.3-70b-versatile">LLaMA 3.3 70B (Versatile)</option>
                <option value="llama-3.1-8b-instant">LLaMA 3.1 8B (Instant)</option>
                <option value="mixtral-8x7b-32768">Mixtral 8x7B</option>
                <option value="gemma2-9b-it">Gemma 2 9B</option>
              </select>
              <p className="text-[11px] text-text-tertiary mt-2">
                Lower-cost models can help when a stronger model hits daily token limits.
              </p>
            </div>
          </motion.div>

          <motion.div
            initial={{ opacity: 0, y: 12 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.16 }}
            className="rounded-xl border border-border bg-surface p-6"
          >
            <div className="flex items-center gap-2 mb-5">
              <Database className="w-4 h-4 text-accent" />
              <h2 className="text-sm font-semibold text-text-primary">Backend Notes</h2>
            </div>

            <p className="text-sm text-text-secondary leading-relaxed">
              Qdrant fields here are still informational only. The chat flow now uses the active Groq key and model from this page, but Qdrant connection settings still come from the backend environment.
            </p>
          </motion.div>

          <div className="flex justify-end">
            <button
              onClick={handleSave}
              className="flex items-center gap-2 px-5 py-2.5 rounded-lg text-sm font-medium
                bg-accent text-white hover:bg-accent/90 transition-all duration-200"
            >
              {saved ? (
                <>
                  <CheckCircle2 className="w-4 h-4" />
                  Saved!
                </>
              ) : (
                <>
                  <Save className="w-4 h-4" />
                  Save Settings
                </>
              )}
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}
