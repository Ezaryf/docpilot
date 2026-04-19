"use client";

import { useEffect, useState } from "react";
import { motion } from "framer-motion";
import {
  Settings as SettingsIcon,
  Key,
  Database,
  Cpu,
  Save,
  CheckCircle2,
  Server,
  Cloud,
} from "lucide-react";
import { Sidebar } from "@/components/sidebar";
import {
  GROQ_MODELS,
  type AppSettings,
  getProviderLabel,
  maskApiKey,
  readStoredSettings,
  saveStoredSettings,
} from "@/lib/settings";

export default function SettingsPage() {
  const [settings, setSettings] = useState<AppSettings>(() => readStoredSettings());
  const [saved, setSaved] = useState(false);

  useEffect(() => {
    setSettings(readStoredSettings());
  }, []);

  const handleSave = () => {
    saveStoredSettings(settings);
    setSaved(true);
    setTimeout(() => setSaved(false), 2000);
  };

  const activeCredential =
    settings.provider === "groq"
      ? maskApiKey(settings.groq.apiKey)
      : settings.openaiCompatible.apiKey
      ? maskApiKey(settings.openaiCompatible.apiKey)
      : "no API key required";

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
            className="rounded-xl border border-accent/20 bg-accent/5 px-5 py-4"
          >
            <p className="text-sm font-semibold text-text-primary">
              Active provider: {getProviderLabel(settings.provider)}
            </p>
            <p className="text-xs text-text-secondary mt-1">
              Chat and evaluation will use {settings.model} with {activeCredential}.
            </p>
          </motion.div>

          <motion.div
            initial={{ opacity: 0, y: 12 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.04 }}
            className="rounded-xl border border-border bg-surface p-6"
          >
            <div className="flex items-center gap-2 mb-5">
              <Cpu className="w-4 h-4 text-accent" />
              <h2 className="text-sm font-semibold text-text-primary">AI Provider</h2>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
              {[
                {
                  id: "groq" as const,
                  title: "Groq Cloud",
                  description: "Fast hosted inference. Best when you have Groq quota available.",
                  icon: Cloud,
                },
                {
                  id: "openai-compatible" as const,
                  title: "Local vLLM / OpenAI-compatible",
                  description: "Use a local server or any API that exposes /v1 chat completions.",
                  icon: Server,
                },
              ].map((provider) => {
                const Icon = provider.icon;
                const active = settings.provider === provider.id;
                return (
                  <button
                    key={provider.id}
                    type="button"
                    onClick={() =>
                      setSettings((current) => ({
                        ...current,
                        provider: provider.id,
                        model:
                          provider.id === "groq" && current.model.startsWith("google/")
                            ? GROQ_MODELS[0].value
                            : current.model,
                      }))
                    }
                    className={`text-left rounded-xl border p-4 transition-colors ${
                      active
                        ? "border-accent/50 bg-accent/10"
                        : "border-border bg-surface-2/60 hover:bg-surface-2"
                    }`}
                  >
                    <div className="flex items-center gap-2 mb-2">
                      <Icon className={active ? "w-4 h-4 text-accent" : "w-4 h-4 text-text-tertiary"} />
                      <span className="text-sm font-semibold text-text-primary">{provider.title}</span>
                    </div>
                    <p className="text-xs text-text-secondary leading-relaxed">{provider.description}</p>
                  </button>
                );
              })}
            </div>
          </motion.div>

          {settings.provider === "groq" ? (
            <motion.div
              initial={{ opacity: 0, y: 12 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.08 }}
              className="rounded-xl border border-border bg-surface p-6"
            >
              <div className="flex items-center gap-2 mb-5">
                <Key className="w-4 h-4 text-accent" />
                <h2 className="text-sm font-semibold text-text-primary">Groq Cloud</h2>
              </div>

              <div className="space-y-4">
                <div>
                  <label className="block text-xs font-medium text-text-secondary mb-1.5">
                    Groq API key
                  </label>
                  <input
                    type="password"
                    value={settings.groq.apiKey}
                    onChange={(e) =>
                      setSettings((current) => ({
                        ...current,
                        groq: { ...current.groq, apiKey: e.target.value },
                      }))
                    }
                    placeholder="gsk_..."
                    className="w-full px-3 py-2 rounded-lg bg-surface-2 border border-border text-sm text-text-primary
                      placeholder-text-tertiary focus:outline-none focus:border-accent/40 focus:ring-1 focus:ring-accent/20
                      transition-all duration-200"
                  />
                  <p className="text-[11px] text-text-tertiary mt-2">
                    If left blank, the backend .env Groq key will be used.
                  </p>
                </div>

                <div>
                  <label className="block text-xs font-medium text-text-secondary mb-1.5">
                    Groq model
                  </label>
                  <select
                    value={settings.model}
                    onChange={(e) => setSettings((current) => ({ ...current, model: e.target.value }))}
                    className="w-full px-3 py-2 rounded-lg bg-surface-2 border border-border text-sm text-text-primary
                      focus:outline-none focus:border-accent/40 focus:ring-1 focus:ring-accent/20
                      transition-all duration-200"
                  >
                    {GROQ_MODELS.map((model) => (
                      <option key={model.value} value={model.value}>
                        {model.label}
                      </option>
                    ))}
                  </select>
                  <p className="text-[11px] text-text-tertiary mt-2">
                    Use 8B Instant when the 70B model hits daily token limits.
                  </p>
                </div>
              </div>
            </motion.div>
          ) : (
            <motion.div
              initial={{ opacity: 0, y: 12 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.08 }}
              className="rounded-xl border border-border bg-surface p-6"
            >
              <div className="flex items-center gap-2 mb-5">
                <Server className="w-4 h-4 text-accent" />
                <h2 className="text-sm font-semibold text-text-primary">
                  Local vLLM / OpenAI-compatible
                </h2>
              </div>

              <div className="space-y-4">
                <div>
                  <label className="block text-xs font-medium text-text-secondary mb-1.5">
                    Base URL
                  </label>
                  <input
                    type="url"
                    value={settings.openaiCompatible.baseUrl}
                    onChange={(e) =>
                      setSettings((current) => ({
                        ...current,
                        openaiCompatible: {
                          ...current.openaiCompatible,
                          baseUrl: e.target.value,
                        },
                      }))
                    }
                    placeholder="http://localhost:8001/v1"
                    className="w-full px-3 py-2 rounded-lg bg-surface-2 border border-border text-sm text-text-primary
                      placeholder-text-tertiary focus:outline-none focus:border-accent/40 focus:ring-1 focus:ring-accent/20
                      transition-all duration-200"
                  />
                  <p className="text-[11px] text-text-tertiary mt-2">
                    Docker users may need http://host.docker.internal:8001/v1.
                  </p>
                </div>

                <div>
                  <label className="block text-xs font-medium text-text-secondary mb-1.5">
                    Model name
                  </label>
                  <input
                    type="text"
                    value={settings.model}
                    onChange={(e) => setSettings((current) => ({ ...current, model: e.target.value }))}
                    placeholder="google/gemma-..."
                    className="w-full px-3 py-2 rounded-lg bg-surface-2 border border-border text-sm text-text-primary
                      placeholder-text-tertiary focus:outline-none focus:border-accent/40 focus:ring-1 focus:ring-accent/20
                      transition-all duration-200"
                  />
                  <p className="text-[11px] text-text-tertiary mt-2">
                    Enter the exact model ID served by vLLM, for example a Gemma model name.
                  </p>
                </div>

                <div>
                  <label className="block text-xs font-medium text-text-secondary mb-1.5">
                    API key (optional)
                  </label>
                  <input
                    type="password"
                    value={settings.openaiCompatible.apiKey}
                    onChange={(e) =>
                      setSettings((current) => ({
                        ...current,
                        openaiCompatible: {
                          ...current.openaiCompatible,
                          apiKey: e.target.value,
                        },
                      }))
                    }
                    placeholder="Leave blank for local vLLM"
                    className="w-full px-3 py-2 rounded-lg bg-surface-2 border border-border text-sm text-text-primary
                      placeholder-text-tertiary focus:outline-none focus:border-accent/40 focus:ring-1 focus:ring-accent/20
                      transition-all duration-200"
                  />
                </div>
              </div>
            </motion.div>
          )}

          <motion.div
            initial={{ opacity: 0, y: 12 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.12 }}
            className="rounded-xl border border-border bg-surface p-6"
          >
            <div className="flex items-center gap-2 mb-5">
              <Database className="w-4 h-4 text-accent" />
              <h2 className="text-sm font-semibold text-text-primary">Backend Notes</h2>
            </div>

            <p className="text-sm text-text-secondary leading-relaxed">
              Qdrant connection settings still come from the backend environment. This page now controls the chat and evaluation LLM provider only.
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
