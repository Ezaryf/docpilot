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
  AlertTriangle,
  Loader2,
  Server,
  Cloud,
  TerminalSquare,
} from "lucide-react";
import { Sidebar } from "@/components/sidebar";
import {
  applyLocalModel,
  checkLlmEnvironment,
  deleteLocalModelToken,
  getLocalModelStatus,
  stopLocalModel,
  testLlmConnection,
  type LlmEnvironmentDiagnostic,
  type LocalModelStatus,
} from "@/lib/api";
import {
  DEFAULT_SETTINGS,
  GROQ_MODELS,
  type AppSettings,
  getLlmRequestConfig,
  getProviderLabel,
  maskApiKey,
  normalizeOpenAiBaseUrl,
  readStoredSettings,
  saveStoredSettings,
} from "@/lib/settings";

export default function SettingsPage() {
  const [settings, setSettings] = useState<AppSettings>(DEFAULT_SETTINGS);
  const [hydrated, setHydrated] = useState(false);
  const [saved, setSaved] = useState(false);
  const [testing, setTesting] = useState(false);
  const [testResult, setTestResult] = useState<{ ok: boolean; message: string } | null>(null);
  const [checkingRuntime, setCheckingRuntime] = useState(false);
  const [runtimeResult, setRuntimeResult] = useState<LlmEnvironmentDiagnostic | null>(null);
  const [localModelStatus, setLocalModelStatus] = useState<LocalModelStatus | null>(null);
  const [localModelBusy, setLocalModelBusy] = useState(false);
  const [hfTokenInput, setHfTokenInput] = useState("");
  const [localModelError, setLocalModelError] = useState("");
  const [copiedCommand, setCopiedCommand] = useState(false);
  const [hfTokenAvailable, setHfTokenAvailable] = useState(false);
  const [gpuMemoryMode, setGpuMemoryMode] = useState<"safe" | "balanced" | "max">("safe");

  useEffect(() => {
    setSettings(readStoredSettings());
    setHydrated(true);
  }, []);

  useEffect(() => {
    if (!hydrated || settings.provider !== "openai-compatible") return;

    let active = true;
    void getLocalModelStatus()
      .then((status) => {
        if (active) setLocalModelStatus(status);
      })
      .catch(() => {
        if (active) setLocalModelStatus(null);
      });

    return () => {
      active = false;
    };
  }, [hydrated, settings.provider]);

  const handleSave = () => {
    if (!hydrated) return;
    const normalizedSettings = {
      ...settings,
      openaiCompatible: {
        ...settings.openaiCompatible,
        baseUrl: normalizeOpenAiBaseUrl(settings.openaiCompatible.baseUrl),
      },
    };
    setSettings(normalizedSettings);
    saveStoredSettings(normalizedSettings);
    setSaved(true);
    setTimeout(() => setSaved(false), 2000);
  };

  const handleTestConnection = async () => {
    if (!hydrated) return;
    setTesting(true);
    setTestResult(null);
    const normalizedSettings = {
      ...settings,
      openaiCompatible: {
        ...settings.openaiCompatible,
        baseUrl: normalizeOpenAiBaseUrl(settings.openaiCompatible.baseUrl),
      },
    };
    setSettings(normalizedSettings);

    try {
      const result = await testLlmConnection(getLlmRequestConfig(normalizedSettings));
      setTestResult({
        ok: Boolean(result.ok),
        message: result.ok ? result.message || "Connection test succeeded." : result.error || "Connection test failed.",
      });
      if (!result.ok && normalizedSettings.provider === "openai-compatible") {
        const runtime = await checkLlmEnvironment(getLlmRequestConfig(normalizedSettings));
        setRuntimeResult(runtime);
        if (!runtime.ok) {
          setTestResult({
            ok: false,
            message: `${runtime.title}: ${runtime.message} ${runtime.recommendation}`,
          });
        }
      }
    } catch (error) {
      setTestResult({
        ok: false,
        message: error instanceof Error ? error.message : "Connection test failed.",
      });
    } finally {
      setTesting(false);
    }
  };

  const handleCheckRuntime = async () => {
    if (!hydrated) return;
    setCheckingRuntime(true);
    setRuntimeResult(null);
    const normalizedSettings = {
      ...settings,
      openaiCompatible: {
        ...settings.openaiCompatible,
        baseUrl: normalizeOpenAiBaseUrl(settings.openaiCompatible.baseUrl),
      },
    };
    setSettings(normalizedSettings);

    try {
      setRuntimeResult(await checkLlmEnvironment(getLlmRequestConfig(normalizedSettings)));
    } catch (error) {
      setRuntimeResult({
        ok: false,
        status: "error",
        issue_code: "frontend_runtime_check_failed",
        title: "Runtime check failed",
        message: error instanceof Error ? error.message : "DocPilot could not check the local vLLM runtime.",
        recommendation: "Start vLLM manually, then use Test Connection.",
      });
    } finally {
      setCheckingRuntime(false);
    }
  };

  const handleCopyDockerCommand = async () => {
    const dockerCommand = buildDisplayDockerCommand(runtimeResult?.docker_command || "");
    const command = hfTokenAvailable
      ? `$env:HF_TOKEN="<your-hugging-face-read-token>"\n${dockerCommand}`
      : dockerCommand;
    if (!command) return;

    try {
      await navigator.clipboard.writeText(command);
      setCopiedCommand(true);
      setTimeout(() => setCopiedCommand(false), 2000);
    } catch {
      setCopiedCommand(false);
    }
  };

  const activeCredential =
    settings.provider === "groq"
      ? maskApiKey(settings.groq.apiKey)
      : settings.openaiCompatible.apiKey
      ? maskApiKey(settings.openaiCompatible.apiKey)
      : "no API key required";
  const resolvedOpenAiBaseUrl = normalizeOpenAiBaseUrl(settings.openaiCompatible.baseUrl);
  const usesDocPilotBackendPort =
    settings.provider === "openai-compatible" &&
    /^https?:\/\/(localhost|127\.0\.0\.1):8000\/v1\b/i.test(resolvedOpenAiBaseUrl);
  const parsedOpenAiUrl = (() => {
    try {
      return new URL(resolvedOpenAiBaseUrl);
    } catch {
      return null;
    }
  })();
  const vllmPort = parsedOpenAiUrl?.port || (parsedOpenAiUrl?.protocol === "https:" ? "443" : "80");
  const vllmModel = settings.model || "google/gemma-4-31B-it";
  const gpuMemoryModes = {
    safe: {
      label: "Safe 10GB GPU",
      description: "Best default for RTX 3080 10GB and background apps.",
      utilization: "0.82",
      maxModelLen: "4096",
    },
    balanced: {
      label: "Balanced",
      description: "A little more VRAM, still conservative.",
      utilization: "0.85",
      maxModelLen: "4096",
    },
    max: {
      label: "Max context",
      description: "Only use after closing GPU-heavy apps.",
      utilization: "0.9",
      maxModelLen: "8192",
    },
  };
  const selectedGpuMode = gpuMemoryModes[gpuMemoryMode];
  const servedModels = runtimeResult?.served_models ?? [];
  const activeServedModel = runtimeResult?.active_served_model || servedModels[0] || "";
  const canUseRunningModel =
    hydrated &&
    settings.provider === "openai-compatible" &&
    Boolean(activeServedModel) &&
    settings.model !== activeServedModel;
  const buildDisplayDockerCommand = (command: string) => {
    if (!command) return "";
    const withoutMemoryFlags = command.replace(
      /\s+--gpu-memory-utilization\s+\S+\s+--max-model-len\s+\S+/,
      ""
    );
    return `${withoutMemoryFlags} --gpu-memory-utilization ${selectedGpuMode.utilization} --max-model-len ${selectedGpuMode.maxModelLen}`;
  };
  const localDockerCommand = `docker run --gpus all --rm -p ${vllmPort}:8000 --ipc=host --env HF_TOKEN vllm/vllm-openai:latest "${vllmModel}" --gpu-memory-utilization ${selectedGpuMode.utilization} --max-model-len ${selectedGpuMode.maxModelLen}`;
  const localDockerAliasCommand = `${localDockerCommand} --served-model-name local-chat`;
  const vllmModelPresets = [
    {
      value: "google/gemma-2-2b-it",
      label: "Gemma 2 2B",
      description: "Gated; requires accepted access and HF_TOKEN.",
    },
    {
      value: "facebook/opt-125m",
      label: "OPT 125M",
      description: "Non-gated smoke test for Docker/vLLM.",
    },
    {
      value: "google/gemma-4-31B-it",
      label: "Gemma 4 31B",
      description: "Gated and large; needs much more VRAM/RAM.",
    },
  ];
  const selectedModelProfile = (() => {
    if (vllmModel === "google/gemma-2-2b-it") {
      return {
        title: "Selected model: Gemma 2 2B",
        tone: "ready",
        message:
          "Good local test model for your Docker/vLLM setup. It is gated, so Docker still needs HF_TOKEN and accepted Hugging Face access.",
      };
    }
    if (vllmModel === "facebook/opt-125m") {
      return {
        title: "Selected model: OPT 125M",
        tone: "ready",
        message:
          "Best smoke-test model when you only want to confirm Docker/vLLM works. It is not gated and should start quickly.",
      };
    }
    if (vllmModel === "google/gemma-4-31B-it") {
      return {
        title: "Selected model: Gemma 4 31B",
        tone: "warning",
        message:
          "This is a large gated model and is not a practical default for a 10GB GPU. Use the running Gemma 2B model, a smaller model, or a quantized/externally hosted server unless you have enough VRAM.",
      };
    }
    return {
      title: `Selected model: ${vllmModel}`,
      tone: "custom",
      message:
        "DocPilot will use this exact served model name. Make sure the vLLM server was started with the same model ID or served-model-name alias.",
    };
  })();
  const modelProfileWarning = selectedModelProfile.tone === "warning";

  const handleUseRunningModel = () => {
    if (!activeServedModel) return;
    const updatedSettings = {
      ...settings,
      model: activeServedModel,
      openaiCompatible: {
        ...settings.openaiCompatible,
        baseUrl: resolvedOpenAiBaseUrl,
      },
    };
    setSettings(updatedSettings);
    saveStoredSettings(updatedSettings);
    setSaved(true);
    setTimeout(() => setSaved(false), 2000);
  };

  const backendGpuMemoryMode =
    gpuMemoryMode === "max" ? "max_context" : gpuMemoryMode === "balanced" ? "balanced" : "safe_10gb";
  const localModelInProgress =
    localModelBusy ||
    ["checking_docker", "starting_docker", "starting_container", "loading_model"].includes(
      localModelStatus?.state || ""
    );

  const persistReadyLocalModel = (status: LocalModelStatus) => {
    const readyModel = status.served_model || status.model;
    if (!readyModel) return;

    const updatedSettings = {
      ...settings,
      provider: "openai-compatible" as const,
      model: readyModel,
      openaiCompatible: {
        ...settings.openaiCompatible,
        baseUrl: normalizeOpenAiBaseUrl(status.base_url || "http://localhost:8001/v1"),
      },
    };
    setSettings(updatedSettings);
    saveStoredSettings(updatedSettings);
    setSaved(true);
    setTimeout(() => setSaved(false), 2000);
  };

  const pollLocalModelUntilSettled = async (initialStatus: LocalModelStatus) => {
    let status = initialStatus;
    for (let attempt = 0; attempt < 450; attempt += 1) {
      setLocalModelStatus(status);

      if (status.state === "ready") {
        persistReadyLocalModel(status);
        return status;
      }

      if (status.state === "failed") {
        setLocalModelError(status.error_message || status.progress_message || "Local model startup failed.");
        return status;
      }

      await new Promise((resolve) => setTimeout(resolve, 2000));
      status = await getLocalModelStatus();
    }

    const timeoutStatus = await getLocalModelStatus();
    setLocalModelStatus(timeoutStatus);
    setLocalModelError("Timed out waiting for the local model to become ready.");
    return timeoutStatus;
  };

  const handleUseLocalModel = async () => {
    if (!hydrated) return;
    const model = settings.model.trim();
    if (!model) {
      setLocalModelError("Enter a model name first.");
      return;
    }

    setLocalModelBusy(true);
    setLocalModelError("");
    setTestResult(null);
    setRuntimeResult(null);

    try {
      const initialStatus = await applyLocalModel({
        model,
        hfToken: hfTokenInput.trim() || undefined,
        gpuMemoryMode: backendGpuMemoryMode,
      });
      setHfTokenInput("");
      await pollLocalModelUntilSettled(initialStatus);
    } catch (error) {
      setLocalModelError(error instanceof Error ? error.message : "Could not start the local model.");
    } finally {
      setLocalModelBusy(false);
    }
  };

  const handleStopLocalModel = async () => {
    setLocalModelBusy(true);
    setLocalModelError("");
    try {
      setLocalModelStatus(await stopLocalModel());
    } catch (error) {
      setLocalModelError(error instanceof Error ? error.message : "Could not stop the local model.");
    } finally {
      setLocalModelBusy(false);
    }
  };

  const handleDeleteLocalToken = async () => {
    setLocalModelError("");
    try {
      setLocalModelStatus(await deleteLocalModelToken());
    } catch (error) {
      setLocalModelError(error instanceof Error ? error.message : "Could not remove the saved Hugging Face token.");
    }
  };

  useEffect(() => {
    if (!hydrated || settings.provider !== "openai-compatible" || localModelBusy) return;
    if (
      !["checking_docker", "starting_docker", "starting_container", "loading_model"].includes(
        localModelStatus?.state || ""
      )
    ) {
      return;
    }

    let active = true;
    const poll = async () => {
      try {
        const status = await getLocalModelStatus();
        if (!active) return;
        setLocalModelStatus(status);
        if (status.state === "ready") {
          persistReadyLocalModel(status);
        }
        if (status.state === "failed") {
          setLocalModelError(status.error_message || status.progress_message || "Local model startup failed.");
        }
      } catch {
        // Keep the last visible state; the next poll or user action can recover.
      }
    };

    const interval = window.setInterval(() => void poll(), 2000);
    void poll();
    return () => {
      active = false;
      window.clearInterval(interval);
    };
  }, [hydrated, settings.provider, localModelStatus?.state, localModelBusy]);

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
            {!hydrated ? (
              <>
                <p className="text-sm font-semibold text-text-primary">Loading saved settings...</p>
                <p className="text-xs text-text-secondary mt-1">
                  DocPilot is loading your browser-saved AI provider configuration.
                </p>
              </>
            ) : (
              <>
                <p className="text-sm font-semibold text-text-primary">
                  Active provider: {getProviderLabel(settings.provider)}
                </p>
                <p className="text-xs text-text-secondary mt-1">
                  Chat and evaluation will use {settings.model} with {activeCredential}.
                </p>
                {settings.provider === "openai-compatible" ? (
                  <p className="text-xs text-text-tertiary mt-1">
                    Base URL: {resolvedOpenAiBaseUrl}
                  </p>
                ) : null}
              </>
            )}
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
                    DocPilot already uses port 8000, so local vLLM should normally use port 8001. Docker users may need http://host.docker.internal:8001/v1.
                  </p>
                  {usesDocPilotBackendPort ? (
                    <div className="mt-3 flex items-start gap-2 rounded-lg border border-warning/30 bg-warning/10 px-3 py-2 text-warning">
                      <AlertTriangle className="w-4 h-4 mt-0.5" />
                      <p className="text-xs leading-relaxed">
                        This points to DocPilot&apos;s backend, not vLLM. Hugging Face examples use port 8000 for standalone vLLM, but DocPilot already uses 8000. Use http://localhost:8001/v1 unless you moved DocPilot.
                      </p>
                    </div>
                  ) : null}
                </div>

                <div>
                  <label className="block text-xs font-medium text-text-secondary mb-1.5">
                    Model name
                  </label>
                  <div className="mb-3 grid grid-cols-1 gap-2 sm:grid-cols-2">
                    {vllmModelPresets.map((preset) => (
                      <button
                        key={preset.value}
                        type="button"
                        onClick={() => setSettings((current) => ({ ...current, model: preset.value }))}
                        className={`rounded-lg border px-3 py-2 text-left transition-colors ${
                          settings.model === preset.value
                            ? "border-accent/50 bg-accent/10"
                            : "border-border bg-surface-2/60 hover:bg-surface-2"
                        }`}
                      >
                        <p className="text-xs font-semibold text-text-primary">{preset.label}</p>
                        <p className="mt-1 text-[11px] text-text-tertiary">{preset.description}</p>
                      </button>
                    ))}
                  </div>
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

                <div className="rounded-xl border border-accent/30 bg-accent/5 p-4 space-y-4">
                  <div className="flex flex-col gap-3 sm:flex-row sm:items-start sm:justify-between">
                    <div>
                      <p className="text-sm font-semibold text-text-primary">Local AI Model</p>
                      <p className="mt-1 text-xs leading-relaxed text-text-secondary">
                        Enter a model name, click once, and DocPilot will start or switch the managed vLLM Docker server for you.
                      </p>
                    </div>
                    <button
                      type="button"
                      onClick={handleUseLocalModel}
                      disabled={!hydrated || localModelInProgress}
                      className="inline-flex items-center justify-center gap-2 rounded-lg bg-accent px-4 py-2 text-sm font-semibold text-white hover:bg-accent/90 disabled:opacity-50"
                    >
                      {localModelInProgress ? <Loader2 className="h-4 w-4 animate-spin" /> : <Server className="h-4 w-4" />}
                      {localModelInProgress ? "Preparing..." : "Use This Local Model"}
                    </button>
                  </div>

                  <div>
                    <label className="block text-xs font-medium text-text-secondary mb-1.5">
                      Hugging Face token (only needed for gated models)
                    </label>
                    <input
                      type="password"
                      value={hfTokenInput}
                      onChange={(event) => setHfTokenInput(event.target.value)}
                      placeholder={
                        localModelStatus?.hf_token_available
                          ? "Saved token available; leave blank to reuse it"
                          : "hf_... read token"
                      }
                      className="w-full px-3 py-2 rounded-lg bg-surface-2 border border-border text-sm text-text-primary
                        placeholder-text-tertiary focus:outline-none focus:border-accent/40 focus:ring-1 focus:ring-accent/20
                        transition-all duration-200"
                    />
                    <p className="mt-2 text-[11px] text-text-tertiary">
                      Tokens are stored only in DocPilot&apos;s local backend config and passed to Docker through the process environment, never localStorage or command text.
                    </p>
                  </div>

                  {localModelStatus ? (
                    <div
                      className={`rounded-lg border px-3 py-3 ${
                        localModelStatus.state === "ready"
                          ? "border-success/30 bg-success/10 text-success"
                          : localModelStatus.state === "failed"
                          ? "border-error/30 bg-error/10 text-error"
                          : "border-border bg-background/70 text-text-primary"
                      }`}
                    >
                      <div className="flex items-start gap-2">
                        {localModelStatus.state === "ready" ? (
                          <CheckCircle2 className="mt-0.5 h-4 w-4" />
                        ) : localModelStatus.state === "failed" ? (
                          <AlertTriangle className="mt-0.5 h-4 w-4" />
                        ) : (
                          <Loader2 className="mt-0.5 h-4 w-4 animate-spin" />
                        )}
                        <div className="space-y-1">
                          <p className="text-xs font-semibold">
                            {localModelStatus.state.replaceAll("_", " ")}
                            {localModelStatus.served_model ? `: ${localModelStatus.served_model}` : ""}
                          </p>
                          <p className="text-xs leading-relaxed">
                            {localModelStatus.error_message || localModelStatus.progress_message}
                          </p>
                          <p className="text-[11px] opacity-80">
                            Server: {localModelStatus.base_url || "http://localhost:8001/v1"} · Container: {localModelStatus.container_name}
                          </p>
                          <p className="text-[11px] opacity-80">
                            HF token: {localModelStatus.hf_token_available ? "saved locally" : "not saved"}
                          </p>
                        </div>
                      </div>
                    </div>
                  ) : null}

                  {localModelError ? (
                    <div className="flex items-start gap-2 rounded-lg border border-error/30 bg-error/10 px-3 py-2 text-error">
                      <AlertTriangle className="mt-0.5 h-4 w-4" />
                      <p className="text-xs leading-relaxed">{localModelError}</p>
                    </div>
                  ) : null}

                  <div className="flex flex-wrap gap-2">
                    <button
                      type="button"
                      onClick={handleStopLocalModel}
                      disabled={localModelInProgress}
                      className="rounded-lg border border-border px-3 py-1.5 text-xs font-medium text-text-secondary hover:bg-surface-2 disabled:opacity-50"
                    >
                      Stop local model
                    </button>
                    <button
                      type="button"
                      onClick={handleDeleteLocalToken}
                      className="rounded-lg border border-border px-3 py-1.5 text-xs font-medium text-text-secondary hover:bg-surface-2"
                    >
                      Remove saved HF token
                    </button>
                  </div>
                </div>

                <details className="rounded-xl border border-border bg-surface-2/60 p-4">
                  <summary className="cursor-pointer text-sm font-semibold text-text-primary">
                    Advanced diagnostics and manual details
                  </summary>
                  <div className="mt-4 space-y-4">

                <div className="rounded-xl border border-border bg-surface-2/60 p-4 space-y-3">
                  <div className="flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between">
                    <div>
                      <p className="text-sm font-medium text-text-primary">Check local vLLM runtime</p>
                      <p className="text-xs text-text-tertiary mt-1">
                        Checks whether your configured OpenAI-compatible server is running, then guides Docker or WSL setup.
                      </p>
                    </div>
                    <button
                      type="button"
                      onClick={handleCheckRuntime}
                      disabled={!hydrated || checkingRuntime}
                      className="inline-flex items-center justify-center gap-2 px-4 py-2 rounded-lg text-sm font-medium
                        bg-surface border border-border text-text-primary hover:bg-background disabled:opacity-50 transition-colors"
                    >
                      {checkingRuntime ? <Loader2 className="w-4 h-4 animate-spin" /> : <TerminalSquare className="w-4 h-4" />}
                      {checkingRuntime ? "Checking..." : "Check Runtime"}
                    </button>
                  </div>

                  {runtimeResult ? (
                    <div
                      className={`rounded-lg border px-3 py-3 ${
                        runtimeResult.ok
                          ? "border-success/30 bg-success/10 text-success"
                          : "border-warning/30 bg-warning/10 text-warning"
                      }`}
                    >
                      <div className="flex items-start gap-2">
                        {runtimeResult.ok ? (
                          <CheckCircle2 className="w-4 h-4 mt-0.5" />
                        ) : (
                          <AlertTriangle className="w-4 h-4 mt-0.5" />
                        )}
                        <div className="space-y-2">
                          <p className="text-xs font-semibold">{runtimeResult.title}</p>
                          <p className="text-xs leading-relaxed">{runtimeResult.message}</p>
                          <p className="text-xs leading-relaxed">{runtimeResult.recommendation}</p>
                          {runtimeResult.server_status ? (
                            <p className="text-[11px] opacity-80">
                              vLLM server: {runtimeResult.server_status.replaceAll("_", " ")}
                            </p>
                          ) : null}
                          {servedModels.length > 0 ? (
                            <div className="rounded-lg border border-border/70 bg-background/70 px-3 py-2 text-text-primary">
                              <div className="flex flex-col gap-2 sm:flex-row sm:items-center sm:justify-between">
                                <div>
                                  <p className="text-[11px] font-semibold">
                                    Running model: {activeServedModel}
                                  </p>
                                  {servedModels.length > 1 ? (
                                    <p className="mt-1 text-[11px] text-text-tertiary">
                                      Served models: {servedModels.join(", ")}
                                    </p>
                                  ) : null}
                                  {canUseRunningModel ? (
                                    <p className="mt-1 text-[11px] text-warning">
                                      DocPilot is currently set to {settings.model}. Sync it to the running vLLM model before chatting.
                                    </p>
                                  ) : null}
                                </div>
                                {canUseRunningModel ? (
                                  <button
                                    type="button"
                                    onClick={handleUseRunningModel}
                                    className="rounded-md border border-accent/40 bg-accent/10 px-3 py-1.5 text-[11px] font-semibold text-accent hover:bg-accent/15"
                                  >
                                    Use running model
                                  </button>
                                ) : null}
                              </div>
                            </div>
                          ) : null}
                          {runtimeResult.platform || runtimeResult.python_version ? (
                            <p className="text-[11px] opacity-80">
                              Runtime: {runtimeResult.platform || "unknown"} / Python {runtimeResult.python_version || "unknown"}
                            </p>
                          ) : null}
                          {runtimeResult.gpu_available ? (
                            <p className="text-[11px] opacity-80">
                              NVIDIA GPU detected. Local serving should use WSL2 or Docker GPU support, not native Windows vLLM.
                            </p>
                          ) : null}
                          {runtimeResult.gpu_memory_snapshot ? (
                            <p className="text-[11px] opacity-80">
                              GPU memory: {runtimeResult.gpu_memory_snapshot}
                            </p>
                          ) : null}
                          {runtimeResult.server_reachable ? (
                            <p className="text-[11px] opacity-80">
                              Server is already running; Docker, WSL, and HF_TOKEN checks only matter when starting or restarting vLLM.
                            </p>
                          ) : null}
                          {!runtimeResult.server_reachable && runtimeResult.docker_status ? (
                            <p className="text-[11px] opacity-80">
                              Docker: {runtimeResult.docker_status.replaceAll("_", " ")}
                            </p>
                          ) : null}
                          {!runtimeResult.server_reachable && runtimeResult.gated_model_likely ? (
                            <p className="text-[11px] opacity-80">
                              HF token: {runtimeResult.hf_token_available ? "available in backend environment" : "missing from backend environment"}
                            </p>
                          ) : null}
                          {!runtimeResult.server_reachable && runtimeResult.wsl_status ? (
                            <p className="text-[11px] opacity-80">
                              WSL: {runtimeResult.wsl_status.replaceAll("_", " ")}
                            </p>
                          ) : null}
                          {!runtimeResult.server_reachable && runtimeResult.docker_command ? (
                            <div className="rounded-lg border border-border/70 bg-background/70 p-3">
                              <div className="mb-2 flex items-center justify-between gap-3">
                                <p className="text-[11px] font-semibold text-text-primary">Fix Local vLLM</p>
                                <button
                                  type="button"
                                  onClick={handleCopyDockerCommand}
                                  className="rounded-md border border-border px-2 py-1 text-[11px] text-text-primary hover:bg-surface"
                                >
                                  {copiedCommand ? "Copied" : "Copy command"}
                                </button>
                              </div>
                              <p className="mb-2 text-[11px] leading-relaxed opacity-80">
                                {runtimeResult.docker_running
                                  ? "Docker is running. Run this in a terminal to start the selected model."
                                  : "Start Docker Desktop first, then run this in a terminal."}
                              </p>
                              <div className="mb-3">
                                <p className="mb-2 text-[11px] font-semibold text-text-primary">GPU memory mode</p>
                                <div className="grid gap-2 sm:grid-cols-3">
                                  {(Object.entries(gpuMemoryModes) as Array<[typeof gpuMemoryMode, typeof selectedGpuMode]>).map(([mode, option]) => (
                                    <button
                                      key={mode}
                                      type="button"
                                      onClick={() => setGpuMemoryMode(mode)}
                                      className={`rounded-md border px-2 py-2 text-left transition-colors ${
                                        gpuMemoryMode === mode
                                          ? "border-accent/50 bg-accent/10"
                                          : "border-border bg-background hover:bg-surface"
                                      }`}
                                    >
                                      <p className="text-[11px] font-semibold text-text-primary">{option.label}</p>
                                      <p className="mt-1 text-[10px] text-text-tertiary">{option.description}</p>
                                      <p className="mt-1 text-[10px] text-text-secondary">
                                        {option.utilization} / {option.maxModelLen}
                                      </p>
                                    </button>
                                  ))}
                                </div>
                              </div>
                              <div className="mb-3 rounded-md border border-warning/30 bg-warning/10 px-2 py-2 text-warning">
                                <p className="text-[11px] font-semibold">Gated Hugging Face model note</p>
                                <p className="mt-1 text-[11px] leading-relaxed">
                                  Gemma models require accepting access on Hugging Face and using a read token. Rotate any token that was pasted into chat or logs, then use a new read-only token.
                                </p>
                                <div className="mt-2 space-y-1 text-[11px]">
                                  <p>{runtimeResult.hf_token_available ? "OK" : "Needs action"}: HF_TOKEN is set in the terminal that starts Docker.</p>
                                  <p>Required: Hugging Face access for {settings.model} has been accepted.</p>
                                  <p>Required: Token is read-only and newly rotated.</p>
                                </div>
                                {runtimeResult.hf_token_recommendation ? (
                                  <p className="mt-2 text-[11px] leading-relaxed">{runtimeResult.hf_token_recommendation}</p>
                                ) : null}
                              </div>
                              <label className="mb-2 flex items-start gap-2 text-[11px] text-text-secondary">
                                <input
                                  type="checkbox"
                                  checked={hfTokenAvailable}
                                  onChange={(event) => setHfTokenAvailable(event.target.checked)}
                                  className="mt-0.5"
                                />
                                <span>
                                  I will set HF_TOKEN in this PowerShell terminal before running Docker. DocPilot will not store or send this token.
                                </span>
                              </label>
                              <code className="block rounded-md bg-background px-2 py-1 text-[11px] text-text-primary">
                                {hfTokenAvailable
                                  ? `$env:HF_TOKEN="<your-hugging-face-read-token>"\n${buildDisplayDockerCommand(runtimeResult.docker_command)}`
                                  : buildDisplayDockerCommand(runtimeResult.docker_command)}
                              </code>
                            </div>
                          ) : null}
                          {runtimeResult.details && runtimeResult.issue_code !== "vllm_server_not_running" ? (
                            <code className="block rounded-md bg-background/80 px-2 py-1 text-[11px] text-text-primary">
                              {runtimeResult.details}
                            </code>
                          ) : null}
                        </div>
                      </div>
                    </div>
                  ) : null}
                </div>

                <div className="rounded-xl border border-border bg-surface-2/60 p-4 space-y-3">
                  <div className="flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between">
                    <div>
                      <p className="text-sm font-medium text-text-primary">Test vLLM connection</p>
                      <p className="text-xs text-text-tertiary mt-1">
                        Calls {resolvedOpenAiBaseUrl} with model {settings.model || "your model"}.
                      </p>
                    </div>
                    <button
                      type="button"
                      onClick={handleTestConnection}
                      disabled={!hydrated || testing}
                      className="inline-flex items-center justify-center gap-2 px-4 py-2 rounded-lg text-sm font-medium
                        bg-accent text-white hover:bg-accent/90 disabled:opacity-50 transition-colors"
                    >
                      {testing ? <Loader2 className="w-4 h-4 animate-spin" /> : <Server className="w-4 h-4" />}
                      {testing ? "Testing..." : "Test Connection"}
                    </button>
                  </div>

                  {testResult ? (
                    <div
                      className={`flex items-start gap-2 rounded-lg border px-3 py-2 ${
                        testResult.ok
                          ? "border-success/30 bg-success/10 text-success"
                          : "border-error/30 bg-error/10 text-error"
                      }`}
                    >
                      {testResult.ok ? (
                        <CheckCircle2 className="w-4 h-4 mt-0.5" />
                      ) : (
                        <AlertTriangle className="w-4 h-4 mt-0.5" />
                      )}
                      <div className="space-y-2">
                        <p className="text-xs leading-relaxed">{testResult.message}</p>
                        {canUseRunningModel ? (
                          <div className="rounded-lg border border-border/70 bg-background/70 px-3 py-2 text-text-primary">
                            <p className="text-[11px] leading-relaxed">
                              The server is already running {activeServedModel}. To chat now, sync DocPilot to that model.
                            </p>
                            <button
                              type="button"
                              onClick={handleUseRunningModel}
                              className="mt-2 rounded-md border border-accent/40 bg-accent/10 px-3 py-1.5 text-[11px] font-semibold text-accent hover:bg-accent/15"
                            >
                              Use running model: {activeServedModel}
                            </button>
                          </div>
                        ) : null}
                      </div>
                    </div>
                  ) : null}
                </div>

                <div className="rounded-xl border border-accent/20 bg-accent/5 p-4">
                  <p className="text-sm font-semibold text-text-primary">Switching Hugging Face models</p>
                  <div className="mt-3 space-y-2 text-xs text-text-secondary leading-relaxed">
                    <p>Hugging Face examples often use http://localhost:8000/v1 because they assume vLLM is the only server. In DocPilot, port 8000 is already the AI backend, so run vLLM on 8001.</p>
                    <p>Do not run native Windows <code>vllm serve</code>; on this machine it fails because native Windows vLLM cannot load <code>vllm._C</code>. Use Docker or WSL instead.</p>
                    {activeServedModel ? (
                      <div className="rounded-lg border border-border/70 bg-background/70 px-3 py-2 text-text-primary">
                        <p className="text-[11px] font-semibold">Running now: {activeServedModel}</p>
                        {canUseRunningModel ? (
                          <>
                            <p className="mt-1 text-[11px] text-text-secondary">
                              Settings is currently selecting {settings.model}. If you want to use the already-running server, sync to {activeServedModel}.
                            </p>
                            <button
                              type="button"
                              onClick={handleUseRunningModel}
                              className="mt-2 rounded-md border border-accent/40 bg-accent/10 px-3 py-1.5 text-[11px] font-semibold text-accent hover:bg-accent/15"
                            >
                              Use running model
                            </button>
                          </>
                        ) : (
                          <p className="mt-1 text-[11px] text-text-secondary">
                            Settings already matches the running vLLM model.
                          </p>
                        )}
                      </div>
                    ) : null}
                    <div
                      className={`rounded-lg border px-3 py-2 ${
                        modelProfileWarning
                          ? "border-warning/30 bg-warning/10 text-warning"
                          : "border-border/70 bg-background/70 text-text-primary"
                      }`}
                    >
                      <p className="text-[11px] font-semibold">{selectedModelProfile.title}</p>
                      <p className="mt-1 text-[11px] leading-relaxed">{selectedModelProfile.message}</p>
                    </div>
                    <p>1. Stop the current vLLM process.</p>
                    <p>2. Start the selected model with Docker on the same DocPilot-compatible port:</p>
                    <code className="block rounded-lg bg-background/80 px-3 py-2 text-[11px] text-text-primary">
                      {localDockerCommand}
                    </code>
                    <p>3. Or use a stable alias so DocPilot can keep the same model name while you swap models behind it:</p>
                    <code className="block rounded-lg bg-background/80 px-3 py-2 text-[11px] text-text-primary">
                      {localDockerAliasCommand}
                    </code>
                    <p>If you use the alias, set Model name to local-chat. After Docker starts, click Check Runtime and then Use running model if DocPilot shows a mismatch.</p>
                  </div>
                </div>
                  </div>
                </details>
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
              disabled={!hydrated}
              className="flex items-center gap-2 px-5 py-2.5 rounded-lg text-sm font-medium
                bg-accent text-white hover:bg-accent/90 disabled:opacity-50 transition-all duration-200"
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
