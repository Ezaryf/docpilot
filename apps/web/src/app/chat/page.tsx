"use client";

import { useRef, useEffect, useState, useCallback } from "react";
import { AnimatePresence } from "framer-motion";
import { Sparkles, PanelRightOpen } from "lucide-react";
import { Sidebar } from "@/components/sidebar";
import { ChatMessage, StreamingIndicator } from "@/components/chat-message";
import { ChatInput } from "@/components/chat-input";
import { FileUploadPanel } from "@/components/file-upload";
import { useChatStore } from "@/stores/chat-store";
import { useDocumentStore } from "@/stores/document-store";
import { applyLocalModel, getLocalModelStatus, streamChat, type LocalModelStatus } from "@/lib/api";
import {
  DEFAULT_SETTINGS,
  getLlmRequestConfig,
  normalizeOpenAiBaseUrl,
  readStoredSettings,
  saveStoredSettings,
} from "@/lib/settings";
import type { Citation, TraceStep } from "@/stores/chat-store";

function isManagedLocalVllm(baseUrl: string) {
  const normalized = normalizeOpenAiBaseUrl(baseUrl);
  return /^https?:\/\/(localhost|127\.0\.0\.1):8001\/v1\b/i.test(normalized);
}

async function waitForLocalModel(model: string, onProgress: (status: LocalModelStatus) => void) {
  let status = await getLocalModelStatus();
  if (status.state === "ready" && (status.served_model === model || status.model === model)) {
    onProgress(status);
    return status;
  }

  status = await applyLocalModel({ model, gpuMemoryMode: "safe_10gb" });
  for (let attempt = 0; attempt < 450; attempt += 1) {
    onProgress(status);
    if (status.state === "ready") return status;
    if (status.state === "failed") return status;
    await new Promise((resolve) => setTimeout(resolve, 2000));
    status = await getLocalModelStatus();
  }

  return {
    ...status,
    state: "failed" as const,
    error_code: "local_model_timeout",
    error_message: "Timed out waiting for the local model to become ready.",
    progress_message: "Timed out waiting for the local model to become ready.",
  };
}

export default function ChatPage() {
  const {
    sessions,
    activeSessionId,
    isStreaming,
    createSession,
    addMessage,
    updateMessage,
    setIsStreaming,
  } = useChatStore();
  const documents = useDocumentStore((state) => state.documents);
  const focusedDocumentNames = useDocumentStore((state) => state.focusedDocumentNames);

  const [showUpload, setShowUpload] = useState(false);
  const [chatSettings, setChatSettings] = useState(DEFAULT_SETTINGS);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const abortRef = useRef<AbortController | null>(null);
  const contentRef = useRef("");

  const activeSession = sessions.find((session) => session.id === activeSessionId);
  const readyDocuments = documents.filter((doc) => doc.status === "ready");

  useEffect(() => {
    setChatSettings(readStoredSettings());
  }, []);

  // Auto-scroll to bottom
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [activeSession?.messages, isStreaming]);

  const handleSend = useCallback(
    async (message: string) => {
      let sessionId = activeSessionId;
      if (!sessionId) {
        sessionId = createSession();
      }

      // Add user message
      addMessage(sessionId, {
        id: crypto.randomUUID(),
        role: "user",
        content: message,
        timestamp: Date.now(),
      });

      // Add placeholder for assistant
      const assistantId = crypto.randomUUID();
      addMessage(sessionId, {
        id: assistantId,
        role: "assistant",
        content: "",
        timestamp: Date.now(),
      });

      setIsStreaming(true);
      contentRef.current = "";
      abortRef.current = new AbortController();
      const latestSettings = readStoredSettings();
      setChatSettings(latestSettings);
      let effectiveSettings = latestSettings;
      let latestLlmConfig = getLlmRequestConfig(effectiveSettings);

      if (
        effectiveSettings.provider === "openai-compatible" &&
        isManagedLocalVllm(effectiveSettings.openaiCompatible.baseUrl)
      ) {
        contentRef.current = "Starting local model...";
        updateMessage(sessionId, assistantId, { content: contentRef.current });

        const localStatus = await waitForLocalModel(effectiveSettings.model, (status) => {
          contentRef.current = `Starting local model...\n\n${status.progress_message || status.state}`;
          updateMessage(sessionId!, assistantId, { content: contentRef.current });
        });

        if (localStatus.state !== "ready") {
          contentRef.current = `⚠️ ${
            localStatus.error_message ||
            localStatus.progress_message ||
            "Local model startup failed. Open Settings to add a Hugging Face token or choose a smaller model."
          }`;
          updateMessage(sessionId, assistantId, { content: contentRef.current });
          setIsStreaming(false);
          return;
        }

        const readyModel = localStatus.served_model || localStatus.model || effectiveSettings.model;
        effectiveSettings = {
          ...effectiveSettings,
          provider: "openai-compatible",
          model: readyModel,
          openaiCompatible: {
            ...effectiveSettings.openaiCompatible,
            baseUrl: normalizeOpenAiBaseUrl(localStatus.base_url || "http://localhost:8001/v1"),
          },
        };
        saveStoredSettings(effectiveSettings);
        setChatSettings(effectiveSettings);
        latestLlmConfig = getLlmRequestConfig(effectiveSettings);
        contentRef.current = "";
        updateMessage(sessionId, assistantId, { content: "" });
      }

      await streamChat(
        message,
        sessionId,
        focusedDocumentNames,
        readyDocuments.length > 0,
        latestLlmConfig,
        {
          onStatus: (statusMessage: string) => {
            if (!contentRef.current && statusMessage) {
              updateMessage(sessionId!, assistantId, { content: statusMessage });
            }
          },
          onToken: (token: string) => {
            contentRef.current += token;
            updateMessage(sessionId!, assistantId, { content: contentRef.current });
          },
          onCitations: (citations: Citation[]) => {
            updateMessage(sessionId!, assistantId, { citations });
          },
          onTrace: (trace: TraceStep[]) => {
            updateMessage(sessionId!, assistantId, { trace });
          },
          onDone: () => {
            setIsStreaming(false);
          },
          onError: (error: string) => {
            contentRef.current = contentRef.current
              ? `${contentRef.current}\n\n⚠️ ${error}`
              : `⚠️ ${error}`;
            updateMessage(sessionId!, assistantId, { content: contentRef.current });
            setIsStreaming(false);
          },
        },
        abortRef.current.signal
      );
    },
    [
      activeSessionId,
      createSession,
      addMessage,
      updateMessage,
      setIsStreaming,
      focusedDocumentNames,
      readyDocuments.length,
    ]
  );

  const handleStop = () => {
    abortRef.current?.abort();
    setIsStreaming(false);
  };

  return (
    <div className="flex h-screen overflow-hidden bg-background">
      <Sidebar />

      {/* Main Chat Area */}
      <div className="flex-1 flex flex-col min-w-0">
        {/* Header */}
        <header className="h-14 flex items-center justify-between px-4 border-b border-border bg-surface/80 backdrop-blur-sm flex-shrink-0">
          <div className="min-w-0" title={`Model: ${chatSettings.model}`}>
            <div className="flex items-center gap-2">
              <Sparkles className="w-4 h-4 text-accent" />
              <h1 className="text-sm font-semibold text-text-primary truncate">
                {activeSession?.title || "New Conversation"}
              </h1>
            </div>
            <p className="text-[11px] text-text-tertiary mt-0.5 truncate">
              {readyDocuments.length === 0
                ? "Upload a document to start grounded chat."
                : focusedDocumentNames.length > 0
                ? `Focused on: ${focusedDocumentNames.join(", ")}`
                : `Chat searches across all indexed documents: ${readyDocuments
                    .map((doc) => doc.name)
                    .join(", ")}`}
            </p>
          </div>
          <button
            onClick={() => setShowUpload(!showUpload)}
            className="p-2 rounded-lg text-text-tertiary hover:text-text-primary hover:bg-surface-2 transition-colors"
            aria-label="Toggle documents panel"
          >
            <PanelRightOpen className="w-4 h-4" />
          </button>
        </header>

        <div className="flex flex-1 overflow-hidden">
          {/* Messages */}
          <div className="flex-1 overflow-y-auto">
            {!activeSession || activeSession.messages.length === 0 ? (
              /* Empty State */
              <div className="flex items-center justify-center h-full">
                <div className="text-center max-w-md px-6">
                  <div className="w-14 h-14 rounded-2xl bg-gradient-to-br from-accent to-accent-2 flex items-center justify-center mx-auto mb-5 shadow-lg shadow-accent/20">
                    <Sparkles className="w-7 h-7 text-white" />
                  </div>
                  <h2 className="text-xl font-semibold text-text-primary mb-2">
                    What would you like to know?
                  </h2>
                  <p className="text-sm text-text-secondary mb-6 leading-relaxed">
                    Upload documents and ask questions. DocPilot will retrieve
                    relevant information, grade its quality, and generate
                    citation-backed answers.
                  </p>
                  <div className="grid grid-cols-1 gap-2">
                    {[
                      "Summarize the key points from my uploaded documents",
                      "What are the main differences between the approaches described?",
                      "Find all mentions of performance metrics",
                    ].map((suggestion) => (
                      <button
                        key={suggestion}
                        onClick={() => handleSend(suggestion)}
                        className="text-left px-4 py-2.5 rounded-lg border border-border bg-surface hover:bg-surface-2 hover:border-border-hover
                          text-sm text-text-secondary hover:text-text-primary transition-all duration-150"
                      >
                        {suggestion}
                      </button>
                    ))}
                  </div>
                </div>
              </div>
            ) : (
              <div className="max-w-3xl mx-auto">
                {activeSession.messages.map((msg, i) => (
                  <ChatMessage
                    key={msg.id}
                    message={msg}
                    isLast={i === activeSession.messages.length - 1}
                  />
                ))}
                {isStreaming && !activeSession.messages[activeSession.messages.length - 1]?.content && (
                  <StreamingIndicator />
                )}
                <div ref={messagesEndRef} />
              </div>
            )}
          </div>

          {/* Upload Panel */}
          <AnimatePresence>
            {showUpload && (
              <FileUploadPanel
                onClose={() => setShowUpload(false)}
                onPromptSelect={(prompt) => {
                  setShowUpload(false);
                  void handleSend(prompt);
                }}
              />
            )}
          </AnimatePresence>
        </div>

        {/* Input */}
        <ChatInput
          onSend={handleSend}
          onFileSelect={() => setShowUpload(true)}
          isStreaming={isStreaming}
          onStop={handleStop}
        />
      </div>
    </div>
  );
}
