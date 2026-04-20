import type { Citation, TraceStep } from "@/stores/chat-store";
import type { LlmRequestConfig } from "@/lib/settings";

const AI_SERVICE_URL = process.env.NEXT_PUBLIC_AI_SERVICE_URL || "http://localhost:8000";

export interface UploadResponse {
  document_name: string;
  chunks: number;
  characters: number;
  status: "ready";
}

export interface IndexedDocument {
  name: string;
  chunks: number;
  total_chunks?: number;
}

export interface LlmEnvironmentDiagnostic {
  ok: boolean;
  status: string;
  issue_code: string;
  title: string;
  message: string;
  recommendation: string;
  platform?: string;
  python_version?: string;
  is_windows?: boolean;
  is_wsl?: boolean;
  vllm_installed?: boolean;
  vllm_import_ok?: boolean;
  native_python_vllm_ok?: boolean;
  native_python_vllm_details?: string;
  server_reachable?: boolean;
  server_status?: string;
  server_details?: string;
  served_models?: string[];
  active_served_model?: string;
  startup_error_kind?: string;
  hf_token_available?: boolean;
  hf_token_source?: string;
  gated_model_likely?: boolean;
  hf_token_recommendation?: string;
  gpu_memory_mode?: string;
  gpu_memory_utilization?: string;
  max_model_len?: string;
  gpu_memory_snapshot?: string;
  base_url?: string;
  model?: string;
  setup_command?: string;
  docker_command?: string;
  docker_hint?: string;
  recommended_action?: string;
  wsl_available?: boolean;
  wsl_status?: string;
  wsl_details?: string;
  docker_available?: boolean;
  docker_running?: boolean;
  docker_status?: string;
  docker_details?: string;
  gpu_available?: boolean;
  gpu_details?: string;
  details?: string;
}

export type LocalModelState =
  | "idle"
  | "checking_docker"
  | "starting_docker"
  | "starting_container"
  | "loading_model"
  | "ready"
  | "failed";

export interface LocalModelStatus {
  state: LocalModelState;
  model: string;
  served_model: string;
  base_url: string;
  container_name: string;
  progress_message: string;
  error_code: string;
  error_message: string;
  logs_tail: string;
  hf_token_available: boolean;
}

export interface ApplyLocalModelRequest {
  model: string;
  hfToken?: string;
  gpuMemoryMode?: "safe_10gb" | "balanced" | "max_context";
}

export interface StreamCallbacks {
  onToken: (token: string) => void;
  onStatus?: (message: string) => void;
  onCitations: (citations: Citation[]) => void;
  onTrace: (trace: TraceStep[]) => void;
  onDone: () => void;
  onError: (error: string) => void;
}

export async function streamChat(
  query: string,
  sessionId: string,
  documentNames: string[],
  hasDocuments: boolean,
  llmConfig: LlmRequestConfig,
  callbacks: StreamCallbacks,
  signal?: AbortSignal
) {
  try {
    const res = await fetch(`${AI_SERVICE_URL}/api/chat`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        query,
        session_id: sessionId,
        document_names: documentNames,
        has_documents: hasDocuments,
        llm_provider: llmConfig.provider,
        llm_model: llmConfig.model,
        groq_api_key: llmConfig.groqApiKey,
        openai_base_url: llmConfig.openaiBaseUrl,
        openai_api_key: llmConfig.openaiApiKey,
      }),
      signal,
    });

    if (!res.ok) {
      const errText = await res.text();
      callbacks.onError(errText || `HTTP ${res.status}`);
      return;
    }

    const reader = res.body?.getReader();
    if (!reader) {
      callbacks.onError("No response stream");
      return;
    }

    const decoder = new TextDecoder();
    let buffer = "";

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;

      buffer += decoder.decode(value, { stream: true });
      const lines = buffer.split("\n");
      buffer = lines.pop() || "";

      for (const line of lines) {
        if (!line.startsWith("data: ")) continue;
        const data = line.slice(6).trim();

        if (data === "[DONE]") {
          callbacks.onDone();
          return;
        }

        try {
          const parsed = JSON.parse(data);

          if (parsed.type === "token") {
            callbacks.onToken(parsed.content);
          } else if (parsed.type === "status") {
            callbacks.onStatus?.(parsed.message || parsed.content || "");
          } else if (parsed.type === "citations") {
            callbacks.onCitations(parsed.citations);
          } else if (parsed.type === "trace") {
            callbacks.onTrace(parsed.trace);
          } else if (parsed.type === "error") {
            callbacks.onError(parsed.message);
          }
        } catch {
          // Not JSON — treat as raw token
          callbacks.onToken(data);
        }
      }
    }

    callbacks.onDone();
  } catch (error: any) {
    if (error.name === "AbortError") return;
    callbacks.onError(error.message || "Stream failed");
  }
}

export async function runEvaluation(llmConfig: LlmRequestConfig) {
  const res = await fetch(`${AI_SERVICE_URL}/api/eval`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      llm_provider: llmConfig.provider,
      llm_model: llmConfig.model,
      groq_api_key: llmConfig.groqApiKey,
      openai_base_url: llmConfig.openaiBaseUrl,
      openai_api_key: llmConfig.openaiApiKey,
    }),
  });

  if (!res.ok) {
    throw new Error(`Evaluation request failed with HTTP ${res.status}`);
  }

  return res.json();
}

export async function testLlmConnection(llmConfig: LlmRequestConfig) {
  const res = await fetch(`${AI_SERVICE_URL}/api/llm/test`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      llm_provider: llmConfig.provider,
      llm_model: llmConfig.model,
      groq_api_key: llmConfig.groqApiKey,
      openai_base_url: llmConfig.openaiBaseUrl,
      openai_api_key: llmConfig.openaiApiKey,
    }),
  });

  if (!res.ok) {
    throw new Error(`Connection test failed with HTTP ${res.status}`);
  }

  return res.json();
}

export async function checkLlmEnvironment(llmConfig: LlmRequestConfig): Promise<LlmEnvironmentDiagnostic> {
  const res = await fetch(`${AI_SERVICE_URL}/api/llm/environment`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      llm_model: llmConfig.model,
      openai_base_url: llmConfig.openaiBaseUrl,
      check_native_python: false,
    }),
  });

  if (!res.ok) {
    throw new Error(`Runtime check failed with HTTP ${res.status}`);
  }

  return res.json();
}

export async function applyLocalModel(request: ApplyLocalModelRequest): Promise<LocalModelStatus> {
  const res = await fetch(`${AI_SERVICE_URL}/api/llm/local/apply`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      model: request.model,
      hf_token: request.hfToken || null,
      gpu_memory_mode: request.gpuMemoryMode || "safe_10gb",
    }),
  });

  if (!res.ok) {
    throw new Error(`Local model request failed with HTTP ${res.status}`);
  }

  return res.json();
}

export async function getLocalModelStatus(): Promise<LocalModelStatus> {
  const res = await fetch(`${AI_SERVICE_URL}/api/llm/local/status`);

  if (!res.ok) {
    throw new Error(`Local model status failed with HTTP ${res.status}`);
  }

  return res.json();
}

export async function stopLocalModel(): Promise<LocalModelStatus> {
  const res = await fetch(`${AI_SERVICE_URL}/api/llm/local/stop`, {
    method: "POST",
  });

  if (!res.ok) {
    throw new Error(`Stop local model failed with HTTP ${res.status}`);
  }

  return res.json();
}

export async function deleteLocalModelToken(): Promise<LocalModelStatus> {
  const res = await fetch(`${AI_SERVICE_URL}/api/llm/local/token`, {
    method: "DELETE",
  });

  if (!res.ok) {
    throw new Error(`Delete local token failed with HTTP ${res.status}`);
  }

  return res.json();
}

export async function uploadDocument(file: File): Promise<UploadResponse> {
  const formData = new FormData();
  formData.append("file", file);

  const res = await fetch(`${AI_SERVICE_URL}/api/upload`, {
    method: "POST",
    body: formData,
  });

  if (!res.ok) {
    const errText = await res.text();
    throw new Error(errText || `Upload failed with HTTP ${res.status}`);
  }

  return res.json();
}

export async function fetchIndexedDocuments(): Promise<IndexedDocument[]> {
  const res = await fetch(`${AI_SERVICE_URL}/api/documents`);

  if (!res.ok) {
    const errText = await res.text();
    throw new Error(errText || `Failed to load documents (HTTP ${res.status})`);
  }

  const data = await res.json();
  return data.documents ?? [];
}

export async function deleteDocument(documentName: string): Promise<void> {
  const res = await fetch(`${AI_SERVICE_URL}/api/documents/${encodeURIComponent(documentName)}`, {
    method: "DELETE",
  });

  if (!res.ok) {
    const errText = await res.text();
    throw new Error(errText || `Failed to delete document (HTTP ${res.status})`);
  }
}
