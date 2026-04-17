import type { Message, Citation, TraceStep } from "@/stores/chat-store";

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

export interface StreamCallbacks {
  onToken: (token: string) => void;
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
  groqApiKey: string | null,
  llmModel: string | null,
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
        groq_api_key: groqApiKey,
        llm_model: llmModel,
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
