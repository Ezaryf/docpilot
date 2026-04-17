"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import type { FileRejection } from "react-dropzone";
import { useDropzone } from "react-dropzone";
import { motion, AnimatePresence } from "framer-motion";
import {
  AlertCircle,
  ArrowUpRight,
  CheckCircle2,
  Crosshair,
  Clock3,
  File,
  Loader2,
  RefreshCw,
  Sparkles,
  Trash2,
  Upload,
  X,
} from "lucide-react";
import { deleteDocument, fetchIndexedDocuments, uploadDocument } from "@/lib/api";
import { useDocumentStore, type UploadedDocument } from "@/stores/document-store";

const MAX_FILE_SIZE = 50 * 1024 * 1024;
const MAX_PARALLEL_UPLOADS = 2;
const ACCEPTED_TYPES = {
  "application/pdf": [".pdf"],
  "text/plain": [".txt"],
  "text/markdown": [".md"],
  "application/vnd.openxmlformats-officedocument.wordprocessingml.document": [".docx"],
};

interface FileUploadPanelProps {
  onClose: () => void;
  onPromptSelect?: (prompt: string) => void;
}

function createUploadErrorMessage(rejection: FileRejection) {
  const messages = rejection.errors.map((error) => {
    if (error.code === "file-too-large") {
      return `${rejection.file.name} is larger than 50MB.`;
    }

    if (error.code === "file-invalid-type") {
      return `${rejection.file.name} is not supported. Use PDF, TXT, MD, or DOCX.`;
    }

    return `${rejection.file.name}: ${error.message}`;
  });

  return messages.join(" ");
}

function formatSize(bytes: number) {
  if (bytes <= 0) return "Unknown size";
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

function getStatusIcon(status: UploadedDocument["status"]) {
  switch (status) {
    case "queued":
      return <Clock3 className="w-4 h-4 text-text-tertiary" />;
    case "uploading":
    case "processing":
      return <Loader2 className="w-4 h-4 text-accent animate-spin" />;
    case "ready":
      return <CheckCircle2 className="w-4 h-4 text-success" />;
    case "error":
      return <AlertCircle className="w-4 h-4 text-error" />;
  }
}

function DocumentCard({
  doc,
  action,
  highlighted = false,
}: {
  doc: UploadedDocument;
  action?: React.ReactNode;
  highlighted?: boolean;
}) {
  return (
    <motion.div
      initial={{ opacity: 0, y: 8 }}
      animate={{ opacity: 1, y: 0 }}
      exit={{ opacity: 0, y: -8 }}
      className={`rounded-2xl border p-3 ${
        highlighted
          ? "border-accent/40 bg-accent/5"
          : "border-border bg-surface-2/60"
      }`}
    >
      <div className="flex items-start gap-3">
        <div className="w-9 h-9 rounded-xl bg-surface flex items-center justify-center border border-border flex-shrink-0">
          {doc.status === "ready" ? (
            <CheckCircle2 className="w-4 h-4 text-success" />
          ) : doc.status === "error" ? (
            <AlertCircle className="w-4 h-4 text-error" />
          ) : (
            <File className="w-4 h-4 text-text-tertiary" />
          )}
        </div>
        <div className="flex-1 min-w-0">
          <div className="flex items-center justify-between gap-3">
            <p className="text-sm font-medium text-text-primary truncate">{doc.name}</p>
            <div className="flex items-center gap-2">
              {action}
              {getStatusIcon(doc.status)}
            </div>
          </div>
          <p
            className={`text-xs mt-1 leading-relaxed ${
              doc.status === "error" ? "text-error" : "text-text-secondary"
            }`}
          >
            {doc.error || doc.stageLabel || "Ready"}
          </p>
          <div className="flex items-center gap-2 text-[11px] text-text-tertiary mt-2 flex-wrap">
            <span>{formatSize(doc.size)}</span>
            <span>•</span>
            <span>{doc.chunks} chunks</span>
            <span>•</span>
            <span>{doc.source === "server" ? "Indexed" : "Local queue"}</span>
          </div>
          {doc.status !== "ready" && doc.status !== "error" && (
            <div className="mt-3 h-1.5 rounded-full bg-border overflow-hidden">
              <motion.div
                className="h-full rounded-full bg-gradient-to-r from-accent to-accent-2"
                initial={{ width: 0 }}
                animate={{ width: `${doc.progress}%` }}
                transition={{ duration: 0.3 }}
              />
            </div>
          )}
        </div>
      </div>
    </motion.div>
  );
}

export function FileUploadPanel({ onClose, onPromptSelect }: FileUploadPanelProps) {
  const {
    documents,
    addDocument,
    updateDocument,
    removeDocument,
    removeDocumentsByName,
    upsertDocuments,
    focusedDocumentNames,
    toggleFocusedDocument,
    clearFocusedDocuments,
  } = useDocumentStore();
  const [dragActive, setDragActive] = useState(false);
  const [notice, setNotice] = useState<string | null>(null);
  const [isSyncing, setIsSyncing] = useState(false);
  const queueRef = useRef<Array<{ docId: string; file: File }>>([]);
  const activeUploadsRef = useRef(0);

  const visibleDocuments = documents;
  const readyDocuments = useMemo(
    () => visibleDocuments.filter((doc) => doc.status === "ready"),
    [visibleDocuments]
  );
  const inFlightDocuments = useMemo(
    () =>
      visibleDocuments.filter((doc) =>
        ["queued", "uploading", "processing"].includes(doc.status)
      ),
    [visibleDocuments]
  );
  const failedDocuments = useMemo(
    () => visibleDocuments.filter((doc) => doc.status === "error"),
    [visibleDocuments]
  );
  const totalChunks = readyDocuments.reduce((sum, doc) => sum + doc.chunks, 0);
  const focusedNamesSet = useMemo(
    () => new Set(focusedDocumentNames.map((name) => name.trim().toLowerCase())),
    [focusedDocumentNames]
  );

  const syncIndexedDocuments = useCallback(async () => {
    setIsSyncing(true);

    try {
      const indexedDocuments = await fetchIndexedDocuments();
      const timestamp = Date.now();

      upsertDocuments(
        indexedDocuments.map((doc, index) => ({
          id: `server:${doc.name}`,
          name: doc.name,
          size: 0,
          type: "indexed",
          chunks: doc.total_chunks || doc.chunks || 0,
          uploadedAt: timestamp - index,
          status: "ready" as const,
          progress: 100,
          source: "server" as const,
          stageLabel: "Indexed and ready for chat",
        }))
      );
      setNotice(null);
    } catch (error: any) {
      setNotice(error.message || "Unable to refresh indexed documents right now.");
    } finally {
      setIsSyncing(false);
    }
  }, [upsertDocuments]);

  const processQueue = useCallback(() => {
    while (
      activeUploadsRef.current < MAX_PARALLEL_UPLOADS &&
      queueRef.current.length > 0
    ) {
      const next = queueRef.current.shift();
      if (!next) return;

      activeUploadsRef.current += 1;
      updateDocument(next.docId, {
        status: "uploading",
        progress: 30,
        stageLabel: "Uploading file",
        error: undefined,
      });

      void (async () => {
        try {
          updateDocument(next.docId, {
            status: "processing",
            progress: 72,
            stageLabel: "Chunking and indexing",
          });

          const result = await uploadDocument(next.file);

          updateDocument(next.docId, {
            name: result.document_name,
            status: "ready",
            progress: 100,
            chunks: result.chunks || 0,
            source: "server",
            stageLabel: "Indexed and ready for chat",
          });
        } catch (error: any) {
          updateDocument(next.docId, {
            status: "error",
            progress: 100,
            stageLabel: "Upload failed",
            error: error.message || "Upload failed",
          });
        } finally {
          activeUploadsRef.current -= 1;
          void syncIndexedDocuments();
          processQueue();
        }
      })();
    }
  }, [syncIndexedDocuments, updateDocument]);

  const enqueueUpload = useCallback(
    (file: File) => {
      const duplicate = visibleDocuments.some(
        (doc) => doc.name === file.name && doc.status !== "error"
      );

      if (duplicate) {
        setNotice(`${file.name} is already in your library or current upload queue.`);
        return;
      }

      const docId = crypto.randomUUID();
      addDocument({
        id: docId,
        name: file.name,
        size: file.size,
        type: file.type,
        chunks: 0,
        uploadedAt: Date.now(),
        status: "queued",
        progress: 8,
        source: "local",
        stageLabel: "Waiting for an upload slot",
      });

      queueRef.current.push({ docId, file });
      processQueue();
    },
    [addDocument, processQueue, visibleDocuments]
  );

  useEffect(() => {
    void syncIndexedDocuments();
  }, [syncIndexedDocuments]);

  const onDrop = useCallback(
    (acceptedFiles: File[], rejectedFiles: FileRejection[]) => {
      setDragActive(false);

      if (rejectedFiles.length > 0) {
        setNotice(rejectedFiles.map(createUploadErrorMessage).join(" "));
      } else {
        setNotice(null);
      }

      for (const file of acceptedFiles) {
        enqueueUpload(file);
      }
    },
    [enqueueUpload]
  );

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: ACCEPTED_TYPES,
    maxSize: MAX_FILE_SIZE,
    onDragEnter: () => setDragActive(true),
    onDragLeave: () => setDragActive(false),
  });

  return (
    <motion.div
      initial={{ opacity: 0, x: 20 }}
      animate={{ opacity: 1, x: 0 }}
      exit={{ opacity: 0, x: 20 }}
      className="w-[360px] border-l border-border bg-surface h-full flex flex-col"
    >
      <div className="flex items-center justify-between p-4 border-b border-border">
        <div>
          <h3 className="text-sm font-semibold text-text-primary">Documents</h3>
          <p className="text-xs text-text-tertiary mt-0.5">
            Upload, index, then ask citation-backed questions.
          </p>
        </div>
        <button
          onClick={onClose}
          className="p-1 rounded-md hover:bg-surface-2 text-text-tertiary hover:text-text-primary transition-colors"
          aria-label="Close documents panel"
        >
          <X className="w-4 h-4" />
        </button>
      </div>

      <div className="p-4 border-b border-border/80 space-y-4">
        <div className="grid grid-cols-3 gap-2">
          <div className="rounded-xl border border-border bg-surface-2/60 px-3 py-3">
            <p className="text-[11px] uppercase tracking-wider text-text-tertiary">Ready</p>
            <p className="text-lg font-semibold text-text-primary mt-1">{readyDocuments.length}</p>
          </div>
          <div className="rounded-xl border border-border bg-surface-2/60 px-3 py-3">
            <p className="text-[11px] uppercase tracking-wider text-text-tertiary">Queue</p>
            <p className="text-lg font-semibold text-text-primary mt-1">{inFlightDocuments.length}</p>
          </div>
          <div className="rounded-xl border border-border bg-surface-2/60 px-3 py-3">
            <p className="text-[11px] uppercase tracking-wider text-text-tertiary">Chunks</p>
            <p className="text-lg font-semibold text-text-primary mt-1">{totalChunks}</p>
          </div>
        </div>

        <div
          {...getRootProps()}
          className={`relative overflow-hidden border rounded-2xl p-5 text-left cursor-pointer transition-all duration-200 ${
            isDragActive || dragActive
              ? "border-accent bg-[linear-gradient(135deg,rgba(102,126,234,0.14),rgba(118,75,162,0.08))]"
              : "border-border bg-[linear-gradient(180deg,rgba(255,255,255,0.02),rgba(255,255,255,0.01))] hover:border-border-hover hover:bg-surface-2/40"
          }`}
        >
          <input {...getInputProps()} />
          <div className="absolute inset-0 pointer-events-none bg-[radial-gradient(circle_at_top_right,rgba(102,126,234,0.16),transparent_40%)]" />
          <div className="relative flex items-start gap-3">
            <div className="w-10 h-10 rounded-2xl bg-gradient-to-br from-accent to-accent-2 flex items-center justify-center shadow-lg shadow-accent/20">
              <Upload className="w-5 h-5 text-white" />
            </div>
            <div className="min-w-0">
              <p className="text-sm font-semibold text-text-primary">
                {isDragActive ? "Drop files to start indexing" : "Drop files or browse your device"}
              </p>
              <p className="text-xs text-text-secondary mt-1 leading-relaxed">
                DocPilot parses the file, chunks the text, and adds it to the retrieval index.
              </p>
              <div className="flex flex-wrap gap-1.5 mt-3 text-[11px] text-text-tertiary">
                {["PDF", "TXT", "MD", "DOCX", "Up to 50MB"].map((tag) => (
                  <span key={tag} className="px-2 py-1 rounded-full border border-border bg-surface/70">
                    {tag}
                  </span>
                ))}
              </div>
            </div>
          </div>
        </div>

        {notice && (
          <div className="rounded-xl border border-warning/20 bg-warning/10 px-3 py-2.5 text-xs text-warning leading-relaxed">
            {notice}
          </div>
        )}

        <div className="flex items-center justify-between gap-3 rounded-xl border border-border bg-surface-2/50 px-3 py-2.5">
          <div className="min-w-0">
            <p className="text-xs font-medium text-text-primary">Index status</p>
            <p className="text-[11px] text-text-tertiary mt-0.5">
              {isSyncing ? "Refreshing the indexed library..." : "Library synced with the backend index."}
            </p>
          </div>
          <button
            type="button"
            onClick={() => void syncIndexedDocuments()}
            className="p-2 rounded-lg text-text-tertiary hover:text-text-primary hover:bg-surface-3 transition-colors"
            aria-label="Refresh document list"
          >
            <RefreshCw className={`w-4 h-4 ${isSyncing ? "animate-spin" : ""}`} />
          </button>
        </div>
      </div>

      <div className="flex-1 overflow-y-auto px-4 py-4 space-y-5">
        {readyDocuments.length > 0 && onPromptSelect && (
          <div className="rounded-2xl border border-accent/20 bg-[linear-gradient(135deg,rgba(102,126,234,0.12),rgba(118,75,162,0.07))] px-4 py-3">
            <div className="flex items-start gap-3">
              <div className="w-9 h-9 rounded-2xl bg-white/8 flex items-center justify-center flex-shrink-0">
                <Sparkles className="w-4 h-4 text-accent" />
              </div>
              <div className="min-w-0">
                <p className="text-sm font-semibold text-text-primary">Ready to ask something?</p>
                <p className="text-xs text-text-secondary mt-1 leading-relaxed">
                  {focusedDocumentNames.length > 0
                    ? `Chat will focus on ${focusedDocumentNames.length} selected document${
                        focusedDocumentNames.length > 1 ? "s" : ""
                      }.`
                    : "Your indexed documents are live. Start with a summary or pull out the key facts."}
                </p>
                <div className="flex flex-wrap gap-2 mt-3">
                  <button
                    type="button"
                    onClick={() => onPromptSelect("Summarize the key points from my uploaded documents")}
                    className="inline-flex items-center gap-1.5 px-3 py-2 rounded-lg text-xs font-medium bg-white/8 text-text-primary hover:bg-white/12 transition-colors"
                  >
                    Summarize documents
                    <ArrowUpRight className="w-3.5 h-3.5" />
                  </button>
                  <button
                    type="button"
                    onClick={() =>
                      onPromptSelect("List the most important facts and metrics from my uploaded documents")
                    }
                    className="inline-flex items-center gap-1.5 px-3 py-2 rounded-lg text-xs font-medium border border-white/10 text-text-secondary hover:text-text-primary hover:bg-white/6 transition-colors"
                  >
                    Pull key facts
                  </button>
                </div>
              </div>
            </div>
          </div>
        )}

        {inFlightDocuments.length > 0 && (
          <section className="space-y-2">
            <div className="flex items-center justify-between">
              <h4 className="text-xs font-medium uppercase tracking-wider text-text-tertiary">
                Upload Queue
              </h4>
              <span className="text-[11px] text-text-tertiary">{inFlightDocuments.length} active</span>
            </div>
            <AnimatePresence initial={false}>
              {inFlightDocuments.map((doc) => (
                <DocumentCard key={doc.id} doc={doc} />
              ))}
            </AnimatePresence>
          </section>
        )}

        {readyDocuments.length > 0 && (
          <section className="space-y-2">
            <div className="flex items-center justify-between gap-3">
              <div>
                <h4 className="text-xs font-medium uppercase tracking-wider text-text-tertiary">
                  Ready For Chat
                </h4>
                <p className="text-[11px] text-text-tertiary mt-1">
                  Select one or more documents to focus chat. Leave all unselected to search everything.
                </p>
              </div>
              <div className="flex items-center gap-2">
                {focusedDocumentNames.length > 0 && (
                  <button
                    type="button"
                    onClick={clearFocusedDocuments}
                    className="px-2.5 py-1 rounded-md border border-border text-[11px] text-text-secondary hover:text-text-primary hover:bg-surface transition-colors"
                  >
                    Search all
                  </button>
                )}
                <span className="text-[11px] text-text-tertiary">{readyDocuments.length} indexed</span>
              </div>
            </div>
            <AnimatePresence initial={false}>
              {readyDocuments.map((doc) => (
                <DocumentCard
                  key={doc.id}
                  doc={doc}
                  highlighted={focusedNamesSet.has(doc.name.trim().toLowerCase())}
                  action={
                    <>
                      <button
                        type="button"
                        onClick={() => toggleFocusedDocument(doc.name)}
                        className={`p-1.5 rounded-lg transition-colors ${
                          focusedNamesSet.has(doc.name.trim().toLowerCase())
                            ? "bg-accent/15 text-accent"
                            : "text-text-tertiary hover:text-text-primary hover:bg-surface"
                        }`}
                        aria-label={`Focus chat on ${doc.name}`}
                        title="Focus chat on this document"
                      >
                        <Crosshair className="w-3.5 h-3.5" />
                      </button>
                      <button
                        type="button"
                        onClick={async () => {
                          if (!confirm(`Delete "${doc.name}" from the index?`)) return;
                          try {
                            await deleteDocument(doc.name);
                            removeDocumentsByName(doc.name);
                            setNotice(null);
                          } catch (err: any) {
                            setNotice(`Failed to delete: ${err.message}`);
                          }
                        }}
                        className="p-1.5 rounded-lg text-text-tertiary hover:text-error hover:bg-surface transition-colors"
                        aria-label={`Delete ${doc.name}`}
                      >
                        <Trash2 className="w-3.5 h-3.5" />
                      </button>
                    </>
                  }
                />
              ))}
            </AnimatePresence>
          </section>
        )}

        {failedDocuments.length > 0 && (
          <section className="space-y-2">
            <div className="flex items-center justify-between">
              <h4 className="text-xs font-medium uppercase tracking-wider text-text-tertiary">
                Needs Attention
              </h4>
              <span className="text-[11px] text-error">{failedDocuments.length} failed</span>
            </div>
            <AnimatePresence initial={false}>
              {failedDocuments.map((doc) => (
                <DocumentCard
                  key={doc.id}
                  doc={doc}
                  action={
                    <button
                      type="button"
                      onClick={() => removeDocument(doc.id)}
                      className="p-1.5 rounded-lg text-text-tertiary hover:text-text-primary hover:bg-surface transition-colors"
                      aria-label={`Dismiss ${doc.name}`}
                    >
                      <X className="w-3.5 h-3.5" />
                    </button>
                  }
                />
              ))}
            </AnimatePresence>
            <p className="text-[11px] text-text-tertiary leading-relaxed">
              Re-add a failed file from your device to retry it. The browser does not keep the original file contents after a failed upload.
            </p>
          </section>
        )}

        {visibleDocuments.length === 0 && (
          <div className="text-center py-10">
            <File className="w-10 h-10 mx-auto text-text-tertiary/30 mb-3" />
            <p className="text-sm text-text-secondary">No documents indexed yet</p>
            <p className="text-xs text-text-tertiary mt-1">
              Upload a file to start building a searchable document library.
            </p>
          </div>
        )}
      </div>
    </motion.div>
  );
}
