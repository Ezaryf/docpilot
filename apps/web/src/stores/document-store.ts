import { create } from "zustand";
import { persist } from "zustand/middleware";

export interface UploadedDocument {
  id: string;
  name: string;
  size: number;
  type: string;
  chunks: number;
  uploadedAt: number;
  status: "queued" | "uploading" | "processing" | "ready" | "error";
  progress: number;
  source: "local" | "server";
  stageLabel?: string;
  error?: string;
}

interface DocumentState {
  documents: UploadedDocument[];
  focusedDocumentNames: string[];
  hasHydrated: boolean;
  addDocument: (doc: UploadedDocument) => void;
  updateDocument: (id: string, updates: Partial<UploadedDocument>) => void;
  removeDocument: (id: string) => void;
  removeDocumentsByName: (name: string) => void;
  upsertDocuments: (docs: UploadedDocument[]) => void;
  toggleFocusedDocument: (name: string) => void;
  clearFocusedDocuments: () => void;
  setFocusedDocuments: (names: string[]) => void;
}

function getDocumentKey(doc: UploadedDocument) {
  return doc.name.trim().toLowerCase();
}

function dedupeDocuments(documents: UploadedDocument[]) {
  const byName = new Map<string, UploadedDocument>();

  for (const doc of documents) {
    const key = getDocumentKey(doc);
    const existing = byName.get(key);

    if (!existing) {
      byName.set(key, doc);
      continue;
    }

    const shouldPreferDoc =
      existing.status !== "ready" && doc.status === "ready"
        ? true
        : existing.source !== "server" && doc.source === "server"
        ? true
        : doc.uploadedAt > existing.uploadedAt;

    byName.set(
      key,
      shouldPreferDoc
        ? { ...existing, ...doc }
        : { ...doc, ...existing }
    );
  }

  return [...byName.values()].sort((a, b) => b.uploadedAt - a.uploadedAt);
}

export const useDocumentStore = create<DocumentState>()(
  persist(
    (set) => ({
      documents: [],
      focusedDocumentNames: [],
      hasHydrated: false,

      addDocument: (doc) =>
        set((state) => ({
          documents: dedupeDocuments([doc, ...state.documents]),
        })),

      updateDocument: (id, updates) =>
        set((state) => {
          const updatedDocuments = state.documents.map((d) =>
            d.id === id ? { ...d, ...updates } : d
          );

          return {
            documents: dedupeDocuments(updatedDocuments),
          };
        }),

      removeDocument: (id) =>
        set((state) => ({
          documents: state.documents.filter((d) => d.id !== id),
        })),

      removeDocumentsByName: (name) =>
        set((state) => ({
          documents: state.documents.filter(
            (d) => d.name.trim().toLowerCase() !== name.trim().toLowerCase()
          ),
          focusedDocumentNames: state.focusedDocumentNames.filter(
            (docName) => docName.trim().toLowerCase() !== name.trim().toLowerCase()
          ),
        })),

      upsertDocuments: (docs) =>
        set((state) => {
          return {
            documents: dedupeDocuments([...state.documents, ...docs]),
          };
        }),

      toggleFocusedDocument: (name) =>
        set((state) => {
          const normalizedName = name.trim().toLowerCase();
          const isFocused = state.focusedDocumentNames.some(
            (docName) => docName.trim().toLowerCase() === normalizedName
          );

          return {
            focusedDocumentNames: isFocused
              ? state.focusedDocumentNames.filter(
                  (docName) => docName.trim().toLowerCase() !== normalizedName
                )
              : [...state.focusedDocumentNames, name],
          };
        }),

      clearFocusedDocuments: () => set({ focusedDocumentNames: [] }),

      setFocusedDocuments: (names) => set({ focusedDocumentNames: names }),
    }),
    {
      name: "docpilot-documents",
      onRehydrateStorage: () => () => {
        useDocumentStore.setState({ hasHydrated: true });
      },
    }
  )
);
