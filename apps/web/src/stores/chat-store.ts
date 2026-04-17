import { create } from "zustand";
import { persist } from "zustand/middleware";

export interface Message {
  id: string;
  role: "user" | "assistant";
  content: string;
  citations?: Citation[];
  trace?: TraceStep[];
  timestamp: number;
}

export interface Citation {
  id: string;
  documentName: string;
  chunkText: string;
  page?: number;
  score: number;
}

export interface TraceStep {
  step: string;
  detail: string;
  duration_ms?: number;
}

export interface ChatSession {
  id: string;
  title: string;
  messages: Message[];
  createdAt: number;
  updatedAt: number;
}

interface ChatState {
  sessions: ChatSession[];
  activeSessionId: string | null;
  isStreaming: boolean;
  hasHydrated: boolean;

  // Actions
  createSession: () => string;
  setActiveSession: (id: string) => void;
  addMessage: (sessionId: string, message: Message) => void;
  updateMessage: (sessionId: string, messageId: string, updates: Partial<Message>) => void;
  setIsStreaming: (streaming: boolean) => void;
  deleteSession: (id: string) => void;
  getActiveSession: () => ChatSession | undefined;
}

export const useChatStore = create<ChatState>()(
  persist(
    (set, get) => ({
      sessions: [],
      activeSessionId: null,
      isStreaming: false,
      hasHydrated: false,

      createSession: () => {
        const id = crypto.randomUUID();
        const session: ChatSession = {
          id,
          title: "New conversation",
          messages: [],
          createdAt: Date.now(),
          updatedAt: Date.now(),
        };
        set((state) => ({
          sessions: [session, ...state.sessions],
          activeSessionId: id,
        }));
        return id;
      },

      setActiveSession: (id) => set({ activeSessionId: id }),

      addMessage: (sessionId, message) =>
        set((state) => ({
          sessions: state.sessions.map((s) => {
            if (s.id !== sessionId) return s;
            const updated = {
              ...s,
              messages: [...s.messages, message],
              updatedAt: Date.now(),
            };
            // Auto-title from first user message
            if (
              s.title === "New conversation" &&
              message.role === "user"
            ) {
              updated.title =
                message.content.slice(0, 50) +
                (message.content.length > 50 ? "..." : "");
            }
            return updated;
          }),
        })),

      updateMessage: (sessionId, messageId, updates) =>
        set((state) => ({
          sessions: state.sessions.map((s) => {
            if (s.id !== sessionId) return s;
            const msgs = s.messages.map((message) =>
              message.id === messageId ? { ...message, ...updates } : message
            );
            return { ...s, messages: msgs, updatedAt: Date.now() };
          }),
        })),

      setIsStreaming: (streaming) => set({ isStreaming: streaming }),

      deleteSession: (id) =>
        set((state) => {
          const nextSessions = state.sessions.filter((s) => s.id !== id);
          return {
            sessions: nextSessions,
            activeSessionId:
              state.activeSessionId === id
                ? nextSessions[0]?.id ?? null
                : state.activeSessionId,
          };
        }),

      getActiveSession: () => {
        const { sessions, activeSessionId } = get();
        return sessions.find((s) => s.id === activeSessionId);
      },
    }),
    {
      name: "docpilot-chat",
      onRehydrateStorage: () => (state) => {
        state?.setIsStreaming(false);
        useChatStore.setState({ hasHydrated: true });
      },
      partialize: (state) => ({
        sessions: state.sessions,
        activeSessionId: state.activeSessionId,
      }),
    }
  )
);
