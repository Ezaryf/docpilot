"use client";

import { motion } from "framer-motion";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import { User, Bot, ChevronDown, ChevronRight, ExternalLink, FileText } from "lucide-react";
import { useState } from "react";
import type { Message, Citation, TraceStep } from "@/stores/chat-store";

function CitationChip({ citation, index }: { citation: Citation; index: number }) {
  const [expanded, setExpanded] = useState(false);

  return (
    <div className="inline-block mr-1.5 mb-1.5">
      <button
        onClick={() => setExpanded(!expanded)}
        className="inline-flex items-center gap-1 px-2 py-0.5 text-xs rounded-md
          bg-accent/10 text-accent border border-accent/20 hover:bg-accent/20
          transition-all duration-150 font-medium"
      >
        <FileText className="w-3 h-3" />
        [{index + 1}] {citation.documentName}
        {citation.page && <span className="text-text-tertiary">p.{citation.page}</span>}
      </button>
      {expanded && (
        <motion.div
          initial={{ opacity: 0, height: 0 }}
          animate={{ opacity: 1, height: "auto" }}
          exit={{ opacity: 0, height: 0 }}
          className="mt-1 p-2.5 rounded-lg bg-surface-2 border border-border text-xs text-text-secondary leading-relaxed"
        >
          {citation.chunkText}
          <div className="mt-1 text-text-tertiary">
            Score: {(citation.score * 100).toFixed(1)}%
          </div>
        </motion.div>
      )}
    </div>
  );
}

function ReasoningTrace({ trace }: { trace: TraceStep[] }) {
  const [expanded, setExpanded] = useState(false);

  return (
    <div className="mt-2">
      <button
        onClick={() => setExpanded(!expanded)}
        className="flex items-center gap-1.5 text-xs text-text-tertiary hover:text-text-secondary transition-colors"
      >
        {expanded ? <ChevronDown className="w-3 h-3" /> : <ChevronRight className="w-3 h-3" />}
        <span>Reasoning trace ({trace.length} steps)</span>
      </button>
      {expanded && (
        <motion.div
          initial={{ opacity: 0, height: 0 }}
          animate={{ opacity: 1, height: "auto" }}
          className="mt-2 space-y-1.5 pl-3 border-l-2 border-border"
        >
          {trace.map((step, i) => (
            <div key={i} className="text-xs">
              <span className="text-accent font-medium">{step.step}</span>
              <span className="text-text-tertiary mx-1">→</span>
              <span className="text-text-secondary">{step.detail}</span>
              {step.duration_ms && (
                <span className="text-text-tertiary ml-1">({step.duration_ms}ms)</span>
              )}
            </div>
          ))}
        </motion.div>
      )}
    </div>
  );
}

export function ChatMessage({ message, isLast }: { message: Message; isLast: boolean }) {
  const isUser = message.role === "user";

  return (
    <motion.div
      initial={{ opacity: 0, y: 12 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.3 }}
      className={`flex gap-3 px-4 py-4 ${isUser ? "" : "bg-surface/50"}`}
    >
      {/* Avatar */}
      <div
        className={`w-7 h-7 rounded-lg flex items-center justify-center flex-shrink-0 mt-0.5
          ${
            isUser
              ? "bg-surface-2 border border-border"
              : "bg-gradient-to-br from-accent to-accent-2"
          }`}
      >
        {isUser ? (
          <User className="w-3.5 h-3.5 text-text-secondary" />
        ) : (
          <Bot className="w-3.5 h-3.5 text-white" />
        )}
      </div>

      {/* Content */}
      <div className="flex-1 min-w-0 space-y-1">
        <p className="text-xs font-medium text-text-tertiary">{isUser ? "You" : "DocPilot"}</p>

        <div className="prose-chat text-sm text-text-primary">
          <ReactMarkdown remarkPlugins={[remarkGfm]}>{message.content}</ReactMarkdown>
        </div>

        {/* Citations */}
        {message.citations && message.citations.length > 0 && (
          <div className="mt-3 pt-2 border-t border-border/50">
            <p className="text-[11px] font-medium uppercase tracking-wider text-text-tertiary mb-1.5">
              Sources
            </p>
            <div className="flex flex-wrap">
              {message.citations.map((c, i) => (
                <CitationChip key={c.id} citation={c} index={i} />
              ))}
            </div>
          </div>
        )}

        {/* Reasoning Trace */}
        {message.trace && message.trace.length > 0 && (
          <ReasoningTrace trace={message.trace} />
        )}
      </div>
    </motion.div>
  );
}

export function StreamingIndicator() {
  return (
    <div className="flex gap-3 px-4 py-4">
      <div className="w-7 h-7 rounded-lg bg-gradient-to-br from-accent to-accent-2 flex items-center justify-center flex-shrink-0">
        <Bot className="w-3.5 h-3.5 text-white" />
      </div>
      <div className="flex items-center gap-1 pt-2">
        <motion.div
          className="w-1.5 h-1.5 rounded-full bg-accent"
          animate={{ opacity: [0.3, 1, 0.3] }}
          transition={{ duration: 1.2, repeat: Infinity, delay: 0 }}
        />
        <motion.div
          className="w-1.5 h-1.5 rounded-full bg-accent"
          animate={{ opacity: [0.3, 1, 0.3] }}
          transition={{ duration: 1.2, repeat: Infinity, delay: 0.2 }}
        />
        <motion.div
          className="w-1.5 h-1.5 rounded-full bg-accent"
          animate={{ opacity: [0.3, 1, 0.3] }}
          transition={{ duration: 1.2, repeat: Infinity, delay: 0.4 }}
        />
      </div>
    </div>
  );
}
