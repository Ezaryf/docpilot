"use client";

import { useState, useRef, useEffect } from "react";
import { Send, Paperclip, Square } from "lucide-react";
import { motion } from "framer-motion";

interface ChatInputProps {
  onSend: (message: string) => void;
  onFileSelect?: () => void;
  isStreaming: boolean;
  onStop?: () => void;
}

export function ChatInput({ onSend, onFileSelect, isStreaming, onStop }: ChatInputProps) {
  const [value, setValue] = useState("");
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  useEffect(() => {
    if (textareaRef.current) {
      textareaRef.current.style.height = "auto";
      textareaRef.current.style.height =
        Math.min(textareaRef.current.scrollHeight, 200) + "px";
    }
  }, [value]);

  const handleSubmit = () => {
    if (!value.trim() || isStreaming) return;
    onSend(value.trim());
    setValue("");
    if (textareaRef.current) {
      textareaRef.current.style.height = "auto";
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSubmit();
    }
  };

  return (
    <div className="border-t border-border bg-surface/80 backdrop-blur-sm p-4">
      <div className="max-w-3xl mx-auto">
        <div
          className="flex items-end gap-2 rounded-xl border border-border bg-surface-2 
          focus-within:border-accent/40 focus-within:shadow-[0_0_0_1px_rgba(102,126,234,0.15)]
          transition-all duration-200 p-2"
        >
          {onFileSelect && (
            <button
              onClick={onFileSelect}
              className="p-2 rounded-lg text-text-tertiary hover:text-text-secondary hover:bg-surface-3 transition-colors"
              aria-label="Attach file"
            >
              <Paperclip className="w-4 h-4" />
            </button>
          )}

          <textarea
            ref={textareaRef}
            value={value}
            onChange={(e) => setValue(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder="Ask about your documents..."
            rows={1}
            className="flex-1 bg-transparent text-sm text-text-primary placeholder-text-tertiary
              resize-none outline-none py-1.5 px-1 max-h-[200px]"
          />

          {isStreaming ? (
            <motion.button
              initial={{ scale: 0.8 }}
              animate={{ scale: 1 }}
              onClick={onStop}
              className="p-2 rounded-lg bg-error/10 text-error hover:bg-error/20 transition-colors"
              aria-label="Stop generating"
            >
              <Square className="w-4 h-4" />
            </motion.button>
          ) : (
            <button
              onClick={handleSubmit}
              disabled={!value.trim()}
              className="p-2 rounded-lg bg-accent text-white hover:bg-accent/90 
                disabled:opacity-30 disabled:cursor-not-allowed transition-all duration-150"
              aria-label="Send message"
            >
              <Send className="w-4 h-4" />
            </button>
          )}
        </div>
        <p className="text-[11px] text-text-tertiary mt-2 text-center">
          DocPilot can make mistakes. Always verify important information with source documents.
        </p>
      </div>
    </div>
  );
}
