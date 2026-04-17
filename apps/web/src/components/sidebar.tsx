"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import {
  MessageSquare,
  Upload,
  BarChart3,
  Settings,
  Plus,
  Trash2,
  PanelLeftClose,
  PanelLeft,
  FileText,
  Sparkles,
} from "lucide-react";
import { useChatStore } from "@/stores/chat-store";

const navItems = [
  { href: "/chat", label: "Chat", icon: MessageSquare },
  { href: "/eval", label: "Evaluation", icon: BarChart3 },
  { href: "/settings", label: "Settings", icon: Settings },
];

export function Sidebar() {
  const pathname = usePathname();
  const [collapsed, setCollapsed] = useState(false);
  const { sessions, activeSessionId, createSession, setActiveSession, deleteSession } =
    useChatStore();

  return (
    <motion.aside
      initial={false}
      animate={{ width: collapsed ? 64 : 280 }}
      transition={{ duration: 0.2, ease: "easeInOut" }}
      className="h-screen flex flex-col border-r border-border bg-surface sticky top-0 z-30 overflow-hidden"
    >
      {/* Header */}
      <div className="flex items-center justify-between h-14 px-3 border-b border-border flex-shrink-0">
        <AnimatePresence mode="wait">
          {!collapsed && (
            <motion.div
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
              className="flex items-center gap-2"
            >
              <div className="w-7 h-7 rounded-lg bg-gradient-to-br from-accent to-accent-2 flex items-center justify-center">
                <Sparkles className="w-4 h-4 text-white" />
              </div>
              <span className="font-semibold text-sm text-text-primary tracking-tight">
                DocPilot
              </span>
            </motion.div>
          )}
        </AnimatePresence>
        <button
          onClick={() => setCollapsed(!collapsed)}
          className="p-1.5 rounded-md hover:bg-surface-2 text-text-secondary hover:text-text-primary transition-colors"
          aria-label={collapsed ? "Expand sidebar" : "Collapse sidebar"}
        >
          {collapsed ? <PanelLeft className="w-4 h-4" /> : <PanelLeftClose className="w-4 h-4" />}
        </button>
      </div>

      {/* New Chat Button */}
      <div className="px-3 py-3 flex-shrink-0">
        <button
          onClick={() => createSession()}
          className="w-full flex items-center gap-2 px-3 py-2 rounded-lg text-sm font-medium
            bg-gradient-to-r from-accent/10 to-accent-2/10 border border-accent/20
            hover:from-accent/20 hover:to-accent-2/20 hover:border-accent/30
            text-text-primary transition-all duration-200"
        >
          <Plus className="w-4 h-4 flex-shrink-0" />
          {!collapsed && <span>New Chat</span>}
        </button>
      </div>

      {/* Chat Sessions */}
      {!collapsed && (
        <div className="flex-1 overflow-y-auto px-2 space-y-0.5 pb-2">
          <p className="px-2 py-1.5 text-[11px] font-medium uppercase tracking-wider text-text-tertiary">
            Recent
          </p>
          {sessions.map((session) => (
            <div
              key={session.id}
              className={`group flex items-center gap-2 px-2.5 py-2 rounded-lg cursor-pointer text-sm transition-all duration-150
                ${
                  session.id === activeSessionId
                    ? "bg-surface-2 text-text-primary"
                    : "text-text-secondary hover:bg-surface-2/50 hover:text-text-primary"
                }`}
            >
              <Link
                href="/chat"
                onClick={() => setActiveSession(session.id)}
                className="flex items-center gap-2 flex-1 min-w-0"
              >
                <FileText className="w-3.5 h-3.5 flex-shrink-0 opacity-50" />
                <span className="truncate">{session.title}</span>
              </Link>
              <button
                onClick={(e) => {
                  e.stopPropagation();
                  deleteSession(session.id);
                }}
                className="opacity-0 group-hover:opacity-100 p-1 hover:bg-error/10 hover:text-error rounded transition-all"
                aria-label="Delete session"
              >
                <Trash2 className="w-3 h-3" />
              </button>
            </div>
          ))}
        </div>
      )}

      {/* Navigation */}
      <nav className="mt-auto border-t border-border px-2 py-2 flex-shrink-0 space-y-0.5">
        {navItems.map((item) => {
          const isActive = pathname === item.href;
          const Icon = item.icon;
          return (
            <Link
              key={item.href}
              href={item.href}
              className={`flex items-center gap-2.5 px-3 py-2 rounded-lg text-sm font-medium transition-all duration-150
                ${
                  isActive
                    ? "bg-accent/10 text-accent border border-accent/20"
                    : "text-text-secondary hover:bg-surface-2 hover:text-text-primary"
                }
                ${collapsed ? "justify-center" : ""}`}
            >
              <Icon className="w-4 h-4 flex-shrink-0" />
              {!collapsed && <span>{item.label}</span>}
            </Link>
          );
        })}
      </nav>
    </motion.aside>
  );
}
