"use client";

import { useState } from "react";
import { motion } from "framer-motion";
import {
  BarChart3,
  Target,
  Clock,
  FileCheck,
  TrendingUp,
  RefreshCw,
  AlertTriangle,
  CheckCircle2,
  ArrowLeft,
} from "lucide-react";
import Link from "next/link";
import { Sidebar } from "@/components/sidebar";

interface Metric {
  label: string;
  value: string;
  subtitle: string;
  icon: React.ElementType;
  color: string;
  bgColor: string;
}

const metrics: Metric[] = [
  {
    label: "Hit@5",
    value: "—",
    subtitle: "Retrieval accuracy",
    icon: Target,
    color: "text-accent",
    bgColor: "bg-accent/10",
  },
  {
    label: "Avg Latency",
    value: "—",
    subtitle: "End to end",
    icon: Clock,
    color: "text-warning",
    bgColor: "bg-warning/10",
  },
  {
    label: "Citation Coverage",
    value: "—",
    subtitle: "Answers with sources",
    icon: FileCheck,
    color: "text-success",
    bgColor: "bg-success/10",
  },
  {
    label: "Groundedness",
    value: "—",
    subtitle: "Answers from docs",
    icon: TrendingUp,
    color: "text-accent-2",
    bgColor: "bg-[rgba(118,75,162,0.1)]",
  },
];

interface EvalResult {
  query: string;
  relevant: boolean;
  rewritten: boolean;
  latency_ms: number;
  citations: number;
  score: number;
}

export default function EvalPage() {
  const [results, setResults] = useState<EvalResult[]>([]);
  const [running, setRunning] = useState(false);
  const [liveMetrics, setLiveMetrics] = useState(metrics);
  const [error, setError] = useState<string | null>(null);

  const runEval = async () => {
    setRunning(true);
    setError(null);
    try {
      const AI_URL = process.env.NEXT_PUBLIC_AI_SERVICE_URL || "http://localhost:8000";
      const res = await fetch(`${AI_URL}/api/eval`, { method: "POST" });

      if (!res.ok) {
        throw new Error(`Evaluation request failed with HTTP ${res.status}`);
      }

      const data = await res.json();
      if (data.error) {
        throw new Error(data.error);
      }
      if (!Array.isArray(data.results)) {
        throw new Error("Evaluation response did not include results.");
      }

      setResults(data.results);
      if (data.metrics) {
        setLiveMetrics((prev) =>
          prev.map((m) => {
            if (m.label === "Hit@5" && data.metrics.hit_at_5 !== undefined)
              return { ...m, value: `${(data.metrics.hit_at_5 * 100).toFixed(1)}%` };
            if (m.label === "Avg Latency" && data.metrics.avg_latency_ms !== undefined)
              return { ...m, value: `${data.metrics.avg_latency_ms.toFixed(0)}ms` };
            if (m.label === "Citation Coverage" && data.metrics.citation_coverage !== undefined)
              return { ...m, value: `${(data.metrics.citation_coverage * 100).toFixed(1)}%` };
            if (m.label === "Groundedness" && data.metrics.groundedness !== undefined)
              return { ...m, value: `${(data.metrics.groundedness * 100).toFixed(1)}%` };
            return m;
          })
        );
      }
    } catch (err) {
      console.error("Eval failed:", err);
      setError(err instanceof Error ? err.message : "Evaluation failed.");
    } finally {
      setRunning(false);
    }
  };

  return (
    <div className="flex h-screen overflow-hidden bg-background">
      <Sidebar />

      <div className="flex-1 overflow-y-auto">
        {/* Header */}
        <header className="h-14 flex items-center justify-between px-6 border-b border-border bg-surface/80 backdrop-blur-sm sticky top-0 z-10">
          <div className="flex items-center gap-3">
            <BarChart3 className="w-4 h-4 text-accent" />
            <h1 className="text-sm font-semibold text-text-primary">
              Evaluation Dashboard
            </h1>
          </div>
          <button
            onClick={runEval}
            disabled={running}
            className="flex items-center gap-2 px-4 py-2 rounded-lg text-sm font-medium
              bg-accent text-white hover:bg-accent/90 disabled:opacity-50
              transition-all duration-200"
          >
            <RefreshCw className={`w-3.5 h-3.5 ${running ? "animate-spin" : ""}`} />
            {running ? "Running..." : "Run Evaluation"}
          </button>
        </header>

        <div className="max-w-5xl mx-auto p-6 space-y-6">
          {/* Metrics Grid */}
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
            {liveMetrics.map((metric, i) => {
              const Icon = metric.icon;
              return (
                <motion.div
                  key={metric.label}
                  initial={{ opacity: 0, y: 12 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ delay: i * 0.08 }}
                  className="p-5 rounded-xl border border-border bg-surface hover:bg-surface-2 transition-colors"
                >
                  <div className="flex items-center justify-between mb-3">
                    <span className="text-xs font-medium text-text-tertiary uppercase tracking-wider">
                      {metric.label}
                    </span>
                    <div className={`w-8 h-8 rounded-lg ${metric.bgColor} flex items-center justify-center`}>
                      <Icon className={`w-4 h-4 ${metric.color}`} />
                    </div>
                  </div>
                  <p className={`text-2xl font-bold ${metric.color}`}>{metric.value}</p>
                  <p className="text-xs text-text-tertiary mt-1">{metric.subtitle}</p>
                </motion.div>
              );
            })}
          </div>

          {/* Results Table */}
          {error ? (
            <div className="rounded-xl border border-error/30 bg-error/10 p-5">
              <div className="flex items-start gap-3">
                <AlertTriangle className="w-5 h-5 text-error mt-0.5" />
                <div>
                  <h3 className="text-sm font-semibold text-error">Evaluation failed</h3>
                  <p className="text-sm text-text-secondary mt-1">{error}</p>
                </div>
              </div>
            </div>
          ) : results.length > 0 ? (
            <div className="rounded-xl border border-border bg-surface overflow-hidden">
              <div className="px-5 py-3 border-b border-border">
                <h3 className="text-sm font-semibold text-text-primary">
                  Evaluation Results ({results.length} queries)
                </h3>
              </div>
              <div className="overflow-x-auto">
                <table className="w-full text-sm">
                  <thead>
                    <tr className="border-b border-border text-left">
                      <th className="px-5 py-3 text-xs font-medium text-text-tertiary uppercase tracking-wider">Query</th>
                      <th className="px-5 py-3 text-xs font-medium text-text-tertiary uppercase tracking-wider">Relevant</th>
                      <th className="px-5 py-3 text-xs font-medium text-text-tertiary uppercase tracking-wider">Rewritten</th>
                      <th className="px-5 py-3 text-xs font-medium text-text-tertiary uppercase tracking-wider">Latency</th>
                      <th className="px-5 py-3 text-xs font-medium text-text-tertiary uppercase tracking-wider">Citations</th>
                      <th className="px-5 py-3 text-xs font-medium text-text-tertiary uppercase tracking-wider">Score</th>
                    </tr>
                  </thead>
                  <tbody>
                    {results.map((r, i) => (
                      <tr key={i} className="border-b border-border/50 hover:bg-surface-2/50 transition-colors">
                        <td className="px-5 py-3 text-text-primary max-w-xs truncate">{r.query}</td>
                        <td className="px-5 py-3">
                          {r.relevant ? (
                            <CheckCircle2 className="w-4 h-4 text-success" />
                          ) : (
                            <AlertTriangle className="w-4 h-4 text-warning" />
                          )}
                        </td>
                        <td className="px-5 py-3 text-text-secondary">{r.rewritten ? "Yes" : "No"}</td>
                        <td className="px-5 py-3 text-text-secondary">{r.latency_ms.toFixed(0)}ms</td>
                        <td className="px-5 py-3 text-text-secondary">{r.citations}</td>
                        <td className="px-5 py-3">
                          <span
                            className={`px-2 py-0.5 rounded-md text-xs font-medium ${
                              r.score >= 0.8
                                ? "bg-success/10 text-success"
                                : r.score >= 0.5
                                ? "bg-warning/10 text-warning"
                                : "bg-error/10 text-error"
                            }`}
                          >
                            {(r.score * 100).toFixed(0)}%
                          </span>
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          ) : (
            <div className="rounded-xl border border-border bg-surface p-12 text-center">
              <BarChart3 className="w-10 h-10 mx-auto text-text-tertiary/30 mb-3" />
              <h3 className="text-sm font-medium text-text-secondary mb-1">No evaluation results yet</h3>
              <p className="text-xs text-text-tertiary">
                Upload documents and click &quot;Run Evaluation&quot; to see retrieval quality metrics.
              </p>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
