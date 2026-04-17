import Link from "next/link";
import {
  Sparkles,
  ArrowRight,
  FileSearch,
  Brain,
  Zap,
  Shield,
  GitBranch,
  BarChart3,
} from "lucide-react";

const features = [
  {
    icon: FileSearch,
    title: "Hybrid Retrieval",
    description: "Dense + sparse vector search with Qdrant for maximum recall and precision.",
  },
  {
    icon: Brain,
    title: "Agentic RAG",
    description: "LangGraph-powered agent decides when to retrieve, rewrite, or answer directly.",
  },
  {
    icon: Zap,
    title: "Streaming Answers",
    description: "Token-by-token streaming via Groq for instant, low-latency responses.",
  },
  {
    icon: Shield,
    title: "Grounded Citations",
    description: "Every answer backed by source references you can verify and expand.",
  },
  {
    icon: GitBranch,
    title: "Query Rewriting",
    description: "Automatic query refinement when initial retrieval doesn't match.",
  },
  {
    icon: BarChart3,
    title: "Evaluation Dashboard",
    description: "Track hit@k, citation coverage, groundedness, and latency metrics.",
  },
];

export default function LandingPage() {
  return (
    <div className="min-h-screen flex flex-col">
      {/* Hero Section */}
      <section className="relative flex-1 flex items-center justify-center overflow-hidden px-6 py-24">
        {/* Background Effects */}
        <div className="absolute inset-0 pointer-events-none">
          <div className="absolute top-1/4 left-1/4 w-96 h-96 rounded-full bg-accent/5 blur-[120px] animate-float" />
          <div
            className="absolute bottom-1/4 right-1/4 w-80 h-80 rounded-full bg-accent-2/5 blur-[100px] animate-float"
            style={{ animationDelay: "3s" }}
          />
          <div className="absolute inset-0 bg-[radial-gradient(ellipse_at_center,transparent_20%,var(--background)_70%)]" />
          {/* Grid pattern */}
          <div
            className="absolute inset-0 opacity-[0.03]"
            style={{
              backgroundImage:
                "linear-gradient(rgba(255,255,255,0.1) 1px, transparent 1px), linear-gradient(90deg, rgba(255,255,255,0.1) 1px, transparent 1px)",
              backgroundSize: "64px 64px",
            }}
          />
        </div>

        <div className="relative max-w-4xl mx-auto text-center z-10">
          {/* Badge */}
          <div className="inline-flex items-center gap-2 px-3 py-1.5 rounded-full border border-accent/20 bg-accent/5 text-accent text-xs font-medium mb-8">
            <Sparkles className="w-3.5 h-3.5" />
            Powered by LangGraph + Qdrant + Groq
          </div>

          {/* Heading */}
          <h1 className="text-5xl md:text-7xl font-bold tracking-tight leading-[1.1] mb-6">
            <span className="text-text-primary">Your documents.</span>
            <br />
            <span className="gradient-text">Intelligent answers.</span>
          </h1>

          <p className="text-lg md:text-xl text-text-secondary max-w-2xl mx-auto mb-10 leading-relaxed">
            Upload your documents and get citation-backed answers powered by
            an agentic RAG pipeline with hybrid retrieval, relevance grading,
            and query rewriting.
          </p>

          {/* CTA Buttons */}
          <div className="flex items-center justify-center gap-4">
            <Link
              href="/chat"
              className="group inline-flex items-center gap-2 px-6 py-3 rounded-xl
                bg-gradient-to-r from-accent to-accent-2 text-white font-semibold text-sm
                shadow-lg shadow-accent/20 hover:shadow-accent/30 hover:brightness-110
                transition-all duration-300"
            >
              Start chatting
              <ArrowRight className="w-4 h-4 group-hover:translate-x-0.5 transition-transform" />
            </Link>
            <Link
              href="/eval"
              className="inline-flex items-center gap-2 px-6 py-3 rounded-xl
                border border-border text-text-secondary font-semibold text-sm
                hover:bg-surface-2 hover:text-text-primary hover:border-border-hover
                transition-all duration-200"
            >
              View Evaluation
            </Link>
          </div>
        </div>
      </section>

      {/* Features Grid */}
      <section className="px-6 pb-24 relative z-10">
        <div className="max-w-5xl mx-auto">
          <div className="text-center mb-14">
            <h2 className="text-3xl font-bold text-text-primary mb-3">
              Production-Grade RAG Architecture
            </h2>
            <p className="text-text-secondary max-w-xl mx-auto">
              Not just retrieval — a full agentic pipeline that routes, retrieves, grades, rewrites, and generates.
            </p>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
            {features.map((feature, i) => {
              const Icon = feature.icon;
              return (
                <div
                  key={feature.title}
                  className="group p-5 rounded-xl border border-border bg-surface hover:bg-surface-2
                    hover:border-border-hover transition-all duration-300 animate-fade-in"
                  style={{ animationDelay: `${i * 80}ms` }}
                >
                  <div className="w-9 h-9 rounded-lg bg-accent/10 flex items-center justify-center mb-3 group-hover:bg-accent/15 transition-colors">
                    <Icon className="w-4.5 h-4.5 text-accent" />
                  </div>
                  <h3 className="text-sm font-semibold text-text-primary mb-1">{feature.title}</h3>
                  <p className="text-sm text-text-secondary leading-relaxed">{feature.description}</p>
                </div>
              );
            })}
          </div>
        </div>
      </section>

      {/* Architecture Diagram */}
      <section className="px-6 pb-24 relative z-10">
        <div className="max-w-3xl mx-auto">
          <div className="rounded-2xl border border-border bg-surface p-8 glass">
            <h3 className="text-lg font-semibold text-text-primary mb-6 text-center">
              Agentic RAG Flow
            </h3>
            <div className="flex flex-col items-center gap-3 text-sm font-mono">
              {[
                { label: "User Query", color: "text-text-primary" },
                { label: "↓", color: "text-text-tertiary" },
                { label: "Query Router", color: "text-accent" },
                { label: "↓", color: "text-text-tertiary" },
                { label: "Hybrid Retrieval (Dense + Sparse)", color: "text-accent" },
                { label: "↓", color: "text-text-tertiary" },
                { label: "Relevance Grader", color: "text-warning" },
                { label: "↓ pass / ↻ rewrite", color: "text-text-tertiary" },
                { label: "Grounded Answer Generator", color: "text-success" },
                { label: "↓", color: "text-text-tertiary" },
                { label: "Citation Formatter → Response", color: "text-text-primary" },
              ].map((step, i) => (
                <div key={i} className={`${step.color}`}>
                  {step.label.includes("↓") || step.label.includes("↻") ? (
                    <span className="text-lg">{step.label}</span>
                  ) : (
                    <div className="px-4 py-2 rounded-lg border border-border bg-surface-2 text-center">
                      {step.label}
                    </div>
                  )}
                </div>
              ))}
            </div>
          </div>
        </div>
      </section>

      {/* Footer */}
      <footer className="border-t border-border py-6 px-6 text-center">
        <p className="text-xs text-text-tertiary">
          Built with Next.js • FastAPI • LangGraph • Qdrant • Groq
        </p>
      </footer>
    </div>
  );
}
