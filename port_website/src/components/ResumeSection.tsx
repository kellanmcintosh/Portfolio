"use client";

import dynamic from "next/dynamic";
import { motion } from "framer-motion";

const ResumePDFViewer = dynamic(() => import("./ResumePDFViewer"), {
  ssr: false,
  loading: () => (
    <div
      className="w-full animate-pulse bg-surface"
      style={{ aspectRatio: "8.5 / 11" }}
    />
  ),
});

const easeOutExpo = [0.16, 1, 0.3, 1] as const;

export default function ResumeSection() {
  return (
    <section id="resume" className="border-t border-border py-24">
      <div className="mx-auto max-w-4xl px-6">
        <h2 className="mb-4 text-5xl font-bold text-text-primary">
          <span className="text-accent">—</span> Resume
        </h2>

        <p className="mb-12 text-lg leading-relaxed text-text-secondary">
          A snapshot of my background, skills, and trajectory.
        </p>

        <motion.div
          initial={{ opacity: 0, y: 16 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.7, ease: easeOutExpo }}
        >
          <ResumePDFViewer />

          <div className="mt-8 flex items-center justify-between">
            <span className="text-xs uppercase tracking-widest text-text-secondary">
              Kellan McIntosh — Resume
            </span>
            <div className="flex items-center gap-6">
              <a
                href="/resume.pdf"
                target="_blank"
                rel="noopener noreferrer"
                className="text-sm text-text-secondary transition-colors duration-200 hover:text-text-primary"
              >
                Open in new tab ↗
              </a>
              <a
                href="/resume.pdf"
                download
                className="group inline-flex items-center gap-2 text-sm font-medium text-text-secondary transition-colors duration-200 hover:text-text-primary"
              >
                <svg
                  className="h-4 w-4 transition-transform duration-200 group-hover:translate-y-0.5"
                  viewBox="0 0 24 24"
                  fill="none"
                  stroke="currentColor"
                  strokeWidth="2"
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  aria-hidden="true"
                >
                  <path d="M12 3v13" />
                  <path d="m5 13 7 7 7-7" />
                  <path d="M3 21h18" />
                </svg>
                Download
              </a>
            </div>
          </div>
        </motion.div>
      </div>
    </section>
  );
}
