"use client";

import { motion } from "framer-motion";

export default function ResumeSection() {
  return (
    <section id="resume" className="border-t border-border py-24">
      <div className="mx-auto max-w-6xl px-6">
        <motion.div
          className="flex flex-col items-center text-center"
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true, margin: "-10%" }}
          transition={{ duration: 0.6, ease: "easeOut" }}
        >
          <h2 className="mb-4 text-3xl font-bold text-text-primary">Resume</h2>
          <p className="mb-10 text-text-secondary">
            Download my resume to learn more about my background and experience.
          </p>
          <a
            href="/resume.pdf"
            download
            className="group inline-flex items-center gap-3 rounded-lg bg-accent px-8 py-4 text-base font-semibold text-background transition-all duration-200 hover:bg-accent/90 hover:shadow-[0_0_32px_rgba(0,217,255,0.35)]"
          >
            <svg
              className="h-5 w-5 transition-transform duration-200 group-hover:translate-y-0.5"
              viewBox="0 0 24 24"
              fill="none"
              stroke="currentColor"
              strokeWidth="2.5"
              strokeLinecap="round"
              strokeLinejoin="round"
            >
              <path d="M12 3v13" />
              <path d="m5 13 7 7 7-7" />
              <path d="M3 21h18" />
            </svg>
            Download Resume
          </a>
        </motion.div>
      </div>
    </section>
  );
}
