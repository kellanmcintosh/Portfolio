"use client";

import { motion } from "framer-motion";

const EASE_OUT_EXPO = [0.16, 1, 0.3, 1] as const;
const EASE_TEXT_WIPE = [0.22, 1, 0.36, 1] as const;

export default function Hero() {
  const scrollToProjects = () => {
    document.getElementById("projects")?.scrollIntoView({ behavior: "smooth" });
  };

  return (
    <section className="relative flex min-h-screen flex-col justify-end overflow-hidden px-10 pb-[20vh] md:px-20">
      {/* Name */}
      <motion.h1
        className="font-bold leading-none tracking-tight text-text-primary"
        style={{
          fontFamily: "var(--font-playfair-display)",
          fontSize: "clamp(4.5rem, 11vw, 9rem)",
        }}
        initial={{ opacity: 0, y: 40 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 1, ease: EASE_OUT_EXPO }}
      >
        Kellan McIntosh
      </motion.h1>

      {/* Gold rule */}
      <motion.div
        className="mt-6 h-px bg-accent"
        style={{ originX: 0 }}
        initial={{ scaleX: 0 }}
        animate={{ scaleX: 1 }}
        transition={{ duration: 0.8, ease: EASE_OUT_EXPO, delay: 0.6 }}
      />

      {/* Subtitle */}
      <motion.p
        className="mt-4 text-sm uppercase tracking-[0.2em] text-text-secondary"
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ duration: 0.9, ease: EASE_TEXT_WIPE, delay: 0.9 }}
      >
        Data Scientist &amp; ML Engineer
      </motion.p>

      {/* CTAs */}
      <motion.div
        className="mt-8 flex flex-wrap gap-4"
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ duration: 0.9, ease: EASE_TEXT_WIPE, delay: 1.1 }}
      >
        <button
          onClick={scrollToProjects}
          className="border border-accent px-7 py-2.5 text-sm font-medium tracking-wide text-accent transition-colors hover:bg-accent hover:text-background"
        >
          View Projects
        </button>
        <a
          href="/resume.pdf"
          download
          className="border border-border px-7 py-2.5 text-sm font-medium tracking-wide text-text-secondary transition-colors hover:border-accent hover:text-accent"
        >
          Download Resume
        </a>
      </motion.div>
    </section>
  );
}
