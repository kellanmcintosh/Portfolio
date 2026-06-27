"use client";

import { motion } from "framer-motion";

const BIO = `I'm Kellan McIntosh, a Data Science and Machine Learning Engineering student at the University of Massachusetts Amherst. Originally from South Africa, I came to UMass to build expertise at the intersection of statistics, systems, and intelligent software. My goal is to work as an AI/ML Engineer or Data Scientist — building models that move beyond notebooks and into production. I care about work that's rigorous, interpretable, and actually ships.`;

const SKILLS = [
  "Python",
  "PyTorch",
  "TensorFlow / Keras",
  "scikit-learn",
  "pandas / NumPy",
  "SQL",
  "Git",
  "Jupyter / Colab",
  "Statistics & Probability",
  "Next.js",
];

const easeOutExpo = [0.16, 1, 0.3, 1] as const;

export default function AboutSection() {
  return (
    <section id="about" className="border-t border-border py-24">
      <div className="mx-auto max-w-2xl px-6">
        <motion.h2
          initial={{ opacity: 0, y: 16 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true, margin: "-10%" }}
          transition={{ duration: 0.7, ease: easeOutExpo }}
          className="mb-12 font-playfair text-5xl font-bold text-text-primary"
        >
          <span className="text-accent">—</span> About
        </motion.h2>

        <motion.p
          className="mb-8 text-lg leading-relaxed text-text-primary"
          initial={{ opacity: 0, y: 16 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true, margin: "-10%" }}
          transition={{ duration: 0.7, ease: easeOutExpo, delay: 0.1 }}
        >
          {BIO}
        </motion.p>

        <motion.p
          className="mb-10 font-playfair text-xl italic text-text-secondary"
          initial={{ opacity: 0, y: 16 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true, margin: "-10%" }}
          transition={{ duration: 0.7, ease: easeOutExpo, delay: 0.2 }}
        >
          &ldquo;Building models that actually ship.&rdquo;
        </motion.p>

        <motion.p
          className="text-sm tracking-wide uppercase text-text-secondary"
          initial={{ opacity: 0, y: 16 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true, margin: "-10%" }}
          transition={{ duration: 0.7, ease: easeOutExpo, delay: 0.3 }}
        >
          {SKILLS.join(", ")}
        </motion.p>
      </div>
    </section>
  );
}
