"use client";

import { motion, type Variants } from "framer-motion";

const BIO = `I'm Kellan McIntosh, a Data Science and Machine Learning Engineering student at the University of Massachusetts Amherst. Originally from South Africa, I came to UMass to build expertise at the intersection of statistics, systems, and intelligent software. My goal is to work as an AI/ML Engineer or Data Scientist — building models that move beyond notebooks and into production. I care about work that's rigorous, interpretable, and actually ships.`;

const SKILLS = [
  { label: "Python", abbr: "Py" },
  { label: "PyTorch", abbr: "PT" },
  { label: "TensorFlow / Keras", abbr: "TF" },
  { label: "scikit-learn", abbr: "sk" },
  { label: "pandas / NumPy", abbr: "pd" },
  { label: "SQL", abbr: "SQL" },
  { label: "Git", abbr: "Git" },
  { label: "Jupyter / Colab", abbr: "Jn" },
  { label: "Statistics & Probability", abbr: "Σ" },
  { label: "Next.js", abbr: "NX" },
];

const spring = { type: "spring", stiffness: 260, damping: 20 } as const;

const skillContainer: Variants = {
  hidden: {},
  show: { transition: { staggerChildren: 0.07 } },
};

const skillItem: Variants = {
  hidden: { opacity: 0, y: 12 },
  show: { opacity: 1, y: 0, transition: spring },
};

export default function AboutSection() {
  return (
    <section id="about" className="border-t border-border py-24">
      <div className="mx-auto max-w-6xl px-6">
        <motion.h2
          initial={{ opacity: 0, y: 16 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true, margin: "-10%" }}
          transition={spring}
          className="mb-12 text-3xl font-bold text-text-primary"
        >
          About
        </motion.h2>
        <div className="grid gap-16 md:grid-cols-2">
          {/* Bio */}
          <motion.p
            className="text-base leading-relaxed text-text-secondary"
            initial={{ opacity: 0, x: -20 }}
            whileInView={{ opacity: 1, x: 0 }}
            viewport={{ once: true, margin: "-10%" }}
            transition={spring}
          >
            {BIO}
          </motion.p>

          {/* Skills grid */}
          <motion.div
            variants={skillContainer}
            initial="hidden"
            whileInView="show"
            viewport={{ once: true, margin: "-10%" }}
            className="grid grid-cols-2 gap-3"
          >
            {SKILLS.map(({ label, abbr }) => (
              <motion.div
                key={label}
                variants={skillItem}
                className="flex items-center gap-3 rounded-lg border border-border bg-surface p-3"
              >
                <span className="flex h-8 w-8 shrink-0 items-center justify-center rounded-md border border-accent/20 bg-accent/10 font-mono text-xs font-semibold text-accent">
                  {abbr}
                </span>
                <span className="text-sm text-text-secondary">{label}</span>
              </motion.div>
            ))}
          </motion.div>
        </div>
      </div>
    </section>
  );
}
