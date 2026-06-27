"use client";

import { motion } from "framer-motion";

const BIO = [
  `I grew up in South Africa, where understanding people came before understanding systems — and that instinct has never left me. Whether it was navigating different communities, working across cultures, or just being curious about why people make the decisions they do, I've always been drawn to the human side of things. That orientation is what eventually led me to data: not as an end in itself, but as a way of making sense of the world and building things that matter to real people.`,
  `Now at UMass Amherst studying Data Science and ML Engineering, I'm building the technical depth to match that curiosity. My goal is to work as an AI/ML Engineer — designing and deploying models that are well-engineered end to end, from experimentation to cloud-hosted production systems managed through proper CI/CD pipelines. I care about the full arc, not just the model.`,
];

const SKILLS = [
  "Python",
  "PyTorch",
  "TensorFlow / Keras",
  "scikit-learn",
  "pandas / NumPy",
  "SQL",
  "Docker",
  "AWS",
  "CI/CD",
  "Git",
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

        {BIO.map((para, i) => (
          <motion.p
            key={i}
            className="mb-6 text-lg leading-relaxed text-text-primary"
            initial={{ opacity: 0, y: 16 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true, margin: "-10%" }}
            transition={{ duration: 0.7, ease: easeOutExpo, delay: 0.1 + i * 0.08 }}
          >
            {para}
          </motion.p>
        ))}

        <motion.p
          className="mt-4 text-sm tracking-wide uppercase text-text-secondary"
          initial={{ opacity: 0, y: 16 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true, margin: "-10%" }}
          transition={{ duration: 0.7, ease: easeOutExpo, delay: 0.2 }}
        >
          {SKILLS.join(", ")}
        </motion.p>
      </div>
    </section>
  );
}
