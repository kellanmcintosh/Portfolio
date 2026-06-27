"use client";

import { motion } from "framer-motion";

type Project = {
  name: string;
  description: string;
  stack: string[];
  href: string;
  comingSoon: boolean;
  category: string;
};

const PROJECTS: Project[] = [
  {
    name: "Credit Risk Scorer",
    description: "Predicting credit default risk using gradient boosting",
    stack: ["Python", "XGBoost", "scikit-learn", "pandas"],
    href: "#",
    comingSoon: true,
    category: "Machine Learning",
  },
  {
    name: "MRI Tumor Detection",
    description: "Deep learning model for MRI brain tumor classification",
    stack: ["Python", "PyTorch", "CNN", "OpenCV"],
    href: "#",
    comingSoon: true,
    category: "Computer Vision",
  },
  {
    name: "Next Project",
    description: "Another project is on the way.",
    stack: [],
    href: "#",
    comingSoon: true,
    category: "In Progress",
  },
];

const easeOutExpo = [0.16, 1, 0.3, 1] as const;

function ProjectRow({ project, index }: { project: Project; index: number }) {
  const num = String(index + 1).padStart(2, "0");

  return (
    <div className="flex flex-col gap-6 border-t border-border py-12 md:flex-row md:gap-12 md:items-start">
      <motion.div
        initial={{ opacity: 0, x: -30 }}
        whileInView={{ opacity: 1, x: 0 }}
        viewport={{ once: true, margin: "-10%" }}
        transition={{ duration: 0.7, ease: easeOutExpo }}
        className="md:w-36 md:shrink-0 flex flex-col justify-between"
      >
        <span className="font-playfair text-7xl font-bold leading-none text-border select-none">
          {num}
        </span>
        <div className="mt-4">
          <div className="w-8 h-px bg-border mb-3" />
          <span className="text-xs tracking-widest uppercase text-text-secondary">
            {project.category}
          </span>
        </div>
      </motion.div>

      <motion.div
        initial={{ opacity: 0, x: 30 }}
        whileInView={{ opacity: 1, x: 0 }}
        viewport={{ once: true, margin: "-10%" }}
        transition={{ duration: 0.7, ease: easeOutExpo, delay: 0.15 }}
        className="flex-1"
      >
        <h3 className="font-playfair text-3xl font-bold text-text-primary mb-3">
          {project.name}
        </h3>
        <p className="text-base text-text-secondary leading-relaxed mb-4">
          {project.description}
          {project.comingSoon && (
            <em className="ml-2 text-text-secondary"> — coming soon</em>
          )}
        </p>
        {project.stack.length > 0 && (
          <p className="text-sm text-text-secondary mb-6">
            {project.stack.join(", ")}
          </p>
        )}
        {!project.comingSoon && (
          <a
            href={project.href}
            className="text-sm font-medium text-accent underline underline-offset-4 decoration-accent hover:opacity-70 transition-opacity"
          >
            View Project →
          </a>
        )}
      </motion.div>
    </div>
  );
}

export default function ProjectsSection() {
  return (
    <section id="projects" className="border-t border-border py-24">
      <div className="mx-auto max-w-6xl px-6">
        <motion.h2
          initial={{ opacity: 0, y: 16 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true, margin: "-10%" }}
          transition={{ duration: 0.7, ease: easeOutExpo }}
          className="mb-4 font-playfair text-5xl font-bold text-text-primary"
        >
          Projects
        </motion.h2>
        <div>
          {PROJECTS.map((project, i) => (
            <ProjectRow key={project.name} project={project} index={i} />
          ))}
        </div>
      </div>
    </section>
  );
}
