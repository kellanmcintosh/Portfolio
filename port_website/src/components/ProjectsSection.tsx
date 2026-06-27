"use client";

import { motion } from "framer-motion";

type Project = {
  name: string;
  description: string;
  stack: string[];
  href: string;
  comingSoon: boolean;
};

const PROJECTS: Project[] = [
  {
    name: "Credit Risk Scorer",
    description: "Predicting credit default risk using gradient boosting",
    stack: ["Python", "XGBoost", "scikit-learn", "pandas"],
    href: "#",
    comingSoon: true,
  },
  {
    name: "MRI Tumor Detection",
    description: "Deep learning model for MRI brain tumor classification",
    stack: ["Python", "PyTorch", "CNN", "OpenCV"],
    href: "#",
    comingSoon: true,
  },
  {
    name: "Next Project",
    description: "Another project is on the way.",
    stack: [],
    href: "#",
    comingSoon: true,
  },
];

function ProjectCard({ project }: { project: Project }) {
  return (
    <motion.a
      href={project.href}
      whileHover={{ y: -6 }}
      transition={{ duration: 0.2, ease: "easeOut" }}
      className="group relative flex min-h-48 flex-col rounded-xl border border-border bg-surface p-6 transition-all duration-200 hover:border-accent hover:shadow-[0_0_28px_rgba(0,217,255,0.12)]"
    >
      {project.comingSoon && (
        <span className="absolute right-4 top-4 rounded-full bg-accent/10 px-2.5 py-1 text-xs font-medium text-accent">
          Coming Soon
        </span>
      )}

      <span className="absolute bottom-4 right-4 translate-x-1 text-accent opacity-0 transition-all duration-200 group-hover:translate-x-0 group-hover:opacity-100">
        →
      </span>

      <h3 className="mb-2 pr-24 text-lg font-semibold text-text-primary">
        {project.name}
      </h3>
      <p className="mb-4 flex-1 text-sm text-text-secondary">
        {project.description}
      </p>

      {project.stack.length > 0 && (
        <div className="flex flex-wrap gap-2">
          {project.stack.map((tag) => (
            <span
              key={tag}
              className="rounded-full border border-accent/20 bg-accent/5 px-2.5 py-0.5 text-xs text-accent"
            >
              {tag}
            </span>
          ))}
        </div>
      )}
    </motion.a>
  );
}

export default function ProjectsSection() {
  return (
    <section id="projects" className="border-t border-border py-24">
      <div className="mx-auto max-w-6xl px-6">
        <h2 className="mb-12 text-3xl font-bold text-text-primary">Projects</h2>
        <div className="grid gap-6 md:grid-cols-3">
          {PROJECTS.map((project) => (
            <ProjectCard key={project.name} project={project} />
          ))}
        </div>
      </div>
    </section>
  );
}
