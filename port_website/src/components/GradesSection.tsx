"use client";

import { motion } from "framer-motion";

const COURSES = [
  { course: "Machine Learning", grade: "A" },
  { course: "Statistics", grade: "A" },
  { course: "Algorithms", grade: "A" },
  { course: "Simulation Modeling", grade: "A" },
];

const spring = { type: "spring", stiffness: 260, damping: 20 } as const;

export default function GradesSection() {
  return (
    <section id="grades" className="border-t border-border py-24">
      <div className="mx-auto max-w-6xl px-6">
        <motion.h2
          initial={{ opacity: 0, y: 16 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true, margin: "-10%" }}
          transition={spring}
          className="mb-12 text-3xl font-bold text-text-primary"
        >
          Grades
        </motion.h2>
        <div className="grid gap-16 md:grid-cols-2 md:items-center">
          {/* GPA callout */}
          <motion.div
            initial={{ opacity: 0, x: -20 }}
            whileInView={{ opacity: 1, x: 0 }}
            viewport={{ once: true, margin: "-10%" }}
            transition={spring}
          >
            <p className="text-8xl font-bold leading-none text-accent md:text-[7rem]">
              4.0
            </p>
            <p className="mt-2 text-3xl font-semibold text-text-primary">
              / 4.0
            </p>
            <p className="mt-4 text-sm text-text-secondary">
              Cumulative GPA · UMass Amherst
            </p>
          </motion.div>

          {/* Coursework table */}
          <motion.div
            initial={{ opacity: 0, x: 20 }}
            whileInView={{ opacity: 1, x: 0 }}
            viewport={{ once: true, margin: "-10%" }}
            transition={spring}
          >
            <p className="mb-4 text-xs font-semibold uppercase tracking-widest text-text-secondary">
              Relevant Coursework
            </p>
            <table className="w-full">
              <thead>
                <tr className="border-b border-border">
                  <th className="pb-3 text-left text-xs font-semibold uppercase tracking-widest text-text-secondary">
                    Course
                  </th>
                  <th className="pb-3 text-right text-xs font-semibold uppercase tracking-widest text-text-secondary">
                    Grade
                  </th>
                </tr>
              </thead>
              <tbody>
                {COURSES.map(({ course, grade }) => (
                  <tr key={course} className="border-b border-border last:border-0">
                    <td className="py-4 text-sm text-text-primary">{course}</td>
                    <td className="py-4 text-right">
                      <span className="rounded-md bg-accent/10 px-2.5 py-1 text-xs font-semibold text-accent">
                        {grade}
                      </span>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </motion.div>
        </div>
      </div>
    </section>
  );
}
