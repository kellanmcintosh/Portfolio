"use client";

import { motion, type Variants } from "framer-motion";
import { useEffect, useRef } from "react";

function DotGrid() {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const SPACING = 32;
    let raf: number;
    let t = 0;

    const resize = () => {
      canvas.width = canvas.offsetWidth;
      canvas.height = canvas.offsetHeight;
    };

    const draw = () => {
      ctx.clearRect(0, 0, canvas.width, canvas.height);

      const cols = Math.ceil(canvas.width / SPACING) + 1;
      const rows = Math.ceil(canvas.height / SPACING) + 1;
      const cx = canvas.width / 2;
      const cy = canvas.height / 2;

      for (let c = 0; c < cols; c++) {
        for (let r = 0; r < rows; r++) {
          const x = c * SPACING;
          const y = r * SPACING;
          const dist = Math.sqrt((x - cx) ** 2 + (y - cy) ** 2);
          const opacity = 0.05 + 0.07 * Math.sin(t * 0.4 - dist * 0.018);
          ctx.beginPath();
          ctx.arc(x, y, 1.5, 0, Math.PI * 2);
          ctx.fillStyle = `rgba(0, 217, 255, ${Math.max(0, opacity)})`;
          ctx.fill();
        }
      }

      t += 0.016;
      raf = requestAnimationFrame(draw);
    };

    resize();
    draw();

    window.addEventListener("resize", resize);
    return () => {
      cancelAnimationFrame(raf);
      window.removeEventListener("resize", resize);
    };
  }, []);

  return (
    <canvas
      ref={canvasRef}
      className="absolute inset-0 h-full w-full pointer-events-none"
    />
  );
}

const container: Variants = {
  hidden: {},
  show: { transition: { staggerChildren: 0.15, delayChildren: 0.2 } },
};

const item: Variants = {
  hidden: { opacity: 0, y: 24 },
  show: { opacity: 1, y: 0, transition: { duration: 0.6, ease: "easeOut" } },
};

export default function Hero() {
  const scrollToProjects = () => {
    document.getElementById("projects")?.scrollIntoView({ behavior: "smooth" });
  };

  return (
    <section className="relative flex min-h-screen flex-col items-center justify-center overflow-hidden">
      <DotGrid />
      <motion.div
        className="relative z-10 flex flex-col items-center px-6 text-center"
        variants={container}
        initial="hidden"
        animate="show"
      >
        <motion.h1
          className="text-6xl font-bold tracking-tight text-text-primary md:text-8xl"
          variants={item}
        >
          Kellan McIntosh
        </motion.h1>
        <motion.p
          className="mt-4 text-xl font-medium text-accent md:text-2xl"
          variants={item}
        >
          Data Science &amp; ML Engineer
        </motion.p>
        <motion.p
          className="mt-3 text-base text-text-secondary md:text-lg"
          variants={item}
        >
          Turning data into decisions.
        </motion.p>
        <motion.div
          className="mt-10 flex flex-col gap-4 sm:flex-row"
          variants={item}
        >
          <button
            onClick={scrollToProjects}
            className="rounded-lg bg-accent px-8 py-3 font-semibold text-background transition-colors hover:bg-accent/90"
          >
            View Projects
          </button>
          <a
            href="#"
            className="rounded-lg border border-border px-8 py-3 font-semibold text-text-primary transition-colors hover:border-accent hover:text-accent"
          >
            Download Resume
          </a>
        </motion.div>
      </motion.div>
    </section>
  );
}
