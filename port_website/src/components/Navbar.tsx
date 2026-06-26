"use client";

import { useEffect, useRef, useState } from "react";
import { motion, AnimatePresence } from "framer-motion";

const NAV_LINKS = [
  { label: "Projects", href: "#projects" },
  { label: "About", href: "#about" },
  { label: "Resume", href: "#resume" },
  { label: "Grades", href: "#grades" },
  { label: "Contact", href: "#contact" },
];

const SECTION_IDS = NAV_LINKS.map((l) => l.href.slice(1));

export default function Navbar() {
  const [scrolled, setScrolled] = useState(false);
  const [activeSection, setActiveSection] = useState<string | null>(null);
  const [drawerOpen, setDrawerOpen] = useState(false);
  const [hoveredLink, setHoveredLink] = useState<string | null>(null);

  // Scroll-aware backdrop
  useEffect(() => {
    const onScroll = () => setScrolled(window.scrollY > 60);
    window.addEventListener("scroll", onScroll, { passive: true });
    return () => window.removeEventListener("scroll", onScroll);
  }, []);

  // Active section via IntersectionObserver
  useEffect(() => {
    const observers: IntersectionObserver[] = [];
    const visibleSections = new Set<string>();

    SECTION_IDS.forEach((id) => {
      const el = document.getElementById(id);
      if (!el) return;
      const obs = new IntersectionObserver(
        ([entry]) => {
          if (entry.isIntersecting) {
            visibleSections.add(id);
          } else {
            visibleSections.delete(id);
          }
          // Pick the first visible section in nav order
          const active = SECTION_IDS.find((s) => visibleSections.has(s));
          setActiveSection(active ?? null);
        },
        { rootMargin: "-20% 0px -60% 0px" }
      );
      obs.observe(el);
      observers.push(obs);
    });

    return () => observers.forEach((o) => o.disconnect());
  }, []);

  const scrollToTop = () => window.scrollTo({ top: 0, behavior: "smooth" });

  const handleNavClick = (href: string) => {
    const id = href.slice(1);
    const el = document.getElementById(id);
    if (el) el.scrollIntoView({ behavior: "smooth" });
    setDrawerOpen(false);
  };

  return (
    <>
      <motion.nav
        className="fixed top-0 left-0 right-0 z-50 h-16"
        animate={{
          backgroundColor: scrolled ? "rgba(10,10,15,0.85)" : "rgba(10,10,15,0)",
          borderBottomColor: scrolled ? "rgba(31,41,55,0.6)" : "rgba(31,41,55,0)",
        }}
        style={{ borderBottomWidth: 1, borderBottomStyle: "solid" }}
        transition={{ duration: 0.3, ease: "easeInOut" }}
      >
        {/* Backdrop blur layer — only visible when scrolled */}
        <motion.div
          className="absolute inset-0 backdrop-blur-md"
          animate={{ opacity: scrolled ? 1 : 0 }}
          transition={{ duration: 0.3 }}
          style={{ zIndex: -1 }}
        />

        <div className="mx-auto flex h-full max-w-6xl items-center justify-between px-6">
          {/* KM monogram */}
          <button
            onClick={scrollToTop}
            className="text-lg font-semibold tracking-widest text-text-primary hover:text-accent transition-colors duration-200"
            aria-label="Scroll to top"
          >
            KM
          </button>

          {/* Desktop links */}
          <ul className="hidden md:flex items-center gap-8">
            {NAV_LINKS.map(({ label, href }) => {
              const id = href.slice(1);
              const isActive = activeSection === id;
              return (
                <li key={href} className="relative">
                  <button
                    onClick={() => handleNavClick(href)}
                    onMouseEnter={() => setHoveredLink(href)}
                    onMouseLeave={() => setHoveredLink(null)}
                    className={`text-sm font-medium transition-colors duration-200 pb-0.5 ${
                      isActive ? "text-accent" : "text-text-secondary hover:text-text-primary"
                    }`}
                  >
                    {label}
                    {/* Cyan underline slide-in */}
                    <motion.span
                      className="absolute bottom-0 left-0 h-px bg-accent"
                      initial={{ scaleX: 0 }}
                      animate={{ scaleX: hoveredLink === href || isActive ? 1 : 0 }}
                      transition={{ duration: 0.2, ease: "easeOut" }}
                      style={{ originX: 0, width: "100%" }}
                    />
                  </button>
                </li>
              );
            })}
          </ul>

          {/* Hamburger */}
          <button
            className="md:hidden flex flex-col gap-1.5 p-2 text-text-secondary hover:text-text-primary transition-colors"
            onClick={() => setDrawerOpen((v) => !v)}
            aria-label={drawerOpen ? "Close menu" : "Open menu"}
          >
            <motion.span
              className="block h-px w-6 bg-current"
              animate={drawerOpen ? { rotate: 45, y: 6 } : { rotate: 0, y: 0 }}
              transition={{ duration: 0.2 }}
            />
            <motion.span
              className="block h-px w-6 bg-current"
              animate={drawerOpen ? { opacity: 0 } : { opacity: 1 }}
              transition={{ duration: 0.2 }}
            />
            <motion.span
              className="block h-px w-6 bg-current"
              animate={drawerOpen ? { rotate: -45, y: -6 } : { rotate: 0, y: 0 }}
              transition={{ duration: 0.2 }}
            />
          </button>
        </div>
      </motion.nav>

      {/* Mobile drawer */}
      <AnimatePresence>
        {drawerOpen && (
          <>
            {/* Overlay */}
            <motion.div
              className="fixed inset-0 z-40 bg-black/60"
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
              transition={{ duration: 0.2 }}
              onClick={() => setDrawerOpen(false)}
            />

            {/* Drawer panel */}
            <motion.div
              className="fixed top-16 left-0 right-0 z-40 bg-surface border-b border-border"
              initial={{ opacity: 0, y: -8 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -8 }}
              transition={{ duration: 0.2, ease: "easeOut" }}
            >
              <ul className="flex flex-col px-6 py-4 gap-1">
                {NAV_LINKS.map(({ label, href }) => {
                  const isActive = activeSection === href.slice(1);
                  return (
                    <li key={href}>
                      <button
                        onClick={() => handleNavClick(href)}
                        className={`w-full text-left py-3 text-base font-medium border-b border-border last:border-0 transition-colors duration-150 ${
                          isActive ? "text-accent" : "text-text-secondary"
                        }`}
                      >
                        {label}
                      </button>
                    </li>
                  );
                })}
              </ul>
            </motion.div>
          </>
        )}
      </AnimatePresence>
    </>
  );
}
