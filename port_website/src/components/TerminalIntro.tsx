"use client";

import { AnimatePresence, motion } from "framer-motion";
import { useCallback, useEffect, useRef, useState } from "react";

type LineKind = "cmd" | "out" | "blank";
type ScriptLine = { kind: LineKind; text: string };

const SCRIPT: ScriptLine[] = [
  { kind: "cmd",   text: "$ whoami" },
  { kind: "out",   text: "> kellan_mcintosh" },
  { kind: "blank", text: "" },
  { kind: "cmd",   text: "$ cat about.txt" },
  { kind: "out",   text: "> Data Science & ML Engineer" },
  { kind: "out",   text: "> UMass Amherst" },
  { kind: "out",   text: "> Building models that actually ship." },
  { kind: "blank", text: "" },
  { kind: "cmd",   text: "$ ./launch_portfolio.sh" },
  { kind: "out",   text: "> Initializing..." },
  { kind: "out",   text: "> Done." },
];

export default function TerminalIntro() {
  const [visible, setVisible] = useState(true);
  const [completedLines, setCompletedLines] = useState<ScriptLine[]>([]);
  const [activeText, setActiveText] = useState("");
  const typingTimer = useRef<ReturnType<typeof setTimeout>>(undefined);
  const exitTimer = useRef<ReturnType<typeof setTimeout>>(undefined);

  const dismiss = useCallback(() => {
    clearTimeout(typingTimer.current);
    clearTimeout(exitTimer.current);
    sessionStorage.setItem("intro_shown", "1");
    setVisible(false);
  }, []);

  useEffect(() => {
    if (
      sessionStorage.getItem("intro_shown") ||
      window.matchMedia("(prefers-reduced-motion: reduce)").matches
    ) {
      setVisible(false);
      return;
    }

    let lineIdx = 0;
    let charIdx = 0;

    const tick = () => {
      if (lineIdx >= SCRIPT.length) {
        exitTimer.current = setTimeout(dismiss, 4000);
        return;
      }

      const line = SCRIPT[lineIdx];

      if (line.kind === "blank") {
        setCompletedLines((prev) => [...prev, line]);
        lineIdx++;
        charIdx = 0;
        typingTimer.current = setTimeout(tick, 100);
        return;
      }

      if (charIdx < line.text.length) {
        charIdx++;
        setActiveText(line.text.slice(0, charIdx));
        typingTimer.current = setTimeout(tick, 40 + Math.random() * 25);
      } else {
        setCompletedLines((prev) => [...prev, line]);
        setActiveText("");
        lineIdx++;
        charIdx = 0;
        typingTimer.current = setTimeout(tick, line.kind === "cmd" ? 350 : 120);
      }
    };

    typingTimer.current = setTimeout(tick, 600);

    return () => {
      clearTimeout(typingTimer.current);
      clearTimeout(exitTimer.current);
    };
  }, [dismiss]);

  const currentLine = SCRIPT[completedLines.length];

  return (
    <AnimatePresence>
      {visible && (
        <motion.div
          className="fixed inset-0 z-[100] font-mono"
          style={{ backgroundColor: "#0F0F0F" }}
          initial={{ opacity: 1 }}
          exit={{ opacity: 0, filter: "blur(10px)", scale: 1.03, y: -16 }}
          transition={{ duration: 0.55, ease: "easeInOut" }}
        >
          <button
            onClick={dismiss}
            className="absolute right-6 top-6 text-sm transition-colors"
            style={{ color: "#5C5A54" }}
            onMouseEnter={(e) => (e.currentTarget.style.color = "#B8860B")}
            onMouseLeave={(e) => (e.currentTarget.style.color = "#5C5A54")}
          >
            Skip →
          </button>

          <div className="px-10 pt-24 md:px-20 md:pt-32">
            {completedLines.map((line, i) =>
              line.kind === "blank" ? (
                <div key={i} className="h-5" />
              ) : (
                <p
                  key={i}
                  className="leading-7"
                  style={{ color: line.kind === "cmd" ? "#B8860B" : "#A3A3A3" }}
                >
                  {line.text}
                </p>
              )
            )}

            {activeText && (
              <p
                className="leading-7"
                style={{ color: currentLine?.kind === "cmd" ? "#B8860B" : "#A3A3A3" }}
              >
                {activeText}
                <span className="animate-pulse">▊</span>
              </p>
            )}
          </div>
        </motion.div>
      )}
    </AnimatePresence>
  );
}
