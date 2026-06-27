"use client";

import { AnimatePresence, motion } from "framer-motion";
import { useCallback, useEffect, useRef, useState } from "react";

type LineKind = "cmd" | "prompt" | "out" | "blank";
type ScriptLine = { kind: LineKind; text: string };

const SCRIPT: ScriptLine[] = [
  { kind: "cmd",    text: "$ python3" },
  { kind: "prompt", text: ">>> import pandas as pd" },
  { kind: "prompt", text: ">>> df = pd.read_csv('credit_risk.csv')" },
  { kind: "prompt", text: ">>> df.shape" },
  { kind: "out",    text: "(10000, 24)" },
  { kind: "blank",  text: "" },
  { kind: "prompt", text: ">>> from sklearn.ensemble import GradientBoostingClassifier" },
  { kind: "prompt", text: ">>> model = GradientBoostingClassifier(n_estimators=200)" },
  { kind: "prompt", text: ">>> model.fit(X_train, y_train)" },
  { kind: "out",    text: "[Parallel(n_jobs=1)]: Done 200 out of 200..." },
  { kind: "blank",  text: "" },
  { kind: "prompt", text: ">>> score = model.score(X_test, y_test)" },
  { kind: "prompt", text: ">>> print(f\"ROC-AUC: {score:.4f}\")" },
  { kind: "out",    text: "ROC-AUC: 0.9421" },
  { kind: "blank",  text: "" },
  { kind: "prompt", text: ">>> # Loading portfolio..." },
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
        // Output lines type faster than prompt lines
        const delay = line.kind === "out"
          ? 20 + Math.random() * 15
          : 40 + Math.random() * 25;
        typingTimer.current = setTimeout(tick, delay);
      } else {
        setCompletedLines((prev) => [...prev, line]);
        setActiveText("");
        lineIdx++;
        charIdx = 0;
        // Pause longer after prompt lines (simulates thinking/executing)
        const pause = line.kind === "prompt" ? 450 : line.kind === "cmd" ? 350 : 120;
        typingTimer.current = setTimeout(tick, pause);
      }
    };

    typingTimer.current = setTimeout(tick, 600);

    return () => {
      clearTimeout(typingTimer.current);
      clearTimeout(exitTimer.current);
    };
  }, [dismiss]);

  const currentLine = SCRIPT[completedLines.length];

  const lineColor = (kind: LineKind) => {
    if (kind === "cmd" || kind === "prompt") return "#B8860B";
    return "#A3A3A3";
  };

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
                  style={{ color: lineColor(line.kind) }}
                >
                  {line.text}
                </p>
              )
            )}

            {activeText && (
              <p
                className="leading-7"
                style={{ color: lineColor(currentLine?.kind ?? "out") }}
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
