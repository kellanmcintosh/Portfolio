"use client";

import { useEffect, useMemo, useRef, useState } from "react";
import { motion } from "framer-motion";

interface Dims { w: number; h: number; }

function gY(x: number, mu: number, sigma: number, amp: number, bl: number) {
  return bl - amp * Math.exp(-((x - mu) ** 2) / (2 * sigma ** 2));
}

function buildCurve(dims: Dims) {
  const { w, h } = dims;
  const mu = w * 0.5;
  const sigma = w * 0.18;
  const amp = h * 0.31;
  const bl = h * 0.68;

  const N = 600;
  const pts: [number, number][] = Array.from({ length: N + 1 }, (_, i) => {
    const x = (i / N) * w;
    return [x, gY(x, mu, sigma, amp, bl)];
  });

  const ptStr = pts.map(([x, y]) => `${x.toFixed(1)},${y.toFixed(1)}`).join(" L ");
  const strokePath = `M ${ptStr}`;
  const fillPath = `M 0,${bl.toFixed(1)} L ${ptStr} L ${w.toFixed(1)},${bl.toFixed(1)} Z`;

  let pathLen = 0;
  for (let i = 1; i < pts.length; i++) {
    const dx = pts[i][0] - pts[i - 1][0];
    const dy = pts[i][1] - pts[i - 1][1];
    pathLen += Math.sqrt(dx * dx + dy * dy);
  }
  pathLen = Math.ceil(pathLen) + 200;

  const peakY = bl - amp;
  const s1L = mu - sigma;
  const s1R = mu + sigma;
  const s1Y = gY(s1L, mu, sigma, amp, bl);
  const s2L = Math.max(0, mu - 2 * sigma);
  const s2R = Math.min(w, mu + 2 * sigma);
  const s2Y = gY(s2L, mu, sigma, amp, bl);

  const scatterPoints = (
    [[0.10, -38], [0.20, 42], [0.32, -30], [0.43, 36],
     [0.58, -24], [0.70, 28], [0.80, -28], [0.90, 20]] as [number, number][]
  ).map(([rx, oy]) => {
    const x = rx * w;
    return { x, y: gY(x, mu, sigma, amp, bl) + oy };
  });

  return {
    w, h, mu, bl, peakY,
    s1L, s1R, s1Y, s2L, s2R, s2Y,
    btY1: bl + 4, btY2: bl + 11,
    strokePath, fillPath, pathLen,
    scatterPoints,
  };
}

// Measures character positions in "Kellan McIntosh" using the actual loaded font
// so weave strip transitions always fall on letter boundaries, not mid-glyph.
async function computeWeaveStrips(w: number): Promise<[number, number][]> {
  const leftPad = w >= 768 ? 80 : 40;
  const fontSize = Math.min(Math.max(w * 0.11, 72), 144); // mirrors CSS clamp(4.5rem,11vw,9rem)

  await document.fonts.ready;

  const canvas = document.createElement("canvas");
  const ctx = canvas.getContext("2d")!;
  ctx.font = `900 ${fontSize}px "Playfair Display", Georgia, serif`;

  // cumW[i] = pixel width of text.slice(0, i) — accounts for kerning in context
  const text = "Kellan McIntosh";
  const cumW = Array.from({ length: text.length + 1 }, (_, i) =>
    ctx.measureText(text.slice(0, i)).width
  );

  const xAt = (i: number) => leftPad + cumW[i];          // left edge of char i
  const wOf = (i: number) => cumW[i + 1] - cumW[i];      // advance width of char i

  // Character indices: K=0 e=1 l=2 l=3 a=4 n=5 ' '=6 M=7 c=8 I=9 n=10 t=11 o=12 s=13 h=14

  return [
    // "Ke" in front → curve goes behind at the clean e→l boundary
    [xAt(0), cumW[2] - cumW[0]],
    // "n" (end of Kellan) in front → transitions at a→n and n→space boundaries
    [xAt(5), wOf(5)],
    // "o" in McIntosh in front → brief beat before h
    [xAt(12), wOf(12)],
    // First ~50% of "h" in front → curve then vanishes behind the right stroke of h
    [xAt(14), wOf(14) * 0.5],
  ];
}

const SCATTER_OPACITIES = [0.19, 0.15, 0.18, 0.21, 0.17, 0.15, 0.19, 0.16];
const SCATTER_RADII = [2.0, 1.5, 2.0, 1.5, 2.0, 1.5, 2.0, 1.5];
const SCATTER_DELAYS = [2.5, 2.65, 2.8, 2.6, 2.75, 2.55, 2.85, 2.7];

export default function Hero() {
  const sectionRef = useRef<HTMLElement>(null);
  const [dims, setDims] = useState<Dims | null>(null);
  const [weaveStrips, setWeaveStrips] = useState<[number, number][]>([]);

  useEffect(() => {
    const el = sectionRef.current;
    if (!el) return;
    const ro = new ResizeObserver(([entry]) => {
      const { width, height } = entry.contentRect;
      setDims({ w: width, h: height });
    });
    ro.observe(el);
    return () => ro.disconnect();
  }, []);

  // Recompute strips whenever section width changes (font-size or padding may shift)
  useEffect(() => {
    if (!dims) return;
    computeWeaveStrips(dims.w).then(setWeaveStrips);
  }, [dims]);

  const curve = useMemo(() => (dims ? buildCurve(dims) : null), [dims]);

  const scrollToProjects = () => {
    document.getElementById("projects")?.scrollIntoView({ behavior: "smooth" });
  };

  return (
    <section
      ref={sectionRef}
      className="relative flex min-h-screen flex-col overflow-hidden"
    >
      {/* Background grid */}
      <div
        className="pointer-events-none absolute inset-0 z-0"
        style={{
          backgroundImage:
            "linear-gradient(rgba(108,100,86,0.07) 1px, transparent 1px), linear-gradient(90deg, rgba(108,100,86,0.07) 1px, transparent 1px)",
          backgroundSize: "64px 64px",
        }}
      />

      {/* Gaussian — back layer (z:1) */}
      {curve && (
        <svg
          className="pointer-events-none absolute inset-0"
          style={{ zIndex: 1, width: "100%", height: "100%" }}
          viewBox={`0 0 ${curve.w} ${curve.h}`}
          xmlns="http://www.w3.org/2000/svg"
        >
          <defs>
            <linearGradient id="kmFill" x1="0" y1="0" x2="0" y2="1">
              <stop offset="0%" stopColor="#B8860B" stopOpacity="0.11" />
              <stop offset="100%" stopColor="#B8860B" stopOpacity="0.01" />
            </linearGradient>
            <filter id="kmGlow" x="-5%" y="-80%" width="110%" height="260%">
              <feGaussianBlur in="SourceGraphic" stdDeviation="2.5" result="blur" />
              <feMerge>
                <feMergeNode in="blur" />
                <feMergeNode in="SourceGraphic" />
              </feMerge>
            </filter>
            <filter id="kmPeak" x="-400%" y="-400%" width="900%" height="900%">
              <feGaussianBlur in="SourceGraphic" stdDeviation="14" result="b1" />
              <feGaussianBlur in="SourceGraphic" stdDeviation="6" result="b2" />
              <feMerge>
                <feMergeNode in="b1" />
                <feMergeNode in="b2" />
                <feMergeNode in="SourceGraphic" />
              </feMerge>
            </filter>
          </defs>

          <motion.path
            d={curve.fillPath}
            fill="url(#kmFill)"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ duration: 0.8, delay: 0.3 }}
          />

          <motion.path
            d={curve.strokePath}
            fill="none"
            stroke="#B8860B"
            strokeWidth="1.6"
            strokeDasharray={curve.pathLen}
            strokeDashoffset={curve.pathLen}
            filter="url(#kmGlow)"
            animate={{ strokeDashoffset: 0 }}
            transition={{ duration: 2.5, delay: 0.1, ease: [0.16, 1, 0.3, 1] }}
          />

          <line x1="0" y1={curve.bl} x2={curve.w} y2={curve.bl} stroke="#B8860B" strokeWidth="0.5" opacity="0.11" />

          <line x1={curve.s1L} y1={curve.s1Y} x2={curve.s1L} y2={curve.bl} stroke="#B8860B" strokeWidth="0.75" strokeDasharray="4,5" opacity="0.26" />
          <line x1={curve.s1R} y1={curve.s1Y} x2={curve.s1R} y2={curve.bl} stroke="#B8860B" strokeWidth="0.75" strokeDasharray="4,5" opacity="0.26" />

          <line x1={curve.s2L} y1={curve.s2Y} x2={curve.s2L} y2={curve.bl} stroke="#B8860B" strokeWidth="0.5" strokeDasharray="3,7" opacity="0.13" />
          <line x1={curve.s2R} y1={curve.s2Y} x2={curve.s2R} y2={curve.bl} stroke="#B8860B" strokeWidth="0.5" strokeDasharray="3,7" opacity="0.13" />

          <line x1={curve.s1L} y1={curve.s1Y} x2={curve.s1R} y2={curve.s1Y} stroke="#B8860B" strokeWidth="0.5" strokeDasharray="5,7" opacity="0.11" />

          <line x1={curve.mu}  y1={curve.btY1} x2={curve.mu}  y2={curve.btY2} stroke="#B8860B" strokeWidth="1.5" opacity="0.40" />
          <line x1={curve.s1L} y1={curve.btY1} x2={curve.s1L} y2={curve.btY2} stroke="#B8860B" strokeWidth="1.0" opacity="0.24" />
          <line x1={curve.s1R} y1={curve.btY1} x2={curve.s1R} y2={curve.btY2} stroke="#B8860B" strokeWidth="1.0" opacity="0.24" />
          <line x1={curve.s2L} y1={curve.btY1} x2={curve.s2L} y2={curve.btY2} stroke="#B8860B" strokeWidth="0.7" opacity="0.14" />
          <line x1={curve.s2R} y1={curve.btY1} x2={curve.s2R} y2={curve.btY2} stroke="#B8860B" strokeWidth="0.7" opacity="0.14" />

          <text x={curve.mu}  y={curve.peakY - 18} textAnchor="middle" fontFamily="'Courier New',monospace" fontSize="13"  fill="#B8860B" opacity="0.44">μ</text>
          <text x={curve.s1L} y={curve.s1Y - 13}   textAnchor="middle" fontFamily="'Courier New',monospace" fontSize="10"  fill="#B8860B" opacity="0.28">−σ</text>
          <text x={curve.s1R} y={curve.s1Y - 13}   textAnchor="middle" fontFamily="'Courier New',monospace" fontSize="10"  fill="#B8860B" opacity="0.28">+σ</text>
          <text x={curve.s2L} y={curve.s2Y - 11}   textAnchor="middle" fontFamily="'Courier New',monospace" fontSize="8.5" fill="#B8860B" opacity="0.17">−2σ</text>
          <text x={curve.s2R} y={curve.s2Y - 11}   textAnchor="middle" fontFamily="'Courier New',monospace" fontSize="8.5" fill="#B8860B" opacity="0.17">+2σ</text>

          <text x={curve.w - 28} y={curve.h - 28} textAnchor="end" fontFamily="'Courier New',monospace" fontSize="11" fill="#B8860B" opacity="0.13" letterSpacing="0.06em">~ N(μ, σ²)</text>

          <motion.circle
            cx={curve.mu}
            cy={curve.peakY}
            r="3.5"
            fill="#B8860B"
            filter="url(#kmPeak)"
            style={{ transformBox: "fill-box", transformOrigin: "center" }}
            initial={{ opacity: 0, scale: 0.1 }}
            animate={{ opacity: 0.75, scale: 1 }}
            transition={{ duration: 0.7, delay: 2.5, ease: [0.34, 1.56, 0.64, 1] }}
          />

          {curve.scatterPoints.map((pt, i) => (
            <motion.circle
              key={i}
              cx={pt.x}
              cy={pt.y}
              r={SCATTER_RADII[i]}
              fill="#B8860B"
              initial={{ opacity: 0 }}
              animate={{ opacity: SCATTER_OPACITIES[i] }}
              transition={{ duration: 0.6, delay: SCATTER_DELAYS[i] }}
            />
          ))}
        </svg>
      )}

      {/* Hero content (z:2) */}
      <div
        className="relative flex flex-1 flex-col justify-end px-10 pb-[20vh] md:px-20"
        style={{ zIndex: 2 }}
      >
        <motion.h1
          className="font-black leading-[0.95] tracking-[-0.02em] text-text-primary"
          style={{
            fontFamily: "var(--font-playfair-display)",
            fontSize: "clamp(4.5rem, 11vw, 9rem)",
          }}
          initial={{ opacity: 0, y: 36 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 1.1, delay: 0.25, ease: [0.16, 1, 0.3, 1] }}
        >
          Kellan McIntosh
        </motion.h1>

        <motion.div
          className="mt-6 h-px bg-accent"
          style={{ originX: 0 }}
          initial={{ scaleX: 0 }}
          animate={{ scaleX: 1 }}
          transition={{ duration: 0.85, delay: 0.85, ease: [0.16, 1, 0.3, 1] }}
        />

        <motion.p
          className="mt-4 uppercase text-text-secondary"
          style={{ fontSize: "11.5px", letterSpacing: "0.22em" }}
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ duration: 0.8, delay: 1.15 }}
        >
          Data Scientist &amp; ML Engineer
        </motion.p>

        <motion.div
          className="mt-8 flex flex-wrap gap-4"
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ duration: 0.8, delay: 1.4 }}
        >
          <button
            onClick={scrollToProjects}
            className="border border-accent px-7 py-2.5 text-accent transition-colors hover:bg-accent hover:text-background"
            style={{ fontSize: "11.5px", fontWeight: 500, letterSpacing: "0.14em", textTransform: "uppercase" }}
          >
            View Projects
          </button>
          <a
            href="/resume.pdf"
            download
            className="border border-border px-7 py-2.5 text-text-secondary transition-colors hover:border-accent hover:text-accent"
            style={{ display: "inline-block", fontSize: "11.5px", fontWeight: 500, letterSpacing: "0.14em", textTransform: "uppercase", textDecoration: "none" }}
          >
            Download Resume
          </a>
        </motion.div>
      </div>

      {/* Gaussian — front layer (z:3, weave through text) */}
      {curve && weaveStrips.length > 0 && (
        <svg
          className="pointer-events-none absolute inset-0"
          style={{ zIndex: 3, width: "100%", height: "100%" }}
          viewBox={`0 0 ${curve.w} ${curve.h}`}
          xmlns="http://www.w3.org/2000/svg"
        >
          <defs>
            <clipPath id="kmWeave">
              {weaveStrips.map(([x, width], i) => (
                <rect key={i} x={x} y="0" width={width} height={curve.h} />
              ))}
            </clipPath>
            <filter id="kmFrontGlow" x="-5%" y="-80%" width="110%" height="260%">
              <feGaussianBlur in="SourceGraphic" stdDeviation="4.5" result="blur" />
              <feMerge>
                <feMergeNode in="blur" />
                <feMergeNode in="SourceGraphic" />
              </feMerge>
            </filter>
          </defs>

          <motion.path
            d={curve.strokePath}
            fill="none"
            stroke="#B8860B"
            strokeWidth="2.5"
            strokeDasharray={curve.pathLen}
            strokeDashoffset={curve.pathLen}
            clipPath="url(#kmWeave)"
            filter="url(#kmFrontGlow)"
            animate={{ strokeDashoffset: 0 }}
            transition={{ duration: 2.5, delay: 0.1, ease: [0.16, 1, 0.3, 1] }}
          />
        </svg>
      )}
    </section>
  );
}
