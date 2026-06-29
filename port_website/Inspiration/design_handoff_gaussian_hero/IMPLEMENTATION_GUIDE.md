# Implementation Guide: Gaussian Hero in Next.js

This guide walks you through converting the Design Component into a production Next.js/React component using Framer Motion and Tailwind.

## Step 1: Create the Hero Component File

Create `src/components/Hero.tsx` and set it up as a client component:

```tsx
"use client";

import { useEffect, useState } from "react";
import { motion } from "framer-motion";

interface GaussianParams {
  mu: number;
  sigma: number;
  amplitude: number;
  baseline: number;
}

export default function Hero() {
  const [dimensions, setDimensions] = useState({ w: 0, h: 0 });
  const [params, setParams] = useState<GaussianParams | null>(null);
  
  // ... implementation continues below
}
```

## Step 2: Handle Responsive Dimensions

```tsx
useEffect(() => {
  const updateDimensions = () => {
    setDimensions({ w: window.innerWidth, h: window.innerHeight });
  };
  
  updateDimensions();
  window.addEventListener("resize", updateDimensions);
  return () => window.removeEventListener("resize", updateDimensions);
}, []);

useEffect(() => {
  if (dimensions.w === 0) return;
  
  setParams({
    mu: dimensions.w * 0.5,
    sigma: dimensions.w * 0.18,
    amplitude: dimensions.h * 0.31,
    baseline: dimensions.h * 0.81,
  });
}, [dimensions]);
```

## Step 3: Generate Gaussian Curve Points

```tsx
function generateGaussianPath(w: number, h: number, params: GaussianParams) {
  const { mu, sigma, amplitude, baseline } = params;
  const N = 600;
  
  const points = Array.from({ length: N + 1 }, (_, i) => {
    const x = (i / N) * w;
    const y = baseline - amplitude * Math.exp(-Math.pow(x - mu, 2) / (2 * sigma * sigma));
    return `${x.toFixed(1)},${y.toFixed(1)}`;
  });
  
  // Stroke path
  const strokePath = `M ${points.join(" L ")}`;
  
  // Fill path (closed area under curve)
  const fillPath = `M 0,${baseline.toFixed(1)} L ${points.join(" L ")} L ${w.toFixed(1)},${baseline.toFixed(1)} Z`;
  
  // Calculate approximate path length for stroke-dasharray animation
  let pathLength = 0;
  for (let i = 1; i < points.length; i++) {
    const [x1, y1] = points[i - 1].split(",").map(Number);
    const [x2, y2] = points[i].split(",").map(Number);
    pathLength += Math.sqrt(Math.pow(x2 - x1, 2) + Math.pow(y2 - y1, 2));
  }
  
  return { strokePath, fillPath, pathLength: Math.ceil(pathLength) + 200 };
}
```

## Step 4: Build the SVG Back Layer

Render the curve, baseline, sigma lines, ticks, and labels:

```tsx
<svg className="absolute inset-0 w-full h-full z-[1] pointer-events-none" viewBox={`0 0 ${dimensions.w} ${dimensions.h}`}>
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
  </defs>
  
  {/* Fill under curve */}
  <motion.path
    d={paths.fillPath}
    fill="url(#kmFill)"
    initial={{ opacity: 0 }}
    animate={{ opacity: 1 }}
    transition={{ duration: 0.8, delay: 0.3 }}
  />
  
  {/* Curve stroke */}
  <motion.path
    d={paths.strokePath}
    fill="none"
    stroke="#B8860B"
    strokeWidth="1.6"
    strokeDasharray={paths.pathLength}
    strokeDashoffset={paths.pathLength}
    filter="url(#kmGlow)"
    animate={{ strokeDashoffset: 0 }}
    transition={{ duration: 2.5, delay: 0.1, ease: [0.16, 1, 0.3, 1] }}
  />
  
  {/* Baseline */}
  <line
    x1="0"
    y1={params.baseline}
    x2={dimensions.w}
    y2={params.baseline}
    stroke="#B8860B"
    strokeWidth="0.5"
    opacity="0.11"
  />
  
  {/* ±σ dashed verticals */}
  {/* ±2σ dashed verticals */}
  {/* Baseline ticks */}
  {/* Labels: μ, −σ, +σ, −2σ, +2σ */}
  {/* Scatter points */}
  
</svg>
```

## Step 5: Create Weave Clip Paths

Define the rectangular strips where the front curve appears in front of text:

```tsx
// In the front SVG layer
<clipPath id="kmWeave">
  {/* Strip 0: through "K" */}
  <rect x={dimensions.w * 0.055} y="0" width={dimensions.w * 0.125} height={dimensions.h} />
  {/* Strip 1: end of "Kellan" */}
  <rect x={dimensions.w * 0.250} y="0" width={dimensions.w * 0.055} height={dimensions.h} />
  {/* Strip 2: "tosh" in McIntosh */}
  <rect x={dimensions.w * 0.775} y="0" width={dimensions.w * 0.085} height={dimensions.h} />
  {/* Strip 3: after "h" */}
  <rect x={dimensions.w * 0.882} y="0" width={dimensions.w * 0.118} height={dimensions.h} />
</clipPath>
```

## Step 6: Animate the Name and CTAs

```tsx
<motion.h1
  className="font-playfair font-black text-[clamp(4.5rem,11vw,9rem)] leading-[0.95] tracking-[-0.02em] text-[#1A1A1A]"
  initial={{ opacity: 0, y: 36 }}
  animate={{ opacity: 1, y: 0 }}
  transition={{ duration: 1.1, delay: 0.25, ease: [0.16, 1, 0.3, 1] }}
>
  Kellan McIntosh
</motion.h1>

<motion.div
  className="w-full h-px bg-[#B8860B] mt-6 origin-left"
  initial={{ scaleX: 0 }}
  animate={{ scaleX: 1 }}
  transition={{ duration: 0.85, delay: 0.85, ease: [0.16, 1, 0.3, 1] }}
/>

<motion.p
  className="mt-4 text-xs font-medium tracking-[0.22em] uppercase text-[#5C5A54]"
  initial={{ opacity: 0 }}
  animate={{ opacity: 1 }}
  transition={{ duration: 0.8, delay: 1.15 }}
>
  Data Scientist &amp; ML Engineer
</motion.p>

<motion.div
  className="mt-8 flex gap-4 flex-wrap"
  initial={{ opacity: 0 }}
  animate={{ opacity: 1 }}
  transition={{ duration: 0.8, delay: 1.4 }}
>
  <button className="border border-[#B8860B] px-7 py-2.5 text-xs font-medium tracking-[0.14em] uppercase text-[#B8860B] hover:bg-[#B8860B] hover:text-white transition-colors">
    View Projects
  </button>
  <a href="/resume.pdf" className="border border-[#D9D2C5] px-7 py-2.5 text-xs font-medium tracking-[0.14em] uppercase text-[#5C5A54] hover:text-[#1A1A1A] transition-colors">
    Download Resume
  </a>
</motion.div>
```

## Step 7: Add Background Grid

Use CSS to create the faint grid pattern:

```tsx
<div
  className="absolute inset-0 pointer-events-none"
  style={{
    backgroundImage: `linear-gradient(rgba(108,100,86,0.07) 1px, transparent 1px), linear-gradient(90deg, rgba(108,100,86,0.07) 1px, transparent 1px)`,
    backgroundSize: "64px 64px",
  }}
/>
```

## Step 8: Build the Full Component

Combine all pieces into a complete, responsive Hero component. Key considerations:

- **State management**: Track window dimensions and regenerate curves on resize
- **Animation timing**: Use Framer Motion's `transition` props with calculated delays
- **SVG rendering**: Use `viewBox` for responsive SVG scaling, or set explicit width/height and use CSS `object-fit`
- **Clip paths**: Define in SVG `<defs>` and reference with `clipPath="url(#kmWeave)"`
- **Performance**: Memoize expensive calculations (Gaussian points) to avoid re-renders on every state change

## Step 9: Integrate into Your Page

Import the Hero component in your `src/app/page.tsx`:

```tsx
import Hero from "@/components/Hero";

export default function Home() {
  return (
    <>
      <Hero />
      {/* Rest of your page content */}
    </>
  );
}
```

## Step 10: Test and Refine

- Test on mobile, tablet, desktop breakpoints
- Verify SVG curve animation plays correctly across browsers
- Check that weave clip regions align with text at all viewport sizes
- Adjust `params` (mu, sigma, amplitude, baseline) or strip positions if needed
- Performance test in DevTools (watch for reflow/repaint during resize)

## Common Adjustments

**Curve intensity**: Increase/decrease `amplitude` (default: h × 0.31)

**Curve width**: Adjust `sigma` (default: w × 0.18)

**Curve position**: Shift `baseline` up/down (default: h × 0.81)

**Weave strips**: Update the `x` and `width` percentages in the `<clipPath>` rectangles

**Animation speed**: Modify `duration` and `delay` values in Framer Motion `transition` props

**Grid size**: Change `backgroundSize` (default: 64px 64px)

---

That's it! You now have a production-ready, responsive Gaussian hero section for your portfolio.
