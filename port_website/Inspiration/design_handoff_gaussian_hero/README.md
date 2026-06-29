# Handoff: Gaussian Bell Curve Hero Section

## Overview
A redesigned hero section for Kellan McIntosh's portfolio that creatively weaves a golden Gaussian bell curve through the hero name and title. The design adds a statistics-themed faint grid background while maintaining a luxurious aesthetic. This creates a sophisticated visual metaphor for data science and ML engineering expertise.

## About the Design Files
The files in this bundle are **design references created in HTML** — high-fidelity prototypes showing the intended look, layout, and animations. Your task is to **recreate this design in your Next.js/React portfolio codebase** using your established patterns, libraries (Framer Motion, Tailwind, etc.), and component structure — not to copy the HTML directly.

## Fidelity
**High-fidelity (hifi)**: Pixel-perfect mockup with final colors, typography, spacing, and animations. Recreate the UI and interactions exactly as shown using your existing Next.js setup.

## Screen: Hero Section

### Purpose
First-impression landing section showcasing Kellan's name, title, and call-to-action buttons. The Gaussian curve is a visual metaphor for statistical expertise and probability modeling.

### Layout
- **Full viewport height** (min-h-screen)
- **Navbar** (fixed/sticky at top): 28px vertical padding, 80px horizontal padding
  - Logo "KM" (Playfair Display, 20px, font-weight: 700)
  - Nav links: Projects, About, Resume, Grades (11px uppercase, letter-spacing: 0.18em)
  - Contact button (right-aligned, gold border)
- **Background grid**: 64px × 64px, faint grey (#6C6456 at 7% opacity), fixed position
- **Hero content** (bottom half of viewport):
  - Name "Kellan McIntosh": Playfair Display, clamp(4.5rem, 11vw, 9rem), font-weight: 900, line-height: 0.95, letter-spacing: −0.02em, color: #1A1A1A
  - Gold dividing rule (1px height, #B8860B) below name with left-origin scale animation
  - Subtitle: 11.5px uppercase, letter-spacing: 0.22em, color: #5C5A54
  - Two CTA buttons: "View Projects" (gold border, hover: gold bg + white text), "Download Resume" (grey border)

### Gaussian Bell Curve (SVG Overlay)

**Two-layer system creates the weave effect:**

**Back layer (z: 1)** — appears *behind* text:
- Smooth curve: Gaussian distribution with μ at 50% viewport width, σ at 18% viewport width
- Curve amplitude: 31% of viewport height
- Baseline: 81% down from top
- Stroke: #B8860B, 1.6px, animated draw (2.5s, cubic-bezier(0.16,1,0.3,1), 0.1s delay)
- Soft fill underneath curve using linear gradient (top: #B8860B at 11% opacity → bottom: #B8860B at 1% opacity)
- Statistical annotations:
  - Baseline horizontal line (0.5px, 11% opacity)
  - ±σ dashed vertical lines (0.75px, 26% opacity)
  - ±2σ dashed vertical lines (0.5px, 13% opacity)
  - Baseline tick marks at μ, ±σ, ±2σ
  - Labels: μ, −σ, +σ, −2σ, +2σ in monospace (Courier New)
  - Watermark: `~ N(μ, σ²)` bottom-right (11px, 13% opacity)
  - 8 scatter data points distributed near curve (circles, 1.5–2px radius, 15–21% opacity)

**Front layer (z: 3)** — appears *in front* of text at weave strips:
- Same path as back, but thicker (2.5px) and more glow
- Clipped to 4 rectangular regions where curve weaves *through* letters:
  - Strip 0: ~5.5% to ~12.5% (through "K" and early "Kellan")
  - Strip 1: ~25% to ~30.5% (end of "Kellan")
  - Strip 2: ~77.5% to ~85.5% (through "tosh" in McIntosh)
  - Strip 3: ~88.2% to ~100% (after "h")

**Peak accent:**
- Golden glowing dot at curve peak (μ, peakY) with radial blur filter
- Animates in (popIn) at 2.5s

### Animations
- **Fade-up**: Name, subtitle, buttons (fadeUp keyframe, 1.1s, 0.25s delay for name)
- **Expand-from-left**: Gold dividing rule (0.85s, 0.85s delay)
- **Draw curve**: Both SVG layers (2.5s cubic-bezier(0.16,1,0.3,1), 0.1s delay)
- **Scatter fade-in**: Data points stagger in (0.6s, 2.5s–2.85s delays)
- **Peak pop-in**: Center dot (0.7s cubic-bezier(0.34,1.56,0.64,1), 2.5s delay)

### Colors & Typography
**Colors:**
- Background: #FAF8F3 (warm off-white)
- Text primary: #1A1A1A (near-black)
- Text secondary: #5C5A54 (warm grey)
- Accent (curve, borders): #B8860B (golden)
- Border: #D9D2C5 (light warm grey)

**Typography:**
- Name: Playfair Display (Google Fonts), 900 weight
- Body text, buttons: Geist Sans (Next.js default)
- Labels/annotations: Courier New (monospace)
- Sizes: See layout section above

### Component Structure (React/Next.js)
```
Hero.tsx (client component)
├── Navbar
│   ├── Logo
│   ├── Nav links
│   └── Contact button
├── Background grid div
├── SVG (back layer with curve, baseline, grid lines, annotations, scatter points)
├── Content container
│   ├── h1 (name)
│   ├── div (gold rule)
│   ├── p (subtitle)
│   └── div (button group)
│       ├── button (View Projects)
│       └── a (Download Resume)
├── SVG (front layer with weave clips)
└── Border glow overlay div
```

### Interactions & Behavior
- **View Projects button**: Scroll to #projects section smoothly
- **Download Resume link**: Download `/resume.pdf`
- **Contact button**: Navigate to contact section or modal
- All buttons have hover states (color/border changes)
- Responsive: All text uses `clamp()` for fluid scaling; grid and curve scale with viewport

### State & Data
- Window dimensions (w, h) tracked on resize
- Gaussian parameters computed from viewport size:
  - μ = w × 0.50
  - σ = w × 0.18
  - amplitude = h × 0.31
  - baseline = h × 0.81
- 601-point SVG path generated from Gaussian formula: y = baseline − amplitude × exp(−(x − μ)² / (2 × σ²))

### Responsive Behavior
- Grid remains 64px × 64px at all sizes (fixed background-size)
- Typography uses `clamp()` for fluid scaling
- SVG curve and all elements scale with viewport dimensions
- Navbar padding and text sizes adjust proportionally
- On small screens, consider single-column button layout

### Design Tokens
**Spacing:**
- Navbar padding: 28px (vertical), 80px (horizontal)
- Grid size: 64px × 64px
- Button padding: 10px 28px
- Gaps: 16px (button group), 44px (nav links)

**Shadows & Effects:**
- Glow border (inset): `inset 0 0 0 1.5px rgba(184,134,11,0.55), inset 0 0 80px rgba(184,134,11,0.22)`
- Curve glow filters: Gaussian blur (stdDeviation: 2.5–14px)

**Border Radius:**
- None (sharp edges throughout)

## Assets
- Google Fonts: Playfair Display (weights 400–900)
- Geist Sans & Mono (bundled with Next.js)
- Icons: None (text-only design)
- Resume PDF: `/public/resume.pdf`

## Files in This Handoff
- `Hero.dc.html` — High-fidelity Design Component showing the final hero
- `README.md` — This documentation

## Next Steps for Implementation
1. Examine the `Hero.dc.html` preview to understand the exact look and feel
2. Recreate the Navbar as a separate component if it isn't already
3. Build the hero section as a new `Hero.tsx` component in your `src/components/` directory
4. Use Framer Motion for animations (fade, scale, draw SVG path)
5. Compute Gaussian curve points dynamically in a `useEffect` hook based on window size
6. Render background grid as a CSS pattern (or SVG `<defs>` pattern)
7. Use SVG `<clipPath>` to create the weave effect (front layer clipped to strips)
8. Test responsive behavior across mobile, tablet, desktop
9. Integrate with your existing page layout and routing

---

**Questions?** Refer to the HTML prototype or ask Claude Code to clarify any measurements or animations.
