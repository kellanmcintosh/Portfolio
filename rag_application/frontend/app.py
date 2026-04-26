import json
import logging
import os
from datetime import datetime

import requests
import chromadb
import streamlit as st
from sentence_transformers import SentenceTransformer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

COLLECTION_NAME = "documents"
CHAT_MODEL = "qwen/qwen3-32b"
SUMMARY_MODEL = "llama-3.1-8b-instant"
TOP_K = 5

CHROMA_HOST = os.environ["CHROMA_HOST"]
CHROMA_PORT = int(os.environ["CHROMA_PORT"])
GROQ_API_KEY = os.environ["GROQ_API_KEY"]

GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
embed_model = SentenceTransformer("all-MiniLM-L6-v2")

SYSTEM_PROMPT = (
    "You are a helpful assistant. Answer the user's question using only "
    "the context provided. If the answer is not in the context, say you don't know."
)

FREE_SYSTEM_PROMPT = (
    "You are a helpful assistant. Answer the user's question freely using your full capabilities."
)

REASONING_SUMMARY_PROMPT = (
    "The text below is the internal reasoning a language model used to answer a question. "
    "Rewrite it in 2-3 clear sentences that anyone can understand, without technical jargon. "
    "Explain what information was used and why the model reached its conclusion."
)

# ── Orb — isolated iframe so Streamlit rerenders never kill the animation ──────

ORB_HTML = """<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<style>
  * { margin: 0; padding: 0; box-sizing: border-box; }
  html, body { width: 100%; height: 100%; background: #0D0D12; overflow: hidden; }
  canvas { display: block; width: 100%; height: 100%; }
  #title {
    position: absolute;
    width: 100%;
    text-align: center;
    bottom: 24px;
    font-family: 'Inter', system-ui, sans-serif;
    pointer-events: none;
  }
  #title h1 {
    font-size: clamp(1.1rem, 2.5vw, 1.6rem);
    font-weight: 700;
    letter-spacing: -0.02em;
    background: linear-gradient(135deg, #A78BFA 0%, #C084FC 45%, #DDD6FE 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin: 0 0 0.35rem;
  }
  #title p {
    font-size: 0.78rem;
    color: rgba(148, 163, 184, 0.55);
    letter-spacing: 0.06em;
    text-transform: uppercase;
    margin: 0;
  }
</style>
</head>
<body>
<canvas id="orb"></canvas>
<div id="title">
  <h1>Internal Knowledge Retrieval Agent</h1>
  <p>Ask a question to get started</p>
</div>
<script>
(function () {
  var canvas = document.getElementById('orb');
  var ctx = canvas.getContext('2d');
  var W, H, restX, restY;
  var orbX, orbY;
  var velX = 0, velY = 0;
  var mouseX = null, mouseY = null;
  var t = 0;

  /* Particles orbiting the orb */
  var COLORS = ['#A855F7','#C084FC','#7C3AED','#DDD6FE','#9333EA','#E879F9'];
  var particles = [];
  for (var i = 0; i < 11; i++) {
    particles.push({
      angle: (i / 11) * Math.PI * 2 + Math.random() * 0.4,
      orbitMult: 1.32 + Math.random() * 0.52,
      speed: (0.004 + Math.random() * 0.006) * (Math.random() < 0.5 ? 1 : -1),
      size: 0.75 + Math.random() * 1.4,
      opacity: 0.28 + Math.random() * 0.48,
      color: COLORS[i % COLORS.length]
    });
  }

  function resize() {
    var rect = canvas.getBoundingClientRect();
    W = canvas.width  = rect.width  || 640;
    H = canvas.height = rect.height || 420;
    restX = W / 2;
    restY = H / 2 - 24; /* shift up to leave room for title */
    if (orbX === undefined) { orbX = restX; orbY = restY; }
  }

  function lerp(a, b, f) { return a + (b - a) * f; }

  /* Layered-sine wobble — never perfectly circular */
  function wobbleR(a, R, t) {
    return R
      + 2.5 * Math.sin(a * 3  + t * 1.40)
      + 1.75 * Math.sin(a * 7  - t * 2.15)
      + 1.0 * Math.sin(a * 13 + t * 0.90)
      + 0.5 * Math.sin(a * 19 - t * 3.10);
  }

  canvas.addEventListener('mousemove', function (e) {
    var r = canvas.getBoundingClientRect();
    mouseX = e.clientX - r.left;
    mouseY = e.clientY - r.top;
  });
  canvas.addEventListener('mouseleave', function () {
    mouseX = null; mouseY = null;
  });
  window.addEventListener('resize', resize);

  function draw() {
    t += 0.016;
    requestAnimationFrame(draw);

    /* Breathing: ~3 s sine */
    var breathe = 1 + 0.075 * Math.sin(t * (2 * Math.PI / 3));
    /* Responsive base radius — 0.0925 = 0.185 * 0.5 */
    var R = Math.min(W, H) * 0.0925;

    /* Lazy cursor follow — square region, spring-back when away */
    if (mouseX !== null) {
      /* Horizontal: square clamp so X travel never exceeds the shorter axis */
      var halfSq = Math.min(W, H) * 0.5;
      var tgtX = restX + Math.max(-halfSq, Math.min(halfSq, mouseX - restX));
      /* Vertical: tight ±20 px clamp — enough to feel alive, never clips */
      var tgtY = restY + Math.max(-20, Math.min(20, mouseY - restY));
      orbX = lerp(orbX, tgtX, 0.06);
      orbY = lerp(orbY, tgtY, 0.06);
      velX = 0; velY = 0;
    } else {
      velX = (velX + (restX - orbX) * 0.08) * 0.72;
      velY = (velY + (restY - orbY) * 0.08) * 0.72;
      orbX += velX;
      orbY += velY;
    }

    /* Fix 2: clamp drawn position — body + outermost glow (R*3.0) never clips edge */
    var buffer = R * 3.2;
    var drawX = Math.max(buffer, Math.min(W - buffer, orbX));
    var drawY = Math.max(buffer, Math.min(H - buffer, orbY));

    /* Stretch uses drawn position so distortion matches what the eye sees */
    var dx = drawX - restX, dy = drawY - restY;
    var dist = Math.sqrt(dx * dx + dy * dy);
    var pullAngle = Math.atan2(dy, dx);
    var ts = Math.min(dist / (R * 1.6), 1);   /* stretch factor 0→1 */
    var sX = 1 + ts * 0.60;                    /* 1.0 → 1.60 */
    var sY = 1 - ts * 0.30;                    /* 1.0 → 0.70 */

    ctx.clearRect(0, 0, W, H);

    /* ── Bloom glow rings ─────────────────────────────────────────── */
    var glows = [
      { r: R * 3.0, a0: 0.055 },
      { r: R * 2.2, a0: 0.10  },
      { r: R * 1.7, a0: 0.17  }
    ];
    for (var gi = 0; gi < glows.length; gi++) {
      var g = glows[gi];
      ctx.save();
      ctx.translate(drawX, drawY);
      ctx.rotate(pullAngle);
      ctx.scale(sX * breathe, sY * breathe);
      ctx.rotate(-pullAngle);
      var gr = ctx.createRadialGradient(0, 0, R * 0.2, 0, 0, g.r);
      gr.addColorStop(0,   'rgba(168,85,247,' + g.a0 + ')');
      gr.addColorStop(0.5, 'rgba(124,58,237,' + (g.a0 * 0.4) + ')');
      gr.addColorStop(1,   'rgba(107,33,168,0)');
      ctx.beginPath();
      ctx.arc(0, 0, g.r, 0, Math.PI * 2);
      ctx.fillStyle = gr;
      ctx.fill();
      ctx.restore();
    }

    /* ── Orb body ─────────────────────────────────────────────────── */
    ctx.save();
    ctx.translate(drawX, drawY);
    ctx.rotate(pullAngle);
    ctx.scale(sX * breathe, sY * breathe);
    ctx.rotate(-pullAngle);

    /* Wobbled silhouette */
    var SEG = 96;
    ctx.beginPath();
    for (var j = 0; j <= SEG; j++) {
      var a = (j / SEG) * Math.PI * 2;
      var r = wobbleR(a, R, t);
      if (j === 0) ctx.moveTo(Math.cos(a) * r, Math.sin(a) * r);
      else         ctx.lineTo(Math.cos(a) * r, Math.sin(a) * r);
    }
    ctx.closePath();

    /* Main radial gradient — off-centre highlight for 3-D depth */
    var mg = ctx.createRadialGradient(-R * 0.28, -R * 0.28, 0, 0, 0, R * 1.05);
    mg.addColorStop(0.00, '#EDE9FE');
    mg.addColorStop(0.15, '#C084FC');
    mg.addColorStop(0.40, '#A855F7');
    mg.addColorStop(0.65, '#7C3AED');
    mg.addColorStop(0.82, '#6B21A8');
    mg.addColorStop(1.00, '#3B0764');
    ctx.fillStyle = mg;
    ctx.fill();

    /* Inner pink glow for warmth */
    var ig = ctx.createRadialGradient(0, 0, 0, 0, 0, R * 0.6);
    ig.addColorStop(0, 'rgba(232,121,249,0.32)');
    ig.addColorStop(1, 'rgba(168,85,247,0)');
    ctx.beginPath();
    for (var k = 0; k <= SEG; k++) {
      var a2 = (k / SEG) * Math.PI * 2;
      var r2 = wobbleR(a2, R, t);
      if (k === 0) ctx.moveTo(Math.cos(a2) * r2, Math.sin(a2) * r2);
      else         ctx.lineTo(Math.cos(a2) * r2, Math.sin(a2) * r2);
    }
    ctx.closePath();
    ctx.fillStyle = ig;
    ctx.fill();

    /* Surface sheen — shifts with pull direction */
    var sheen = 0.20 + ts * 0.08;
    var sg = ctx.createRadialGradient(-R*0.32,-R*0.45, 0, -R*0.12,-R*0.2, R*0.78);
    sg.addColorStop(0,    'rgba(255,255,255,' + sheen + ')');
    sg.addColorStop(0.55, 'rgba(255,255,255,0.03)');
    sg.addColorStop(1,    'rgba(255,255,255,0)');
    ctx.beginPath();
    for (var m = 0; m <= SEG; m++) {
      var a3 = (m / SEG) * Math.PI * 2;
      var r3 = wobbleR(a3, R, t);
      if (m === 0) ctx.moveTo(Math.cos(a3) * r3, Math.sin(a3) * r3);
      else         ctx.lineTo(Math.cos(a3) * r3, Math.sin(a3) * r3);
    }
    ctx.closePath();
    ctx.fillStyle = sg;
    ctx.fill();

    ctx.restore();

    /* ── Particles ────────────────────────────────────────────────── */
    for (var pi = 0; pi < particles.length; pi++) {
      var p = particles[pi];
      p.angle += p.speed;
      var pr  = R * p.orbitMult * breathe;
      var px  = drawX + Math.cos(p.angle) * pr * sX;
      var py  = drawY + Math.sin(p.angle) * pr * sY;
      var pAlpha = p.opacity * (0.6 + 0.4 * Math.sin(t * 1.5 + p.angle));

      /* Dot */
      ctx.beginPath();
      ctx.arc(px, py, p.size, 0, Math.PI * 2);
      ctx.fillStyle = p.color;
      ctx.globalAlpha = pAlpha;
      ctx.fill();

      /* Soft glow halo around each particle */
      var halo = ctx.createRadialGradient(px, py, 0, px, py, p.size * 4);
      halo.addColorStop(0, 'rgba(168,85,247,' + (pAlpha * 0.5) + ')');
      halo.addColorStop(1, 'rgba(168,85,247,0)');
      ctx.beginPath();
      ctx.arc(px, py, p.size * 4, 0, Math.PI * 2);
      ctx.fillStyle = halo;
      ctx.fill();

      ctx.globalAlpha = 1;
    }
  }

  resize();
  draw();
})();
</script>
</body>
</html>"""

# ── Three-dot typing indicator shown while LLM generates ──────────────────────

TYPING_HTML = """
<style>
@keyframes dot-bounce {
  0%, 60%, 100% { transform: translateY(0);    opacity: 0.35; }
  30%            { transform: translateY(-6px); opacity: 1;    }
}
.td {
  display: inline-block;
  width: 8px; height: 8px;
  background: #A855F7;
  border-radius: 50%;
  margin: 0 3px;
  animation: dot-bounce 1.2s ease-in-out infinite;
}
.td:nth-child(2) { animation-delay: 0.18s; }
.td:nth-child(3) { animation-delay: 0.36s; }
</style>
<div style="padding: 4px 2px 2px;">
  <span class="td"></span><span class="td"></span><span class="td"></span>
</div>
"""

# ── Global CSS — dark theme, hide chrome, message bubbles ─────────────────────

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600&display=swap');

/* ── Page background ───────────────────────────────────────────────── */
html, body,
[data-testid="stAppViewContainer"],
[data-testid="stMain"],
.main {
    background: #0D0D12 !important;
    color: #E2E8F0 !important;
    font-family: 'Inter', system-ui, -apple-system, sans-serif !important;
}

/* ── Hide Streamlit chrome ─────────────────────────────────────────── */
#MainMenu, footer,
[data-testid="stToolbar"],
[data-testid="stDecoration"],
[data-testid="stHeader"] {
    display: none !important;
    visibility: hidden !important;
}

/* ── Main layout ───────────────────────────────────────────────────── */
/* Emotion injects into <head> after us, so this is belt-and-suspenders;
   the real override is the JS head-injection below. */
[data-testid="stMainBlockContainer"],
[data-testid="stBottomBlockContainer"] {
    max-width: 760px !important;
    width: 760px !important;
    margin-left: auto !important;
    margin-right: auto !important;
    padding-top: 0 !important;
    padding-bottom: 6rem !important;
    padding-left: 1rem !important;
    padding-right: 1rem !important;
}

/* ── Message fade-in ───────────────────────────────────────────────── */
@keyframes fadeInUp {
    from { opacity: 0; transform: translateY(6px); }
    to   { opacity: 1; transform: translateY(0);   }
}

[data-testid="stChatMessage"] {
    animation: fadeInUp 0.28s ease-out;
    margin-bottom: 8px !important;
    padding: 8px 12px !important;
    border-radius: 18px !important;
}

/* ── User bubble — right-aligned, deep purple ──────────────────────── */
[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) {
    background: rgba(109, 40, 217, 0.35) !important;
    margin-left: 12% !important;
    margin-right: 0 !important;
    border-radius: 18px 18px 4px 18px !important;
    border: 1px solid rgba(168, 85, 247, 0.4) !important;
    box-shadow: 0 0 12px rgba(168, 85, 247, 0.2) !important;
    backdrop-filter: blur(8px) !important;
}
[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) p,
[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) li,
[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) code,
[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) strong {
    color: #F3E8FF !important;
}

/* ── Assistant bubble — left-aligned, near-black ───────────────────── */
[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-assistant"]) {
    background: rgba(15, 10, 30, 0.7) !important;
    margin-right: 12% !important;
    margin-left: 0 !important;
    border-radius: 18px 18px 18px 4px !important;
    border: 1px solid rgba(88, 28, 135, 0.3) !important;
    box-shadow: 0 0 8px rgba(88, 28, 135, 0.15) !important;
    backdrop-filter: blur(8px) !important;
}
[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-assistant"]) p,
[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-assistant"]) li,
[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-assistant"]) code,
[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-assistant"]) strong {
    color: #CBD5E1 !important;
}

/* ── Timestamp ─────────────────────────────────────────────────────── */
.msg-timestamp {
    font-size: 0.67rem;
    color: rgba(167, 139, 250, 0.5);
    display: block;
    margin-top: 5px;
}

/* ── Chat input ────────────────────────────────────────────────────── */
[data-baseweb="textarea"] {
    background: #13131A !important;
    border: 1px solid rgba(168, 85, 247, 0.25) !important;
    border-radius: 24px !important;
    transition: border-color 0.2s ease, box-shadow 0.2s ease !important;
}
[data-baseweb="textarea"]:focus-within {
    border-color: #A855F7 !important;
    box-shadow: 0 0 0 2px rgba(168, 85, 247, 0.28) !important;
}
[data-baseweb="base-input"] {
    background: transparent !important;
    border: none !important;
    padding-left: 16px !important;
}
[data-baseweb="base-input"] textarea {
    color: #E2E8F0 !important;
    caret-color: #A855F7 !important;
}
[data-baseweb="base-input"] textarea::placeholder {
    color: rgba(148, 163, 184, 0.4) !important;
}
[data-testid="stChatInput"] > div {
    border: none !important;
    outline: none !important;
    box-shadow: none !important;
}
[data-testid="stChatInput"] > div:focus-within {
    border: none !important;
    outline: none !important;
    box-shadow: none !important;
}
[data-testid="stChatInputSubmitButton"] {
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
}
[data-testid="stChatInputSubmitButton"] svg {
    fill: rgba(148, 163, 184, 0.4) !important;
    transition: fill 0.15s ease !important;
}
[data-testid="stChatInput"]:focus-within [data-testid="stChatInputSubmitButton"] svg {
    fill: #A855F7 !important;
}

/* ── Expanders ─────────────────────────────────────────────────────── */
[data-testid="stExpander"] {
    background: rgba(168, 85, 247, 0.05) !important;
    border: 1px solid rgba(168, 85, 247, 0.15) !important;
    border-radius: 10px !important;
}
[data-testid="stExpander"] summary,
[data-testid="stExpander"] summary span {
    color: #A855F7 !important;
}
[data-testid="stExpander"] .stMarkdown * {
    color: #CBD5E1 !important;
}

/* ── Source captions ───────────────────────────────────────────────── */
[data-testid="stCaptionContainer"] * {
    color: rgba(100, 116, 139, 0.85) !important;
}

/* ── Sidebar ───────────────────────────────────────────────────────── */
[data-testid="stSidebar"] {
    background: #0D0D12 !important;
    border-left: 3px solid #7C3AED !important;
    border-right: 1px solid rgba(168, 85, 247, 0.12) !important;
}
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] span,
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3,
[data-testid="stSidebar"] li,
[data-testid="stSidebar"] small,
[data-testid="stSidebar"] .stMarkdown {
    color: #E2E8F0 !important;
}
[data-testid="stSidebar"] .stButton button {
    border: 1px solid rgba(168, 85, 247, 0.35) !important;
    background: rgba(168, 85, 247, 0.08) !important;
    color: #C084FC !important;
    border-radius: 8px !important;
    font-size: 0.88rem !important;
    transition: all 0.15s ease !important;
}
[data-testid="stSidebar"] .stButton button:hover {
    background: rgba(168, 85, 247, 0.2) !important;
    border-color: #A855F7 !important;
    color: #EDE9FE !important;
}
[data-testid="stSidebar"] hr {
    border-color: rgba(168, 85, 247, 0.15) !important;
}

/* ── Iframe (orb component) ────────────────────────────────────────── */
[data-testid="stCustomComponentV1"] iframe {
    border: none !important;
    display: block !important;
}

/* ── Mode toggle — purple active track ─────────────────────────────── */
[data-testid="stToggleSwitch"] [role="switch"] {
    background-color: rgba(88, 28, 135, 0.3) !important;
    border-color: rgba(168, 85, 247, 0.3) !important;
}
[data-testid="stToggleSwitch"] [role="switch"][aria-checked="true"] {
    background-color: #A855F7 !important;
    border-color: #A855F7 !important;
}

/* ── Custom scrollbar ──────────────────────────────────────────────── */
::-webkit-scrollbar { width: 4px; }
::-webkit-scrollbar-track { background: #0D0D12; }
::-webkit-scrollbar-thumb { background: #3B0764; border-radius: 2px; }
::-webkit-scrollbar-thumb:hover { background: #7C3AED; }

/* ── Mobile ────────────────────────────────────────────────────────── */
@media (max-width: 640px) {
    .main .block-container {
        padding-left: 0.75rem !important;
        padding-right: 0.75rem !important;
    }
    [data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]),
    [data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-assistant"]) {
        margin-left: 0 !important;
        margin-right: 0 !important;
    }
}
</style>
"""

# ── Business logic (unchanged — all tests pass against these) ──────────────────

@st.cache_resource
def get_collection():
    logger.info("Connecting to ChromaDB at %s:%s", CHROMA_HOST, CHROMA_PORT)
    chroma_client = chromadb.HttpClient(host=CHROMA_HOST, port=CHROMA_PORT)
    collection = chroma_client.get_or_create_collection(COLLECTION_NAME)
    logger.info("Collection '%s' loaded — %d chunk(s) stored", COLLECTION_NAME, collection.count())
    return collection


def embed_query(text: str) -> list[float]:
    logger.debug("Embedding query: %.80s", text)
    return embed_model.encode(text, normalize_embeddings=True).tolist()


def retrieve(collection, question: str) -> tuple[list[str], list[str]]:
    count = collection.count()
    if count == 0:
        raise ValueError(
            "No documents have been ingested yet. "
            "Add files to the documents/ folder and run add-documents.command first."
        )
    n = min(TOP_K, count)
    logger.info("Retrieving top %d chunk(s) for query: %.80s", n, question)
    results = collection.query(
        query_embeddings=[embed_query(question)],
        n_results=n,
        include=["documents", "metadatas"],
    )
    chunks = results["documents"][0]
    sources = [m["source"] for m in results["metadatas"][0]]
    logger.debug("Retrieved %d chunk(s) from source(s): %s", len(chunks), set(sources))
    return chunks, sources


def build_prompt(question: str, chunks: list[str], sources: list[str]) -> str:
    context_blocks = "\n\n".join(
        f"[Source: {src}]\n{chunk}" for src, chunk in zip(sources, chunks)
    )
    return f"Context:\n{context_blocks}\n\nQuestion: {question}"


def parse_response(content: str) -> tuple[str, str]:
    """Split a Qwen3 response into (thinking, answer)."""
    if "<think>" in content and "</think>" in content:
        thinking = content.split("<think>", 1)[1].split("</think>", 1)[0].strip()
        answer = content.split("</think>", 1)[1].strip()
        return thinking, answer
    return "", content


def summarize_thinking(thinking: str) -> str:
    logger.info("Summarizing model reasoning (%d chars)", len(thinking))
    response = requests.post(
        GROQ_URL,
        headers={"Authorization": f"Bearer {GROQ_API_KEY}"},
        json={
            "model": SUMMARY_MODEL,
            "messages": [
                {"role": "system", "content": REASONING_SUMMARY_PROMPT},
                {"role": "user", "content": thinking},
            ],
        },
    )
    if not response.ok:
        logger.warning("Reasoning summarization failed: %s", response.status_code)
        return ""
    return response.json()["choices"][0]["message"]["content"]


def ask(
    collection, question: str, include_reasoning: bool = False
) -> tuple[str, list[str], str, list[tuple[str, str]]]:
    chunks, sources = retrieve(collection, question)
    logger.info("Calling Groq (%s)", CHAT_MODEL)
    response = requests.post(
        GROQ_URL,
        headers={"Authorization": f"Bearer {GROQ_API_KEY}"},
        json={
            "model": CHAT_MODEL,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": build_prompt(question, chunks, sources)},
            ],
        },
    )
    if not response.ok:
        logger.error("Groq request failed: %s %s", response.status_code, response.reason)
        raise Exception(f"{response.status_code} {response.reason}: {response.json()}")
    content = response.json()["choices"][0]["message"]["content"]
    raw_thinking, answer = parse_response(content)
    reasoning = summarize_thinking(raw_thinking) if include_reasoning and raw_thinking else ""
    unique_sources = list(dict.fromkeys(sources))
    logger.info("Answer received — sources: %s", unique_sources)
    return answer, unique_sources, reasoning, list(zip(chunks, sources))


def stream_answer_tokens(
    question: str,
    chunks: list[str],
    sources: list[str],
    thinking_sink: list[str],
    system_prompt: str = SYSTEM_PROMPT,
):
    """Yields answer tokens, buffering the <think> block silently."""
    user_content = build_prompt(question, chunks, sources) if chunks else question
    response = requests.post(
        GROQ_URL,
        headers={"Authorization": f"Bearer {GROQ_API_KEY}"},
        json={
            "model": CHAT_MODEL,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ],
            "stream": True,
        },
        stream=True,
    )
    if not response.ok:
        logger.error("Groq stream failed: %s %s", response.status_code, response.reason)
        raise Exception(f"{response.status_code} {response.reason}: {response.json()}")

    buffer = ""
    in_think = False
    think_done = False

    for line in response.iter_lines():
        if not line:
            continue
        text = line.decode("utf-8")
        if not text.startswith("data: "):
            continue
        payload = text[6:]
        if payload == "[DONE]":
            break
        try:
            token = json.loads(payload)["choices"][0]["delta"].get("content", "")
        except (KeyError, json.JSONDecodeError):
            continue
        if not token:
            continue

        buffer += token

        if not in_think and not think_done:
            if "<think>" in buffer:
                in_think = True
                buffer = buffer.split("<think>", 1)[1]
            elif len(buffer) > 15 and "<" not in buffer:
                think_done = True
                yield buffer
                buffer = ""
        elif in_think:
            if "</think>" in buffer:
                think_text, answer_part = buffer.split("</think>", 1)
                thinking_sink.append(think_text)
                buffer = answer_part.lstrip("\n")
                in_think = False
                think_done = True
                if buffer:
                    yield buffer
                    buffer = ""
        else:
            yield buffer
            buffer = ""

    if buffer:
        yield buffer


# ── UI helpers ─────────────────────────────────────────────────────────────────

def render_message(msg: dict, show_reasoning: bool) -> None:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("timestamp"):
            st.markdown(
                f'<span class="msg-timestamp">{msg["timestamp"]}</span>',
                unsafe_allow_html=True,
            )
        if show_reasoning and msg.get("reasoning"):
            with st.expander("Thought Train"):
                st.markdown(msg["reasoning"])
        if msg.get("sources"):
            st.caption("Sources: " + " · ".join(msg["sources"]))
        for i, (chunk, source) in enumerate(msg.get("chunk_sources", []), 1):
            with st.expander(f"Excerpt {i} — {source}"):
                st.write(chunk)


# ── Page setup ─────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Internal Knowledge Retrieval Agent",
    page_icon="🔮",
    layout="wide",
    initial_sidebar_state="expanded",
)
st.markdown(CSS, unsafe_allow_html=True)

# Streamlit's Emotion CSS is injected into <head> at runtime and beats any <style>
# placed in the body by st.markdown(). Inject our width rule into the parent
# document's <head> via components iframe so it comes last and wins the cascade.
st.components.v1.html("""
<script>
(function () {
  var s = window.parent.document.getElementById('_ikra_width_fix');
  if (!s) {
    s = window.parent.document.createElement('style');
    s.id = '_ikra_width_fix';
    window.parent.document.head.appendChild(s);
  }
  s.textContent =
    '[data-testid="stMainBlockContainer"],' +
    '[data-testid="stBottomBlockContainer"] {' +
    '  max-width: 760px !important;' +
    '  width: 760px !important;' +
    '  margin-left: auto !important;' +
    '  margin-right: auto !important;' +
    '}';
})();
</script>
""", height=0)

# ── Sidebar ────────────────────────────────────────────────────────────────────

if "strict_mode" not in st.session_state:
    st.session_state["strict_mode"] = True

with st.sidebar:
    st.markdown(
        """
        <div style="padding:1.4rem 0 0.8rem;">
          <div style="display:flex;align-items:center;gap:10px;margin-bottom:0.5rem;">
            <div style="width:26px;height:26px;border-radius:50%;
                        background:linear-gradient(135deg,#7C3AED,#A855F7);
                        box-shadow:0 0 14px rgba(168,85,247,0.55);flex-shrink:0;"></div>
            <span style="font-size:0.92rem;font-weight:600;color:#E2E8F0;letter-spacing:-0.01em;">
              Knowledge Agent
            </span>
          </div>
          <p style="font-size:0.76rem;color:rgba(100,116,139,0.85);margin:0;line-height:1.55;">
            Grounded answers from your internal document library.
          </p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.divider()
    st.markdown(
        """
        <p style="font-size:0.72rem;font-weight:600;color:#4B5563;
                  text-transform:uppercase;letter-spacing:0.07em;margin-bottom:0.65rem;">
          How to use
        </p>
        <ul style="font-size:0.79rem;color:#64748B;padding-left:1.1rem;
                   line-height:1.85;margin:0 0 0.5rem;">
          <li>Ask a question in plain English</li>
          <li>Answers are grounded in your documents</li>
          <li>Toggle Thought Train to see reasoning</li>
        </ul>
        """,
        unsafe_allow_html=True,
    )
    st.divider()
    show_reasoning = st.toggle("Thought Train", value=False)
    msg_count = len(st.session_state.get("messages", []))
    st.markdown(
        f'<p style="font-size:0.76rem;color:#374151;margin:0.6rem 0 0.8rem;">'
        f'{msg_count} message{"s" if msg_count != 1 else ""} this session</p>',
        unsafe_allow_html=True,
    )
    st.divider()
    _mode_label = "Strict Mode 🔒" if st.session_state.strict_mode else "Free Mode 🔓"
    strict_mode = st.toggle(_mode_label, key="strict_mode")
    _mode_desc = (
        "Answers limited to the knowledge base"
        if strict_mode
        else "Answers anything, knowledge base not enforced"
    )
    st.markdown(
        f'<p style="font-size:0.72rem;color:rgba(100,116,139,0.7);margin:0.25rem 0 0.5rem;">'
        f"{_mode_desc}</p>",
        unsafe_allow_html=True,
    )
    st.divider()
    if st.button("Clear conversation", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# ── Mode badge — injected above the chat input in the bottom bar ───────────────

_badge_label = "Strict Mode 🔒" if st.session_state.strict_mode else "Free Mode 🔓"
st.components.v1.html(f"""
<script>
(function() {{
  var bottom = window.parent.document.querySelector('[data-testid="stBottomBlockContainer"]');
  if (!bottom) return;
  var badge = window.parent.document.getElementById('_ikra_mode_badge');
  if (!badge) {{
    badge = window.parent.document.createElement('div');
    badge.id = '_ikra_mode_badge';
    badge.style.cssText = 'text-align:center;padding:6px 0 4px;';
    bottom.insertBefore(badge, bottom.firstChild);
  }}
  badge.innerHTML = '<span style="display:inline-block;padding:3px 14px;'
    + 'background:rgba(109,40,217,0.3);border:1px solid rgba(168,85,247,0.5);'
    + 'border-radius:20px;font-size:0.7rem;color:rgba(221,214,254,0.8);'
    + 'font-family:Inter,system-ui,sans-serif;letter-spacing:0.03em;">'
    + '{_badge_label}</span>';
}})();
</script>
""", height=0)

# ── App state & collection ─────────────────────────────────────────────────────

collection = get_collection()

if "messages" not in st.session_state:
    st.session_state.messages = []

# ── Hero orb — only shown on empty state so it doesn't push chat down ──────────

if not st.session_state.messages:
    st.components.v1.html(ORB_HTML, height=560)

# ── Chat history ───────────────────────────────────────────────────────────────

for msg in st.session_state.messages:
    render_message(msg, show_reasoning)

# ── Chat input ─────────────────────────────────────────────────────────────────

if question := st.chat_input("Ask your knowledge base…"):
    user_ts = datetime.now().strftime("%I:%M %p")
    user_msg = {
        "role": "user",
        "content": question,
        "timestamp": user_ts,
        "sources": [],
        "reasoning": "",
        "chunk_sources": [],
    }
    st.session_state.messages.append(user_msg)
    render_message(user_msg, show_reasoning)

    answer = ""
    sources = []
    reasoning = ""
    chunk_sources = []

    with st.chat_message("assistant"):
        response_slot = st.empty()
        try:
            if st.session_state.get("strict_mode", True):
                with st.spinner("Searching knowledge base…"):
                    retrieved_chunks, retrieved_sources = retrieve(collection, question)
                active_prompt = SYSTEM_PROMPT
            else:
                retrieved_chunks, retrieved_sources = [], []
                active_prompt = FREE_SYSTEM_PROMPT

            thinking_sink: list[str] = []

            # Typing indicator occupies the slot until the first token arrives
            response_slot.markdown(TYPING_HTML, unsafe_allow_html=True)

            full_answer = ""
            for token in stream_answer_tokens(
                question, retrieved_chunks, retrieved_sources, thinking_sink, active_prompt
            ):
                full_answer += token
                # Streaming cursor gives typewriter feel; markdown re-renders via React diff
                response_slot.markdown(full_answer + " ▌")

            response_slot.markdown(full_answer if full_answer else "*(no response)*")
            answer = full_answer

            sources = list(dict.fromkeys(retrieved_sources))
            chunk_sources = list(zip(retrieved_chunks, retrieved_sources))

            if show_reasoning and thinking_sink:
                with st.spinner("Summarizing reasoning…"):
                    reasoning = summarize_thinking("".join(thinking_sink))

        except Exception as e:
            logger.exception("Failed to answer question: %.80s", question)
            answer = f"Something went wrong: {e}"
            response_slot.error(answer)

        assistant_ts = datetime.now().strftime("%I:%M %p")
        st.markdown(
            f'<span class="msg-timestamp">{assistant_ts}</span>',
            unsafe_allow_html=True,
        )
        if show_reasoning and reasoning:
            with st.expander("Thought Train"):
                st.markdown(reasoning)
        if sources:
            st.caption("Sources: " + " · ".join(sources))
        for i, (chunk, source) in enumerate(chunk_sources, 1):
            with st.expander(f"Excerpt {i} — {source}"):
                st.write(chunk)

    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "timestamp": assistant_ts,
        "sources": sources,
        "reasoning": reasoning,
        "chunk_sources": chunk_sources,
    })

    # Scroll to latest message — height=0 keeps it invisible
    st.components.v1.html(
        """<script>
        var main = window.parent.document.querySelector('[data-testid="stMain"]');
        if (main) main.scrollTo({ top: main.scrollHeight, behavior: 'smooth' });
        </script>""",
        height=0,
    )
