"""Static-asset loaders and parent-document <head> injectors.

`apply_global_styles` is enough for most CSS, but Streamlit's Emotion
runtime injects styles into <head> AFTER our `st.markdown(<style>)` block
in the body, so any rule that competes with theirs (notably the layout
width) loses the cascade. The `inject_*` helpers run a small script that
appends a <style> element to the parent document's <head> from a
components iframe — guaranteeing our rules come last and win.
"""

from pathlib import Path

import streamlit as st

STATIC_DIR = Path(__file__).resolve().parent.parent / "static"


def load_static(name: str) -> str:
    return (STATIC_DIR / name).read_text(encoding="utf-8")


def apply_global_styles() -> None:
    css = load_static("styles.css")
    st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)


# Streamlit's Emotion CSS is injected into <head> at runtime and beats any <style>
# placed in the body by st.markdown(). Inject the width rule into the parent
# document's <head> via a components iframe so it comes last and wins the cascade.
def inject_width_fix() -> None:
    st.components.v1.html(
        """
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
""",
        height=0,
    )


def inject_mode_badge(label: str) -> None:
    st.components.v1.html(
        f"""
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
    + '{label}</span>';
}})();
</script>
""",
        height=0,
    )


def scroll_to_latest() -> None:
    st.components.v1.html(
        """<script>
        var main = window.parent.document.querySelector('[data-testid="stMain"]');
        if (main) main.scrollTo({ top: main.scrollHeight, behavior: 'smooth' });
        </script>""",
        height=0,
    )
