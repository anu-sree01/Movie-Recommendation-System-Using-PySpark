"""Full-screen, one-time opening animation overlay for CineSphere."""

from __future__ import annotations

import base64
from pathlib import Path

import streamlit as st


HTML = '<span aria-hidden="true"></span>'
JS = r"""
export default function(component) {
  const { data, setStateValue } = component;
  if (sessionStorage.getItem('cinesphere-opening-played') === '1') {
    setStateValue('done', true);
    return;
  }
  const overlay = document.createElement('div');
  overlay.className = 'cs-opening-overlay';
  overlay.setAttribute('role', 'presentation');
  Object.assign(overlay.style, {
    position: 'fixed',
    inset: '0',
    zIndex: '2147483647',
    display: 'grid',
    placeItems: 'center',
    overflow: 'hidden',
    background: '#05070d',
    opacity: '1',
    transition: 'opacity 650ms ease',
  });
  const video = document.createElement('video');
  video.className = 'cs-opening-video';
  Object.assign(video.style, {
    display: 'block',
    width: '100%',
    height: '100%',
    objectFit: 'contain',
    background: '#05070d',
  });
  video.autoplay = true;
  video.muted = true;
  video.playsInline = true;
  video.preload = 'auto';
  video.src = `data:video/mp4;base64,${data.videoBase64}`;
  overlay.appendChild(video);

  const skip = document.createElement('button');
  skip.type = 'button';
  skip.textContent = 'Skip intro';
  skip.setAttribute('aria-label', 'Skip CineSphere introduction');
  Object.assign(skip.style, {
    position: 'absolute',
    right: 'max(1rem, env(safe-area-inset-right))',
    bottom: 'max(1rem, env(safe-area-inset-bottom))',
    zIndex: '1',
    border: '1px solid rgba(255,255,255,.42)',
    borderRadius: '999px',
    padding: '.7rem 1rem',
    color: '#fff',
    background: 'rgba(12,13,20,.88)',
    font: '600 .9rem system-ui,sans-serif',
    cursor: 'pointer',
  });
  overlay.appendChild(skip);
  document.body.appendChild(overlay);

  let finished = false;
  const finish = () => {
    if (finished) return;
    finished = true;
    sessionStorage.setItem('cinesphere-opening-played', '1');
    overlay.classList.add('is-leaving');
    overlay.style.opacity = '0';
    window.setTimeout(() => {
      overlay.remove();
      setStateValue('done', true);
    }, 650);
  };
  skip.addEventListener('click', finish, { once: true });
  video.addEventListener('ended', finish, { once: true });
  video.addEventListener('error', finish, { once: true });
  window.setTimeout(finish, 5000);
  video.play().catch(finish);
  return () => { if (!finished) overlay.remove(); };
}
"""

_RENDERER = None


def register_opening_animation_component():
    global _RENDERER
    _RENDERER = st.components.v2.component(
        "cinesphere_opening_animation",
        html=HTML,
        js=JS,
        isolate_styles=True,
    )
    return _RENDERER


def show_opening_animation(video_path: Path) -> None:
    """Mount the overlay once per Streamlit session; the browser remembers the tab too."""
    if st.session_state.get("cinesphere_intro_seen", False):
        return
    if not video_path.is_file():
        st.session_state.cinesphere_intro_seen = True
        return

    renderer = _RENDERER or register_opening_animation_component()
    video_base64 = base64.b64encode(video_path.read_bytes()).decode("ascii")

    def mark_intro_seen():
        st.session_state.cinesphere_intro_seen = True

    result = renderer(
        data={"videoBase64": video_base64},
        key="cs-opening-animation",
        default={"done": False},
        on_done_change=mark_intro_seen,
        height=0,
    )
    if result.get("done"):
        st.session_state.cinesphere_intro_seen = True
