"""Poster cards with verified, muted YouTube previews on hover or tap."""

from __future__ import annotations

import streamlit as st


HTML = """
<div class="cs-preview" tabindex="0">
  <img class="cs-preview-poster" alt="">
  <div class="cs-preview-fallback" aria-hidden="true"><span>CINESPHERE</span><small>Artwork unavailable</small></div>
  <div class="cs-preview-player" hidden></div>
  <button class="cs-preview-trigger" type="button" aria-label="Play muted trailer preview">▶ Preview</button>
</div>
"""

CSS = """
:host { display:block; width:100%; }
.cs-preview { position:relative; width:100%; aspect-ratio:2/3; overflow:hidden; border-radius:12px; background:radial-gradient(circle at 50% 30%,rgba(168,121,255,.28),transparent 44%),linear-gradient(155deg,#252235,#121720 72%,#12232a); }
.cs-preview-poster, .cs-preview-player, .cs-preview-player iframe, .cs-preview-fallback { position:absolute; inset:0; width:100%; height:100%; }
.cs-preview-poster { z-index:1; object-fit:cover; transition:transform .35s ease,opacity .25s ease; }
.cs-preview-player { z-index:2; background:transparent; }
.cs-preview-player iframe { border:0; opacity:.93; animation:cs-preview-in .25s ease both; }
.cs-preview-fallback { z-index:0; display:flex; flex-direction:column; align-items:center; justify-content:center; gap:.5rem; color:#ddd6ed; font:600 .9rem system-ui,sans-serif; }
.cs-preview-fallback small { color:#a5a6b4; font:400 .72rem system-ui,sans-serif; }
.cs-preview-trigger { position:absolute; z-index:3; left:50%; bottom:.75rem; transform:translate(-50%,8px); opacity:0; pointer-events:none; border:1px solid rgba(255,255,255,.26); border-radius:999px; padding:.48rem .85rem; color:#fff; background:rgba(12,13,20,.88); font:600 .76rem system-ui,sans-serif; cursor:pointer; transition:opacity .2s ease,transform .2s ease; }
.cs-preview:hover .cs-preview-poster, .cs-preview:focus-within .cs-preview-poster { transform:scale(1.045); }
.cs-preview:hover .cs-preview-trigger, .cs-preview:focus-within .cs-preview-trigger { opacity:1; transform:translate(-50%,0); pointer-events:auto; }
@media (hover:none) { .cs-preview-trigger { opacity:1; transform:translate(-50%,0); pointer-events:auto; } }
@media (prefers-reduced-motion:reduce) { *,*::before,*::after { transition:none !important; animation:none !important; } }
@keyframes cs-preview-in { from { opacity:0; } to { opacity:.93; } }
"""

JS = """
export default function(component) {
  const { parentElement, data } = component;
  const root = parentElement;
  const card = root.querySelector('.cs-preview');
  const poster = root.querySelector('.cs-preview-poster');
  const player = root.querySelector('.cs-preview-player');
  const button = root.querySelector('.cs-preview-trigger');
  if (!card || !poster || !player || !button) return;

  if (data.posterUrl) {
    poster.src = data.posterUrl;
    poster.alt = data.title ? `${data.title} poster` : 'Movie poster';
    poster.loading = 'lazy';
    poster.decoding = 'async';
    poster.hidden = false;
  } else {
    poster.hidden = true;
  }

  let activeFrame = null;
  const stop = () => {
    if (activeFrame) {
      activeFrame.src = 'about:blank';
      activeFrame.remove();
      activeFrame = null;
    }
    player.hidden = true;
  };
  const start = () => {
    if (!data.videoId || activeFrame) return;
    const frame = document.createElement('iframe');
    frame.src = `https://www.youtube-nocookie.com/embed/${encodeURIComponent(data.videoId)}?autoplay=1&mute=1&controls=0&playsinline=1&loop=1&playlist=${encodeURIComponent(data.videoId)}&rel=0`;
    frame.title = data.title ? `${data.title} muted trailer preview` : 'Muted trailer preview';
    frame.allow = 'autoplay; encrypted-media; picture-in-picture';
    frame.allowFullscreen = true;
    activeFrame = frame;
    player.replaceChildren(frame);
    player.hidden = false;
  };
  const hasHover = window.matchMedia('(hover: hover) and (pointer: fine)').matches;
  const toggle = (event) => {
    event.preventDefault();
    event.stopPropagation();
    if (hasHover) {
      start();
      return;
    }
    if (activeFrame) stop(); else start();
  };
  const onFocusOut = (event) => {
    if (!card.contains(event.relatedTarget)) stop();
  };
  if (data.videoId) {
    card.addEventListener('mouseenter', start);
    card.addEventListener('mouseleave', stop);
    card.addEventListener('focusin', start);
    card.addEventListener('focusout', onFocusOut);
    button.addEventListener('click', toggle);
  } else {
    button.hidden = true;
  }
  return () => {
    card.removeEventListener('mouseenter', start);
    card.removeEventListener('mouseleave', stop);
    card.removeEventListener('focusin', start);
    card.removeEventListener('focusout', onFocusOut);
    button.removeEventListener('click', toggle);
    stop();
  };
}
"""


_RENDERER = None


def register_movie_preview_component():
    """Register once per Streamlit script run, then reuse for all visible cards."""
    global _RENDERER
    _RENDERER = st.components.v2.component(
        "cinesphere_movie_preview",
        html=HTML,
        css=CSS,
        js=JS,
        isolate_styles=True,
    )
    return _RENDERER


def render_movie_preview(*, movie_id: int, title: str, poster_url: str | None,
                         video_id: str | None, key_suffix: str = "card"):
    renderer = _RENDERER or register_movie_preview_component()
    data = {
        "title": str(title),
        "posterUrl": str(poster_url) if poster_url else "",
        "videoId": str(video_id) if video_id else "",
    }
    renderer(data=data, key=f"cs-preview-{key_suffix}-{int(movie_id)}")
