"""CineSphere: MovieLens discovery, pilot ALS recommendations, and user lists."""

from __future__ import annotations

import html
import math
import os
import re
import sys
import time
from pathlib import Path
from urllib.parse import parse_qs, urlencode, urlparse

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
OPENING_VIDEO = Path(__file__).resolve().parent / "media" / "assets" / "CineSphere_logo_animation_no_gemini.mp4"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from member3_dashboard.utils import database as db
from member3_dashboard.utils.catalog import (
    DataContractError,
    load_als,
    load_genre_movies,
    load_metrics,
    load_movies,
    load_popular,
)
from member3_dashboard.utils.media import load_media
from member3_dashboard.utils.media import METADATA_FILE as MEDIA_METADATA_FILE
# Keep OMDb title matching in the dashboard utilities so hot reloads don't depend
# on the command-line acquisition module's import cache.
from member3_dashboard.utils.on_demand_poster import fetch_visible_movies
from member3_dashboard.utils.recommendations import (
    genre_discovery,
    personalized,
    pilot_user_ids,
    popular_fallback,
)
from member3_dashboard.ui.movie_preview import register_movie_preview_component, render_movie_preview
from member3_dashboard.ui.opening_animation import (
    register_opening_animation_component,
    show_opening_animation,
)


st.set_page_config(page_title="CineSphere", page_icon="🎬", layout="wide", initial_sidebar_state="expanded")


def inject_styles():
    css_file = Path(__file__).resolve().parent / "frontend" / "styles.css"
    st.markdown(f"<style>{css_file.read_text(encoding='utf-8')}</style>", unsafe_allow_html=True)


def title_banner(title: str, subtitle: str = ""):
    st.markdown(f'<div class="eyebrow">CINESPHERE · MOVIE DISCOVERY</div><h1>{html.escape(title)}</h1>', unsafe_allow_html=True)
    if subtitle:
        st.caption(subtitle)


def _safe_int(value):
    try:
        return int(value)
    except (ValueError, TypeError, OverflowError):
        return None


ARTICLE_WORDS = {"the", "a", "an", "les", "le", "la", "el", "das", "der", "die"}


def display_title(value) -> str:
    title = str(value or "").strip()
    title = re.sub(r"\s*\((?:19|20)\d{2}\)\s*$", "", title).strip()
    match = re.search(r",\s*([A-Za-z]+)(?=\s*(?:\([^()]*\))?$)", title)
    if match and match.group(1).casefold() in ARTICLE_WORDS:
        suffix = title[match.end():]
        title = f"{match.group(1)} {title[:match.start()].strip()}{suffix}"
    return title


def display_genres(value) -> str:
    values = [part.strip() for part in str(value or "").split("|")]
    return " · ".join(part for part in values if part and part.casefold() != "unknown" and part != "(no genres listed)")


def _normalized_search_title(value: str) -> str:
    text = display_title(value).casefold()
    words = re.findall(r"[a-z0-9]+", text)
    if words and words[0] in ARTICLE_WORDS:
        words = words[1:]
    return " ".join(words)


def fetch_page_media(items, media: dict) -> None:
    if os.environ.get("CINESPHERE_DISABLE_ON_DEMAND") == "1":
        return
    remaining = int(st.session_state.get("omdb_prefetch_remaining", 48))
    deadline = float(st.session_state.get("omdb_prefetch_deadline", time.monotonic()))
    if remaining <= 0 or time.monotonic() >= deadline:
        return
    candidates = []
    seen = set()
    for item in items:
        item = item.to_dict() if hasattr(item, "to_dict") else dict(item)
        movie_id = _safe_int(item.get("movieId"))
        if movie_id is not None and movie_id not in media and movie_id not in seen:
            seen.add(movie_id)
            candidates.append(item)
    candidates = candidates[:remaining]
    if not candidates:
        return
    with st.spinner("Finding artwork for movies on this page…"):
        submitted = fetch_visible_movies(candidates, media,
                                         max_wait=max(0.0, min(4.3, deadline-time.monotonic())),
                                         max_requests=remaining)
    st.session_state.omdb_prefetch_remaining = remaining - submitted


def movie_dict(movie_id: int, movies_by_id: dict, als: pd.DataFrame | None = None) -> dict:
    base = dict(movies_by_id.get(int(movie_id), {}))
    if als is not None and not als.empty:
        extra = als.loc[als["movieId"].eq(int(movie_id))]
        if not extra.empty:
            base.update(extra.iloc[0].to_dict())
    return base


def trailer_action(movie: dict, asset: dict, key: str):
    trailer_status = str(asset.get("trailer_status", "")).lower()
    trailer_url = asset.get("trailer_url")
    if trailer_status == "verified" and trailer_url:
        with st.expander("▶ Watch Trailer"):
            st.video(trailer_url)
    elif trailer_status in {"unverified", "candidate_unverified"} and trailer_url:
        st.link_button("Trailer (unverified source)", trailer_url, key=f"trailer-view-{key}", use_container_width=True)
    elif trailer_status == "search_only" and trailer_url:
        st.link_button("Search Trailers", trailer_url, key=f"trailer-search-{key}", use_container_width=True)
    elif trailer_status == "link_only" and trailer_url:
        is_search = str(asset.get("trailer_source", "")).startswith("YouTube search link")
        link_label = "Search Trailers" if is_search else "Watch on YouTube"
        st.link_button(link_label, trailer_url, key=f"trailer-link-{key}", use_container_width=True)
        if not is_search:
            st.caption("Trailer (unverified source)")
    elif trailer_status == "unavailable_confirmed":
        st.button("Trailer Unavailable", key=f"trailer-none-{key}", disabled=True, use_container_width=True)
        query = f"{movie.get('title', '')} {movie.get('year', '')} official trailer".strip()
        search_url = "https://www.youtube.com/results?" + urlencode({"search_query": query})
        st.link_button("Search on YouTube", search_url, key=f"trailer-search-unavailable-{key}", use_container_width=True)
    else:
        query = f"{movie.get('title', '')} {movie.get('year', '')} official trailer".strip()
        search_url = "https://www.youtube.com/results?" + urlencode({"search_query": query})
        st.link_button("Search Trailers", search_url, key=f"trailer-generic-search-{key}", use_container_width=True)


def poster_status_note(asset: dict, key: str):
    status = str(asset.get("poster_status", "")).lower()
    source = asset.get("poster_source")
    if status == "permission_required":
        st.caption("A matching poster source was found, but reuse rights are not cleared.")
    elif status == "mismatch_review":
        st.caption("Poster title/year differs from the MovieLens record; hidden pending review.")
    elif status == "not_suitable":
        st.caption("A matching image was found, but it is not a suitable movie poster.")
    elif status in {"not_found_in_search", "no_tmdb_id", "not_available"}:
        st.caption("No usable poster is available in the current media records.")
    if source.startswith(("https://", "http://")) and status != "verified":
        st.link_button("Review poster source", source, key=f"poster-source-{key}")


def poster_credit(asset: dict):
    source = str(asset.get("poster_source", ""))
    if source.lower().startswith("omdb"):
        st.caption("Poster source: OMDb")


def has_poster(asset: dict) -> bool:
    return str(asset.get("poster_status", "")).lower() in {"verified", "ok"} and bool(asset.get("poster_url"))


def verified_youtube_id(asset: dict) -> str | None:
    if str(asset.get("trailer_status", "")).lower() != "verified":
        return None
    url = str(asset.get("trailer_url", ""))
    parsed = urlparse(url)
    host = (parsed.hostname or "").lower()
    video_id = ""
    if host in {"youtu.be", "www.youtu.be"}:
        video_id = parsed.path.strip("/").split("/")[0]
    elif host in {"youtube.com", "www.youtube.com", "m.youtube.com", "youtube-nocookie.com", "www.youtube-nocookie.com"}:
        if parsed.path == "/watch":
            video_id = parse_qs(parsed.query).get("v", [""])[0]
        elif parsed.path.startswith(("/embed/", "/shorts/")):
            video_id = parsed.path.split("/")[2]
    return video_id if re.fullmatch(r"[A-Za-z0-9_-]{11}", video_id) else None


def _hero_interacted(state_key: str):
    st.session_state[f"{state_key}_last_interaction"] = time.monotonic()


def _step_hero(state_key: str, movie_ids: list[int], direction: int):
    if not movie_ids:
        return
    try:
        index = movie_ids.index(int(st.session_state.get(state_key, movie_ids[0])))
    except (ValueError, TypeError):
        index = 0
    st.session_state[state_key] = movie_ids[(index + direction) % len(movie_ids)]
    _hero_interacted(state_key)


@st.fragment(run_every="75s")
def render_home_hero(user: dict, popular: pd.DataFrame, movies_by_id: dict, media: dict):
    user_id = int(user["application_user_id"])
    movie_ids = [int(value) for value in popular.head(30)["movieId"].tolist()]
    if not movie_ids:
        st.info("The MovieLens catalogue has no featured movies to show yet.")
        return
    state_key = f"hero_featured_movie_{user_id}"
    auto_key = f"hero_auto_rotate_{user_id}"
    requested_feature = _safe_int(st.session_state.pop(f"hero_requested_feature_{user_id}", None))
    if requested_feature in movies_by_id:
        st.session_state[state_key] = requested_feature
    selected = _safe_int(st.session_state.get(state_key))
    if selected is not None and selected in movies_by_id and selected not in movie_ids:
        movie_ids.append(selected)
    default_id = next((mid for mid in movie_ids if verified_youtube_id(media.get(mid, {}))), movie_ids[0])
    if st.session_state.get(state_key) not in movie_ids:
        st.session_state[state_key] = default_id
    st.session_state.setdefault(auto_key, True)
    now = time.monotonic()
    last_interaction = float(st.session_state.get(f"{state_key}_last_interaction", now))
    if st.session_state[auto_key] and now - last_interaction >= 75:
        _step_hero(state_key, movie_ids, 1)

    selected_id = int(st.session_state[state_key])
    featured = movie_dict(selected_id, movies_by_id)
    asset = media.get(selected_id, {})
    video_id = verified_youtube_id(asset)
    poster_url = asset.get("poster_url") if has_poster(asset) else None
    year = featured.get("year")
    year_text = str(int(year)) if pd.notna(year) and str(year).replace(".", "", 1).isdigit() else ""
    genres = display_genres(featured.get("genres", ""))

    with st.container(key="cs-hero"):
        if poster_url:
            with st.container(key="cs-hero-poster"):
                st.image(poster_url, use_container_width=True)
        if video_id:
            embed_url = f"https://www.youtube-nocookie.com/embed/{video_id}?autoplay=1&mute=1&controls=1&playsinline=1&loop=1&playlist={video_id}&rel=0"
            with st.container(key="cs-hero-video"):
                st.iframe(embed_url, width="stretch", height=470)
        st.markdown('<div class="cs-hero-shade"></div>', unsafe_allow_html=True)
        with st.container(key="cs-hero-content"):
            st.markdown('<div class="eyebrow">CINESPHERE FEATURE</div>', unsafe_allow_html=True)
            st.markdown(f'<h1 class="cs-hero-title">{html.escape(display_title(featured.get("title", "Featured movie")))}</h1>', unsafe_allow_html=True)
            metadata = " · ".join(value for value in (year_text, genres) if value)
            st.markdown(f'<div class="cs-hero-meta">{html.escape(metadata)}</div>', unsafe_allow_html=True)
            rating_count = pd.to_numeric(featured.get("rating_count"), errors="coerce")
            if pd.notna(rating_count) and rating_count == 0:
                st.caption("No MovieLens ratings yet")
            elif pd.notna(featured.get("avg_rating")) and pd.notna(rating_count):
                st.caption(f"MovieLens historical rating · {float(featured['avg_rating']):.2f} / 5 · {int(rating_count):,} ratings")
            overview = asset.get("overview")
            if overview and pd.notna(overview):
                st.markdown(f'<p class="cs-hero-copy">{html.escape(str(overview))}</p>', unsafe_allow_html=True)
            if video_id:
                st.caption("Muted verified trailer preview · use the player volume control to enable sound.")
            st.selectbox("Featured movie", movie_ids, key=state_key,
                         format_func=lambda mid: str(movies_by_id.get(int(mid), {}).get("title", mid)),
                         on_change=_hero_interacted, args=(state_key,))
            nav = st.columns([1, 1, 2])
            with nav[0]:
                st.button("‹", key=f"hero-prev-{user_id}", help="Previous featured movie", on_click=_step_hero, args=(state_key, movie_ids, -1), use_container_width=True)
            with nav[1]:
                st.button("›", key=f"hero-next-{user_id}", help="Next featured movie", on_click=_step_hero, args=(state_key, movie_ids, 1), use_container_width=True)
            with nav[2]:
                st.checkbox("Auto rotate", key=auto_key, on_change=_hero_interacted, args=(state_key,))
            actions = st.columns([1.2, 1.2, 1.6])
            status = str(asset.get("trailer_status", "")).lower()
            trailer_url = asset.get("trailer_url")
            if status == "verified" and trailer_url:
                label, target = "Watch Trailer", trailer_url
            elif status in {"candidate_unverified", "unverified"} and trailer_url:
                label, target = "Open Trailer Link", trailer_url
            elif status == "link_only" and trailer_url:
                label, target = "Watch on YouTube", trailer_url
            else:
                label = "Search YouTube"
                target = "https://www.youtube.com/results?" + urlencode({"search_query": f"{featured.get('title', '')} {year_text} official trailer".strip()})
            with actions[0]:
                st.link_button(label, target, use_container_width=True)
            with actions[1]:
                if st.button("More Info", key=f"hero-details-{user_id}-{selected_id}", use_container_width=True):
                    st.session_state.detail_movie_id = selected_id
                    st.rerun()
            with actions[2]:
                if st.button("Add to Watchlist", key=f"hero-save-{user_id}-{selected_id}", use_container_width=True):
                    db.add_to_watchlist(user_id, selected_id)
                    st.toast("Added to your watchlist")
            st.caption(f"Featured {movie_ids.index(selected_id) + 1} of {len(movie_ids)}")


def render_topbar(user: dict):
    name = html.escape(str(user.get("name", "Movie fan")))
    st.markdown(f'<div class="cs-topbar"><div><span class="cs-brand">CINE<span>SPHERE</span></span><span class="cs-tagline">Explore a universe of movies</span></div><div class="cs-profile">Your space · {name}</div></div>', unsafe_allow_html=True)


def movie_rating_caption(item: dict, score_field: str | None) -> str | None:
    rating_count = pd.to_numeric(item.get("rating_count"), errors="coerce")
    if pd.notna(rating_count) and rating_count == 0:
        return "No ratings yet"
    if score_field == "predicted_rating" and pd.notna(item.get(score_field)):
        return f"ALS estimate · {float(item[score_field]):.2f} / 5"
    if score_field and pd.notna(item.get(score_field)) and pd.notna(item.get("avg_rating")):
        count_note = f" · {int(rating_count):,} {'rating' if rating_count == 1 else 'ratings'}" if pd.notna(rating_count) else ""
        return f"Average rating · {float(item['avg_rating']):.2f} / 5{count_note}"
    return None




def render_movie_card(item: dict, *, user_id: int, movies_by_id: dict, media: dict,
                      context: str, score_field: str | None = None, show_actions: bool = True):
    movie_id = _safe_int(item.get("movieId"))
    if movie_id is None:
        return
    item = {**movies_by_id.get(movie_id, {}), **item}
    asset = media.get(movie_id, {})
    with st.container(key=f"cs-card-{context}-{user_id}-{movie_id}"):
        poster_url = asset.get("poster_url") if has_poster(asset) else None
        title = display_title(item.get("title", "Untitled movie"))
        render_movie_preview(movie_id=movie_id, title=title,
                             poster_url=poster_url, video_id=verified_youtube_id(asset),
                             key_suffix=context)
        poster_credit(asset)
        st.markdown(f'<div class="movie-title">{html.escape(title)}</div>', unsafe_allow_html=True)
        genres = display_genres(item.get("genres", ""))
        year = item.get("year", "")
        year_text = str(int(year)) if pd.notna(year) and str(year).replace(".", "", 1).isdigit() else ""
        metadata = " · ".join(value for value in (year_text, genres) if value)
        if metadata:
            st.markdown(f'<div class="movie-meta">{html.escape(metadata)}</div>', unsafe_allow_html=True)
        rating = movie_rating_caption(item, score_field)
        if rating:
            st.caption(rating)
        with st.container(key=f"cs-card-actions-{context}-{user_id}-{movie_id}"):
            if st.button("More Info", key=f"details-{context}-{user_id}-{movie_id}", use_container_width=True):
                st.session_state.detail_movie_id = movie_id
                st.rerun()
            if show_actions:
                if movie_id in db.watched_ids(user_id):
                    st.caption("Already watched")
                else:
                    left, right = st.columns(2)
                    with left:
                        if st.button("＋", key=f"save-{context}-{user_id}-{movie_id}", help="Add to Watchlist", use_container_width=True):
                            db.add_to_watchlist(user_id, movie_id)
                            st.toast("Added to your watchlist")
                            st.rerun()
                    with right:
                        if st.button("✓", key=f"seen-{context}-{user_id}-{movie_id}", help="Mark as watched", use_container_width=True):
                            db.mark_watched(user_id, movie_id)
                            st.toast("Added to watched movies")
                            st.rerun()


def render_movie_grid(frame: pd.DataFrame, *, user_id: int, movies_by_id: dict, media: dict,
                      context: str, score_field: str | None = None, show_actions: bool = True,
                      columns_per_row: int = 4):
    if frame.empty:
        st.info("No movies to show here yet. Try another genre or search.")
        return
    fetch_page_media(frame.to_dict("records"), media)
    for start in range(0, len(frame), columns_per_row):
        columns = st.columns(columns_per_row, gap="medium")
        for column, (_, row) in zip(columns, frame.iloc[start:start + columns_per_row].iterrows()):
            with column:
                render_movie_card(row.to_dict(), user_id=user_id, movies_by_id=movies_by_id,
                                  media=media, context=context, score_field=score_field,
                                  show_actions=show_actions)


def render_movie_carousel(frame: pd.DataFrame, *, user_id: int, movies_by_id: dict, media: dict,
                          context: str, score_field: str | None = None, show_actions: bool = True,
                          page_size: int = 6):
    if frame.empty:
        st.info("No movies to show here yet. Try another genre or search.")
        return
    pages = max(1, math.ceil(len(frame) / page_size))
    state_key = f"carousel_page_{context}_{user_id}"
    page = min(int(st.session_state.get(state_key, 0)), pages - 1)
    left, center, right = st.columns([1, 8, 1])
    with left:
        if st.button("‹", key=f"carousel-prev-{context}-{user_id}", disabled=page == 0, help="Previous movies"):
            st.session_state[state_key] = page - 1
            st.rerun()
    with center:
        st.caption(f"Collection · {page + 1} / {pages}")
    with right:
        if st.button("›", key=f"carousel-next-{context}-{user_id}", disabled=page >= pages - 1, help="Next movies"):
            st.session_state[state_key] = page + 1
            st.rerun()
    visible = frame.iloc[page * page_size:(page + 1) * page_size]
    fetch_page_media(visible.to_dict("records"), media)
    with st.container(key=f"cs-carousel-{context}-{user_id}"):
        columns = st.columns(len(visible), gap="small")
        for column, (_, row) in zip(columns, visible.iterrows()):
            with column:
                render_movie_card(row.to_dict(), user_id=user_id, movies_by_id=movies_by_id,
                                  media=media, context=context, score_field=score_field,
                                  show_actions=show_actions)


@st.cache_data(show_spinner=False)
def cached_catalog():
    return load_movies()


@st.cache_data(show_spinner=False)
def cached_recommendations():
    return load_als(), load_popular(), load_genre_movies(), load_metrics()


@st.cache_data(show_spinner=False)
def cached_media(modified_at: int):
    return load_media()


def register_panel():
    with st.sidebar.expander("Add local demo account"):
        with st.form("register-form", clear_on_submit=True):
            name = st.text_input("Your name")
            submitted = st.form_submit_button("Create account", use_container_width=True)
            if submitted:
                try:
                    new_id = db.register_user(name)
                except Exception as exc:
                    st.error(f"Could not create account: {exc}")
                else:
                    st.session_state.active_user_id = new_id
                    st.rerun()








def render_home(user: dict, movies_by_id: dict, movies: pd.DataFrame, als: pd.DataFrame,
                popular: pd.DataFrame, genre_rows: pd.DataFrame, media: dict, metrics: dict):
    app_id = int(user["application_user_id"])
    hidden = db.watched_ids(app_id)
    genres = db.user_genres(app_id)
    render_home_hero(user, popular, movies_by_id, media)

    recent_ids = [int(mid) for mid in st.session_state.get("recently_viewed_movie_ids", []) if int(mid) in movies_by_id]
    recent_frame = pd.DataFrame([movies_by_id[mid] for mid in recent_ids])
    if not recent_frame.empty:
        st.markdown('<div class="section-heading"><h2>Continue Exploring</h2><span>Recently opened movies in this session</span></div>', unsafe_allow_html=True)
        render_movie_carousel(recent_frame, user_id=app_id, movies_by_id=movies_by_id, media=media,
                              context="home-recent", show_actions=False)

    if not popular.empty:
        st.markdown('<div class="section-heading"><h2>Popular Movies</h2><span>Member 2 community ranking</span></div>', unsafe_allow_html=True)
        render_movie_carousel(popular.head(30), user_id=app_id, movies_by_id=movies_by_id, media=media,
                              context="home-popular", score_field="weighted_rating")

    profile_id = user.get("movielens_user_id")
    if profile_id is not None:
        als_rows = personalized(als, int(profile_id), hidden)
        if not als_rows.empty:
            st.markdown(f'<div class="section-heading"><h2>Recommended For You</h2><span>ALS pilot · MovieLens user {int(profile_id)}</span></div>', unsafe_allow_html=True)
            st.caption("Predicted scores are model estimates, not historical ratings.")
            render_movie_carousel(als_rows, user_id=app_id, movies_by_id=movies_by_id, media=media,
                                  context="home-als", score_field="predicted_rating")
    else:
        st.markdown('<div class="source-note">No MovieLens pilot profile is linked to this application account. Genre discovery remains separate from personalized ALS recommendations.</div>', unsafe_allow_html=True)

    if "avg_rating" in movies.columns and "rating_count" in movies.columns:
        top_rated = movies.loc[(movies["rating_count"] >= 50) & movies["avg_rating"].notna()].sort_values(
            ["avg_rating", "rating_count", "title"], ascending=[False, False, True], kind="stable").head(30)
        if not top_rated.empty:
            st.markdown('<div class="section-heading"><h2>Top Rated Movies</h2><span>MovieLens historical average · 50+ ratings</span></div>', unsafe_allow_html=True)
            render_movie_carousel(top_rated, user_id=app_id, movies_by_id=movies_by_id, media=media,
                                  context="home-top-rated", score_field="weighted_rating", show_actions=False)

    all_genres = sorted({value for item in movies["genres"].dropna() for value in str(item).split("|") if value and value != "(no genres listed)" and value.casefold() != "unknown"})
    selected_genre = st.session_state.get("home_browse_genre", "")
    st.markdown('<div class="section-heading"><h2>Browse By Genre</h2><span>Pick a genre to explore its community-ranked movies</span></div>', unsafe_allow_html=True)
    with st.container(key="home-genre-chips"):
        chip_values = ["All Genres", *all_genres]
        for offset in range(0, len(chip_values), 7):
            group = chip_values[offset:offset + 7]
            columns = st.columns(len(group), gap="small")
            for column, label in zip(columns, group):
                with column:
                    active = (label == "All Genres" and not selected_genre) or label == selected_genre
                    if st.button(label, key=f"home-genre-{label}", type="primary" if active else "secondary", use_container_width=True):
                        st.session_state.home_browse_genre = "" if label == "All Genres" else label
                        st.rerun()
    if selected_genre:
        browse = genre_discovery(genre_rows, [selected_genre], hidden)
        st.caption(f"Genre discovery for {selected_genre}; not ALS-personalized.")
        render_movie_carousel(browse, user_id=app_id, movies_by_id=movies_by_id, media=media,
                              context=f"home-genre-{selected_genre}", score_field="weighted_rating")
    elif genres:
        discovery = genre_discovery(genre_rows, genres, hidden)
        st.caption("Picked from your saved favourites; genre discovery is not ALS-personalized.")
        render_movie_carousel(discovery, user_id=app_id, movies_by_id=movies_by_id, media=media,
                              context="home-favourites", score_field="weighted_rating")

    watchlist_frame = records_with_movies(db.movie_activity(app_id, "watchlist"), movies_by_id)
    if not watchlist_frame.empty:
        watchlist_frame = watchlist_frame.sort_values("added_at", ascending=False, kind="stable").head(30)
        st.markdown('<div class="section-heading"><h2>Recently Added to Watchlist</h2><span>Your saved movies</span></div>', unsafe_allow_html=True)
        render_movie_carousel(watchlist_frame, user_id=app_id, movies_by_id=movies_by_id, media=media,
                              context="home-watchlist", show_actions=False)

    if "avg_rating" in movies.columns and "rating_count" in movies.columns:
        hidden_gems = movies.loc[(movies["rating_count"] >= 50) & (movies["rating_count"] <= 1000) & movies["avg_rating"].notna()]
        hidden_gems = hidden_gems.sort_values(["avg_rating", "rating_count"], ascending=[False, False], kind="stable").head(30)
        if not hidden_gems.empty:
            st.markdown('<div class="section-heading"><h2>Hidden Gems</h2><span>Well rated by a smaller MovieLens audience</span></div>', unsafe_allow_html=True)
            render_movie_carousel(hidden_gems, user_id=app_id, movies_by_id=movies_by_id, media=media,
                                  context="home-hidden-gems", score_field="weighted_rating", show_actions=False)

    with st.expander("Edit favourite genres"):
        with st.form("favorite-genres"):
            selected = st.multiselect("Choose your genres", all_genres, default=genres)
            if st.form_submit_button("Save genres"):
                db.save_genres(app_id, selected)
                st.rerun()
    if metrics:
        with st.expander("About the ALS pilot evaluation"):
            st.write("Validation metrics supplied by Member 2. They describe the pilot model and do not guarantee a recommendation’s actual rating.")
            col1, col2 = st.columns(2)
            col1.metric("RMSE", f"{float(metrics['rmse']):.4f}")
            col2.metric("MAE", f"{float(metrics['mae']):.4f}")


def filter_explore_movies(movies: pd.DataFrame, *, search: str = "", genres=(),
                          include_rare: bool = False, only_artwork: bool = False,
                          media: dict | None = None) -> pd.DataFrame:
    filtered = movies.copy()
    counts = pd.to_numeric(filtered.get("rating_count", pd.Series(0, index=filtered.index)), errors="coerce").fillna(0)
    if not include_rare:
        filtered = filtered.loc[counts >= 100].copy()
    if genres:
        selected = set(genres)
        filtered = filtered.loc[filtered["genres"].fillna("").str.split("|").map(lambda values: bool(selected.intersection(values)))].copy()
    if search.strip():
        needle = _normalized_search_title(search.strip())
        titles = filtered["title"].fillna("").astype(str).map(_normalized_search_title)
        filtered = filtered.loc[titles.str.contains(needle, regex=False)].copy()
        titles = filtered["title"].fillna("").astype(str).map(_normalized_search_title)
        filtered["_match_quality"] = titles.map(lambda title: 0 if title == needle else 1 if title.startswith(needle) else 2)
    if only_artwork:
        media = media or {}
        filtered = filtered.loc[filtered["movieId"].map(lambda mid: has_poster(media.get(int(mid), {})))].copy()
    filtered["_rating_count_sort"] = pd.to_numeric(filtered.get("rating_count", 0), errors="coerce").fillna(0)
    if search.strip():
        filtered = filtered.sort_values(["_match_quality", "_rating_count_sort", "title"], ascending=[True, False, True], kind="stable")
    else:
        filtered = filtered.sort_values(["weighted_rating", "_rating_count_sort", "title"], ascending=[False, False, True], na_position="last", kind="stable") if "weighted_rating" in filtered else filtered.sort_values(["_rating_count_sort", "title"], ascending=[False, True], kind="stable")
    return filtered.drop(columns=["_match_quality", "_rating_count_sort"], errors="ignore").reset_index(drop=True)


@st.fragment
def render_explore(user: dict, movies: pd.DataFrame, movies_by_id: dict, media: dict):
    st.session_state.omdb_prefetch_remaining = 48
    st.session_state.omdb_prefetch_deadline = time.monotonic() + 4.5
    title_banner("Explore", "Search the MovieLens catalogue, browse genres, and filter by artwork.")
    app_id = int(user["application_user_id"])
    if st.session_state.pop("explore_clear_requested", False):
        st.session_state.explore_search = ""
    search_col, clear_col = st.columns([5, 1])
    with search_col:
        search = st.text_input("Search movies", placeholder="Search titles...", key="explore_search")
    with clear_col:
        st.markdown("<div style='height:1.85rem'></div>", unsafe_allow_html=True)
        if st.button("Clear", key="explore-clear", use_container_width=True):
            st.session_state.explore_clear_requested = True
            st.session_state.explore_selected_genres = []
            st.session_state.explore_page_number = 1
            st.session_state.explore_include_rare = False
            st.session_state.explore_only_artwork = False
            st.rerun()
    toggle_left, toggle_right = st.columns(2)
    with toggle_left:
        include_rare = st.checkbox("Include rarely rated movies", key="explore_include_rare", help="By default, Explore shows movies with at least 100 ratings.")
    with toggle_right:
        only_artwork = st.checkbox("Only movies with artwork", key="explore_only_artwork", help="Show only movies with a verified poster URL.")
    genre_values = sorted({g for value in movies["genres"].dropna() for g in str(value).split("|") if g and g != "(no genres listed)" and g.casefold() != "unknown"})
    selected = set(st.session_state.get("explore_selected_genres", []))
    st.markdown('<div class="eyebrow">BROWSE BY GENRE</div>', unsafe_allow_html=True)
    with st.container(key="genre-chips"):
        chip_labels = ["All Movies", *genre_values]
        for offset in range(0, len(chip_labels), 6):
            row = st.columns(min(6, len(chip_labels) - offset), gap="small")
            for col, label in zip(row, chip_labels[offset:offset + 6]):
                with col:
                    active = not selected if label == "All Movies" else label in selected
                    if st.button(label, key=f"genre-chip-{label}", type="primary" if active else "secondary", use_container_width=True):
                        if label == "All Movies":
                            selected.clear()
                        elif label in selected:
                            selected.remove(label)
                        else:
                            selected.add(label)
                        st.session_state.explore_selected_genres = sorted(selected)
                        st.session_state.explore_page_number = 1
                        st.rerun()
    sort_by = "Match quality" if search.strip() else st.selectbox(
        "Sort by", ["Weighted rating", "Rating activity", "Historical rating", "Release year", "Title"], key="explore_sort")
    filtered = filter_explore_movies(movies, search=search, genres=selected,
                                     include_rare=include_rare, only_artwork=only_artwork, media=media)
    if not search.strip():
        if sort_by == "Weighted rating" and "weighted_rating" in filtered:
            filtered = filtered.sort_values(["weighted_rating", "rating_count", "title"], ascending=[False, False, True], na_position="last", kind="stable")
        elif sort_by == "Historical rating" and "avg_rating" in filtered:
            filtered = filtered.sort_values(["avg_rating", "rating_count", "title"], ascending=[False, False, True], na_position="last", kind="stable")
        elif sort_by == "Release year" and "year" in filtered:
            filtered = filtered.sort_values(["year", "rating_count", "title"], ascending=[False, False, True], na_position="last", kind="stable")
        elif sort_by == "Title":
            filtered = filtered.sort_values("title", kind="stable")
    total = len(filtered)
    filter_signature = (search.strip().casefold(), tuple(sorted(selected)), include_rare, only_artwork, sort_by)
    if st.session_state.get("explore_filter_signature") != filter_signature:
        st.session_state.explore_filter_signature = filter_signature
        st.session_state.explore_page_number = 1
    st.markdown(f'<div class="section-heading"><h2>{total:,} movies</h2><span>{" · ".join(sorted(selected)) if selected else "At least 100 ratings" if not include_rare else "Complete catalogue"}</span></div>', unsafe_allow_html=True)
    if total == 0:
        st.markdown('<div class="empty-state"><h3>No movies found</h3><p>Try another title or change your filters.</p></div>', unsafe_allow_html=True)
        return
    if not search.strip() and not selected and not only_artwork:
        st.markdown('<div class="section-heading"><h2>Popular by rating activity</h2><span>Sorted by rating count</span></div>', unsafe_allow_html=True)
        render_movie_carousel(filtered.head(30), user_id=app_id, movies_by_id=movies_by_id,
                              media=media, context="explore-popular", show_actions=False)
    page_size = 32
    page_count = max(1, math.ceil(total / page_size))
    if int(st.session_state.get("explore_page_number", 1)) > page_count:
        st.session_state.explore_page_number = page_count
    page_number = st.selectbox("Catalogue page", range(1, page_count + 1),
                               format_func=lambda value: f"Page {value} of {page_count}", key="explore_page_number")
    start = (page_number - 1) * page_size
    visible = filtered.iloc[start:start + page_size]
    st.caption(f"Showing {start + 1:,}–{start + len(visible):,} of {total:,}")
    render_movie_grid(visible, user_id=app_id, movies_by_id=movies_by_id, media=media,
                      context="explore", score_field="weighted_rating")


@st.dialog("Movie details", width="large")
def render_movie_details(movie_id: int, user: dict, movies_by_id: dict, media: dict, als: pd.DataFrame):
    item = movies_by_id.get(int(movie_id))
    if not item:
        st.warning("This movie is no longer present in the loaded MovieLens catalogue.")
        return
    asset = media.get(int(movie_id), {})
    recent = [int(mid) for mid in st.session_state.get("recently_viewed_movie_ids", []) if int(mid) != int(movie_id)]
    st.session_state.recently_viewed_movie_ids = [int(movie_id), *recent][:20]
    if not asset:
        fetch_page_media([item], media)
        asset = media.get(int(movie_id), {})
    st.divider()
    st.subheader("Movie details")
    image_col, info_col = st.columns([1, 3], gap="large")
    with image_col:
        if has_poster(asset):
            st.image(asset["poster_url"], use_container_width=True)
            poster_credit(asset)
        else:
            st.markdown('<div class="poster-fallback">Artwork unavailable</div>', unsafe_allow_html=True)
            poster_status_note(asset, f"details-{user['application_user_id']}-{movie_id}")
    with info_col:
        st.markdown(f"### {html.escape(display_title(item.get('title', 'Untitled movie')))}")
        detail_genres = display_genres(item.get("genres", ""))
        if detail_genres:
            st.write(detail_genres)
        rating_count = pd.to_numeric(item.get("rating_count"), errors="coerce")
        if pd.notna(rating_count) and rating_count == 0:
            st.caption("No MovieLens ratings yet")
        else:
            metric_cols = st.columns(2)
            if pd.notna(item.get("avg_rating")):
                metric_cols[0].metric("Historical MovieLens average", f"{float(item['avg_rating']):.2f} / 5")
            if pd.notna(rating_count):
                metric_cols[1].metric("MovieLens rating count", f"{int(rating_count):,}")
        profile_id = user.get("movielens_user_id")
        if profile_id is not None and (pd.isna(rating_count) or rating_count > 0):
            model_row = als.loc[als["userId"].eq(int(profile_id)) & als["movieId"].eq(int(movie_id))]
            if not model_row.empty and pd.notna(model_row.iloc[0].get("predicted_rating")):
                st.caption(f"ALS predicted score: {float(model_row.iloc[0]['predicted_rating']):.2f} / 5 · model estimate, not a historical rating")
        overview = asset.get("overview")
        if overview and pd.notna(overview):
            st.write(str(overview))
            st.caption("Synopsis supplied by the recorded media metadata source.")
        else:
            st.caption("No verified synopsis is available in the current local metadata.")
        trailer_action(item, asset, f"details-{user['application_user_id']}-{movie_id}")
        action_col1, action_col2, action_col3 = st.columns(3)
        with action_col1:
            if st.button("＋ Add to Watchlist", key=f"details-watchlist-{user['application_user_id']}-{movie_id}"):
                db.add_to_watchlist(int(user["application_user_id"]), int(movie_id))
                st.toast("Added to your watchlist")
                st.rerun()
        with action_col2:
            if st.button("✓ Mark as Watched", key=f"details-watched-{user['application_user_id']}-{movie_id}"):
                db.mark_watched(int(user["application_user_id"]), int(movie_id))
                st.toast("Added to watched movies")
                st.rerun()
        with action_col3:
            if st.button("Close Details", key=f"details-close-{user['application_user_id']}-{movie_id}"):
                st.session_state.pop("detail_movie_id", None)
                st.rerun()
        if st.button("Set as Home feature", key=f"feature-movie-{user['application_user_id']}-{movie_id}"):
            st.session_state[f"hero_requested_feature_{int(user['application_user_id'])}"] = int(movie_id)
            st.rerun(scope="app")


def render_recommendations(user: dict, als: pd.DataFrame, popular: pd.DataFrame, genre_rows: pd.DataFrame, movies_by_id: dict, media: dict):
    title_banner("Recommendations", "Three separate discovery sources, clearly identified.")
    app_id = int(user["application_user_id"])
    hidden = db.watched_ids(app_id)
    profile_id = user.get("movielens_user_id")
    if profile_id is not None:
        recommendations = personalized(als, int(profile_id), hidden)
        st.markdown(f'<div class="source-note">Personalized ALS pilot recommendations · MovieLens user {int(profile_id)} · ordered by Member 2’s rank. Predicted scores are model estimates, not historical ratings.</div>', unsafe_allow_html=True)
        render_movie_grid(recommendations, user_id=app_id, movies_by_id=movies_by_id, media=media, context="als", score_field="predicted_rating")
        if recommendations.empty:
            st.info("There are no unwatched ALS results left for this pilot profile. Showing popular fallback results.")
            st.caption("Popular fallback · globally ranked by weighted rating; these are not personalized ALS recommendations.")
            render_movie_grid(popular_fallback(popular, hidden).head(12), user_id=app_id, movies_by_id=movies_by_id, media=media, context="popular-empty-als", score_field="weighted_rating")
    else:
        st.markdown('<div class="source-note">This new CineSphere account has no MovieLens pilot profile. Application IDs are not MovieLens user IDs, so recommendations use your selected genres or popular fallback.</div>', unsafe_allow_html=True)
        genres = db.user_genres(app_id)
        if genres:
            rows = genre_discovery(genre_rows, genres, hidden)
            st.subheader("Genre discovery")
            st.caption("Based on your saved favourite genres and Member 2’s weighted genre rankings; not ALS-personalized.")
            render_movie_grid(rows, user_id=app_id, movies_by_id=movies_by_id, media=media, context="genre-rec", score_field="weighted_rating")
        else:
            st.info("Add favourite genres on Home to see genre discovery. Showing popular fallback results for now.")
            render_movie_grid(popular_fallback(popular, hidden).head(12), user_id=app_id, movies_by_id=movies_by_id, media=media, context="popular-new-user", score_field="weighted_rating")
    with st.expander("Popular movies fallback"):
        st.caption("Popular fallback · globally ranked by weighted rating. These are not personalized ALS recommendations.")
        render_movie_grid(popular_fallback(popular, hidden).head(12), user_id=app_id, movies_by_id=movies_by_id, media=media, context="popular", score_field="weighted_rating")


def records_with_movies(records: list[dict], movies_by_id: dict) -> pd.DataFrame:
    rows = []
    for record in records:
        row = dict(movies_by_id.get(int(record["movieId"]), {"movieId": int(record["movieId"]), "title": "Movie unavailable in current catalogue", "genres": ""}))
        row.update(record)
        rows.append(row)
    return pd.DataFrame(rows)






def render_library_card(row: dict, *, user_id: int, media: dict, section: str):
    movie_id = int(row["movieId"])
    asset = media.get(movie_id, {})
    with st.container(key=f"cs-card-library-{section}-{user_id}-{movie_id}"):
        poster_url = asset.get("poster_url") if has_poster(asset) else None
        title = display_title(row.get("title", "Untitled movie"))
        render_movie_preview(movie_id=movie_id, title=title,
                             poster_url=poster_url, video_id=verified_youtube_id(asset),
                             key_suffix=f"library-{section}-{user_id}")
        poster_credit(asset)
        st.markdown(f'<div class="movie-title">{html.escape(title)}</div>', unsafe_allow_html=True)
        genres = display_genres(row.get("genres", ""))
        if genres:
            st.markdown(f'<div class="movie-meta">{html.escape(genres)}</div>', unsafe_allow_html=True)
        timestamp = row.get("added_at") if section == "watchlist" else row.get("watched_at")
        if timestamp:
            st.caption(("Saved" if section == "watchlist" else "Watched") + f" · {timestamp}")
        if st.button("More Info", key=f"library-details-{section}-{user_id}-{movie_id}", use_container_width=True):
            st.session_state.detail_movie_id = movie_id
            st.rerun()
        if section == "watchlist":
            action, remove = st.columns(2)
            with action:
                if st.button("Mark watched", key=f"watchlisted-seen-{user_id}-{movie_id}", use_container_width=True):
                    db.mark_watched(user_id, movie_id)
                    st.rerun()
            with remove:
                if st.button("Remove", key=f"watchlisted-remove-{user_id}-{movie_id}", use_container_width=True):
                    db.remove_from_watchlist(user_id, movie_id)
                    st.rerun()
        else:
            rating = row.get("rating")
            st.caption(f"Your rating · {float(rating):g} / 5" if pd.notna(rating) else "Not rated yet")
            with st.expander("Update rating or remove"):
                score = st.select_slider("Your rating", options=[0.5 * n for n in range(1, 11)],
                                         value=float(rating) if pd.notna(rating) else 4.0,
                                         key=f"rating-{user_id}-{movie_id}")
                save, remove = st.columns(2)
                with save:
                    if st.button("Save rating", key=f"save-rating-{user_id}-{movie_id}", use_container_width=True):
                        db.mark_watched(user_id, movie_id, score)
                        st.toast("Rating saved")
                        st.rerun()
                with remove:
                    if st.button("Remove", key=f"watched-remove-{user_id}-{movie_id}", use_container_width=True):
                        db.remove_watched(user_id, movie_id)
                        st.rerun()
        trailer_action(row, asset, f"library-{section}-{user_id}-{movie_id}")


def render_library_grid(frame: pd.DataFrame, *, user_id: int, media: dict, section: str):
    for start in range(0, len(frame), 3):
        columns = st.columns(3, gap="medium")
        for column, (_, row) in zip(columns, frame.iloc[start:start + 3].iterrows()):
            with column:
                render_library_card(row.to_dict(), user_id=user_id, media=media, section=section)


def render_watchlist(user: dict, movies_by_id: dict, media: dict):
    title_banner("Watchlist", "Your saved movies, ready when you are.")
    user_id = int(user["application_user_id"])
    frame = records_with_movies(db.movie_activity(user_id, "watchlist"), movies_by_id)
    if frame.empty:
        st.markdown('<div class="empty-state"><h3>Your watchlist is waiting</h3><p>Save a movie from Home or Explore and it will appear here.</p></div>', unsafe_allow_html=True)
        return
    st.caption(f"{len(frame):,} saved movie{'s' if len(frame) != 1 else ''}")
    render_library_grid(frame, user_id=user_id, media=media, section="watchlist")


def render_watched(user: dict, movies_by_id: dict, media: dict):
    title_banner("Watched Movies", "A personal log of the stories you have seen.")
    user_id = int(user["application_user_id"])
    frame = records_with_movies(db.movie_activity(user_id, "watched_movies"), movies_by_id)
    if frame.empty:
        st.markdown('<div class="empty-state"><h3>Your viewing story starts here</h3><p>Mark a movie as watched to build your history and add a rating.</p></div>', unsafe_allow_html=True)
        return
    st.caption(f"{len(frame):,} watched movie{'s' if len(frame) != 1 else ''}")
    render_library_grid(frame, user_id=user_id, media=media, section="watched")


def main():
    inject_styles()
    register_movie_preview_component()
    register_opening_animation_component()
    show_opening_animation(OPENING_VIDEO)
    st.session_state.omdb_prefetch_remaining = 48
    st.session_state.omdb_prefetch_deadline = time.monotonic() + 4.5
    db.initialize()
    try:
        movies = cached_catalog()
        als, popular, genre_rows, metrics = cached_recommendations()
    except (DataContractError, OSError, pd.errors.ParserError) as exc:
        st.error(f"CineSphere could not load its project data: {exc}")
        st.info("Check the cleaned MovieLens catalogue and Member 2 CSV files under output/clean/ and member2_recommendation/outputs/.")
        st.stop()
    movies_by_id = {int(row["movieId"]): row for row in movies.to_dict("records")}
    media_mtime = MEDIA_METADATA_FILE.stat().st_mtime_ns if MEDIA_METADATA_FILE.is_file() else 0
    media = cached_media(f"{media_mtime}:{db.movie_media_cache_version()}")
    all_genres = sorted({g for value in movies["genres"].dropna() for g in str(value).split("|") if g and g != "(no genres listed)" and g.casefold() != "unknown"})
    ids = pilot_user_ids(als)

    st.sidebar.markdown("# 🎬 CineSphere")
    st.sidebar.caption("Explore a universe of movies.")
    users = db.list_users()
    if not users:
        st.markdown('<div class="hero"><div class="eyebrow">WELCOME TO CINESPHERE</div><h1>Explore a universe<br>of movies.</h1><p>Create a local demo profile to save your genres, watchlist and viewing history. No password or email is collected.</p></div>', unsafe_allow_html=True)
        with st.form("first-registration"):
            name = st.text_input("Your name", max_chars=80)
            submitted = st.form_submit_button("Create my account")
            if submitted:
                try:
                    new_id = db.register_user(name)
                except Exception as exc:
                    st.error(f"Could not create account: {exc}")
                else:
                    st.session_state.active_user_id = new_id
                    st.rerun()
        st.stop()
    register_panel()
    users = db.list_users()
    user_options = [int(user["application_user_id"]) for user in users]
    if st.session_state.get("active_user_id") not in user_options:
        st.session_state.active_user_id = user_options[0]
    selected_id = st.sidebar.selectbox(
        "Application account",
        user_options,
        index=user_options.index(st.session_state.active_user_id),
        format_func=lambda value: next(f"{u['name']} · app ID {u['application_user_id']}" for u in users if int(u["application_user_id"]) == value),
        key="active_user_select",
    )
    st.session_state.active_user_id = selected_id
    user = next(item for item in users if int(item["application_user_id"]) == selected_id)
    if not user.get("onboarding_complete"):
        st.markdown('<div class="hero"><div class="eyebrow">A LITTLE ABOUT YOUR TASTE</div><h1>Choose your genres.</h1><p>Pick a few favourites and CineSphere will start with matching movies from the MovieLens catalogue.</p></div>', unsafe_allow_html=True)
        with st.form("genre-onboarding"):
            genres = st.multiselect("Favourite genres", all_genres)
            save = st.form_submit_button("Save genres and continue")
            skip = st.form_submit_button("Skip for now")
            if save or skip:
                db.save_genres(selected_id, genres if save else [])
                st.rerun()
        st.stop()
    st.sidebar.caption(f"Application ID: {selected_id}")
    st.sidebar.caption(f"MovieLens pilot profile: {user['movielens_user_id'] if user['movielens_user_id'] is not None else 'not linked'}")
    with st.sidebar.expander("Pilot demo profile settings"):
        with st.form("pilot-profile-form"):
            profile_options = [None, *ids]
            current_profile = int(user["movielens_user_id"]) if user["movielens_user_id"] is not None else None
            if current_profile not in profile_options:
                current_profile = None
            new_profile = st.selectbox(
                "MovieLens pilot profile",
                profile_options,
                index=profile_options.index(current_profile),
                format_func=lambda value: "No profile (use genre/popular discovery)" if value is None else f"Demo MovieLens user {value}",
                help="This is a demo linkage only. It does not convert the application user ID into a MovieLens user ID.",
            )
            if st.form_submit_button("Save profile link"):
                try:
                    db.set_pilot_profile(selected_id, new_profile)
                except Exception as exc:
                    st.error(f"Could not save this profile link: {exc}")
                else:
                    st.rerun()
    section = st.sidebar.radio("Navigate", ["Home", "Explore", "Recommendations", "Watchlist", "Watched Movies"], label_visibility="collapsed")
    st.sidebar.divider()
    st.sidebar.caption("Media appears only when its movieId match and status are recorded. Missing or permission-unclear artwork is not substituted.")
    with st.sidebar.expander("About & Credits"):
        st.caption("Movie titles, IDs and genres are from the cleaned MovieLens catalogue. Media provenance and licensing notes are stored in the movie media manifest.")
        uses_tmdb = any(
            "tmdb" in str(item.get("source") if pd.notna(item.get("source")) else item.get("poster_source", "")).lower()
            for item in media.values()
        )
        if uses_tmdb:
            st.markdown(
                '<a href="https://www.themoviedb.org" target="_blank" rel="noopener">'
                '<img src="https://www.themoviedb.org/assets/v4/logos/v2/blue_short-8e7b30f73a4020692ccca9c88bafe5dcb6f8a62a4c6bc55cd9ba82bb2cd95f6c.svg" '
                'alt="The Movie Database" style="width:145px;max-width:100%;height:auto"></a>',
                unsafe_allow_html=True,
            )
            st.caption("This product uses the TMDB API but is not endorsed or certified by TMDB.")
            st.markdown("[TMDB API and attribution](https://developer.themoviedb.org/docs/faq)")

    render_topbar(user)
    if section == "Home":
        render_home(user, movies_by_id, movies, als, popular, genre_rows, media, metrics)
    elif section == "Explore":
        render_explore(user, movies, movies_by_id, media)
    elif section == "Recommendations":
        render_recommendations(user, als, popular, genre_rows, movies_by_id, media)
    elif section == "Watchlist":
        render_watchlist(user, movies_by_id, media)
    else:
        render_watched(user, movies_by_id, media)

    detail_id = st.session_state.get("detail_movie_id")
    if detail_id is not None:
        render_movie_details(int(detail_id), user, movies_by_id, media, als)

    st.divider()
    st.caption("MovieLens movie records and ratings are provided by GroupLens. CineSphere uses evaluation metrics from the supplied ALS pilot output.")


if __name__ == "__main__":
    main()
