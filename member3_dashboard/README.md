# Member 3: CineSphere dashboard and SQLite

## Run locally

From the repository root:

```powershell
python -m pip install -r requirements.txt
streamlit run member3_dashboard/app.py
```

The app reads the cleaned MovieLens catalogue at `output/clean/movies_clean.csv` and the existing Phase 3 exports under `member2_recommendation/outputs/`. It does not start the Spark pipeline. SQLite creates `database/cinesphere.sqlite3` on first run; the database is excluded from Git.

## Account and recommendation IDs

Each new account receives an auto-incremented `application_user_id`. Its optional `movielens_user_id` is a separate field and can only be selected from IDs present in `pilot_user_recommendations.csv`. Linking one explicitly enables the supplied pilot ALS rows for demonstration. New accounts without a linked pilot profile receive genre discovery or global popular fallback results. Those sources are labelled separately from personalized ALS results.

The dashboard uses `movieId` to join catalog and recommendation rows. It filters ALS rows by the linked MovieLens `userId`, orders by `rank`, and removes watched movies from visible results. `predicted_rating` is shown as an ALS estimate; `avg_rating`, `rating_count`, and `weighted_rating` remain historical catalog statistics.

The Home hero rotates through the supplied popular-movie list and can be changed manually or from a movie's details dialog. Only records marked `trailer_status=verified` are embedded for the muted hero and client-side hover/tap preview; these currently represent a small subset of the catalogue. The YouTube player volume control lets viewers enable sound. Candidate/unverified trailers remain ordinary links. When no verified trailer is available, the poster remains visible and the detail view supplies the recorded trailer link or a YouTube search link. Browser playback depends on YouTube availability and the browser's autoplay policy.

## SQLite tables

- `users`: app account, generated ID, optional MovieLens pilot profile.
- `user_genres`: selected genres.
- `watchlist`: movies saved for later.
- `watched_movies`: watched history and optional 0.5–5.0 star user rating.

Marking a movie watched also removes it from that user's watchlist. Movie records and historical MovieLens ratings stay in the CSV/parquet data rather than being copied into SQLite.

## Posters and trailers

`scripts/fetch_media.py` uses one catalogue-wide priority queue: Member 2's popular, ALS, and genre outputs first, then the remaining catalogue ordered by rating count. It joins `movieId` to MovieLens `links.csv` IMDb IDs, hotlinks OMDb poster URLs, and never downloads poster files. The default daily cap is 950 and is shared with app lookups through SQLite. The manifest and `acquisition_progress.json` preserve statuses and the resume movie ID. OMDb title/year checks use normalized fuzzy matching; mismatches stay hidden for review. MovieLens YouTube mappings without an oEmbed check are link-only candidates and are not embedded. To add those local mapping links without making OMDb requests, run `python scripts/fetch_media.py --seed-youtube-links-only`. A saved network failure is retried with one probe before the next OMDb batch continues.

1. Put `OMDB_API_KEY=...` in the ignored root `.env` file and verify it with `python scripts/check_omdb_key.py`.
2. Put MovieLens links at `data/raw/ml-32m/links.csv` and the public mapping at `data/raw/ml-youtube.csv`.
3. Run or resume the priority queue from the repository root with `python scripts/fetch_media.py --daily-limit 950`. Each later daily run skips successful results and recent `not_found` rows.
5. Inspect `media/metadata/movie_media.csv`, `acquisition_progress.json`, and `media_acquisition_report.json`. Poster URLs are remote; no image files should be committed.

The older optional TMDB adapter remains available separately, but is not used by this workflow.

TMDB asks applications using its API to show an approved TMDB logo and this notice in an About/Credits area: “This product uses the TMDB API but is not endorsed or certified by TMDB.” CineSphere shows that attribution only when TMDB-sourced media is present. See [TMDB API FAQ and attribution requirements](https://developer.themoviedb.org/docs/faq). Poster and trailer availability is not guaranteed. Images remain on TMDB's image CDN; the team should follow its applicable terms rather than mirror assets without permission.

## Integration files

- `app.py`: Streamlit application and five sections.
- `utils/catalog.py`: validates and loads Member 1 and Member 2 CSV contracts.
- `utils/recommendations.py`: distinct ALS, genre discovery, and popular fallback selections.
- `utils/database.py`: SQLite user and activity operations.
- `utils/acquire_tmdb_media.py`: bounded, retryable TMDB reference acquisition and coverage records.
- `scripts/fetch_media.py`: prioritized OMDb poster URL acquisition, matching, coverage report, and resumable quota handling.
- `utils/media.py`: verified media metadata reader.
- `ui/movie_preview.py`: isolated client-side, muted preview component; pointer hover starts playback and pointer leave stops it without a Python rerun.
- `frontend/`: future reusable presentation assets and components.
