import json
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from member3_dashboard.utils import acquire_tmdb_media, database
from scripts.fetch_media import (match_omdb_record, title_match, title_similarity,
                                 youtube_search_url, _canonical_media_status)
from member3_dashboard.utils.catalog import load_als, load_genre_movies, load_movies, load_popular
from member3_dashboard.utils.media import load_media
from member3_dashboard.utils.recommendations import genre_discovery, personalized, pilot_user_ids, popular_fallback
from member3_dashboard.app import (display_genres, display_title, filter_explore_movies,
                                   movie_rating_caption, verified_youtube_id)


class DashboardIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.test_db = Path(__file__).resolve().parents[2] / "database" / "cinesphere_test.sqlite3"
        for suffix in ("", "-journal"):
            Path(str(self.test_db) + suffix).unlink(missing_ok=True)

    def tearDown(self):
        for suffix in ("", "-journal"):
            Path(str(self.test_db) + suffix).unlink(missing_ok=True)

    def test_current_integration_files_match_expected_contracts(self):
        movies = load_movies()
        als = load_als()
        popular = load_popular()
        genre = load_genre_movies()
        self.assertGreater(len(movies), 80_000)
        self.assertEqual(len(als), 500)
        self.assertEqual(len(pilot_user_ids(als)), 50)
        self.assertEqual(len(popular), 50)
        self.assertEqual(len(genre), 380)
        self.assertTrue(als.groupby("userId").size().eq(10).all())

    def test_als_is_ranked_and_filtered_by_movielens_id(self):
        als = pd.DataFrame({"userId": [7, 7, 8], "movieId": [22, 11, 33], "rank": [2, 1, 1]})
        rows = personalized(als, 7, hidden_movie_ids={11})
        self.assertEqual(rows["movieId"].tolist(), [22])

    def test_genre_and_popular_results_remain_distinct_fallbacks(self):
        genres = pd.DataFrame({"genre": ["Drama", "Comedy"], "rank": [1, 1], "movieId": [1, 2], "weighted_rating": [4.2, 4.1]})
        popular = pd.DataFrame({"movieId": [2, 1], "weighted_rating": [4.5, 4.3]})
        self.assertEqual(genre_discovery(genres, ["Drama"])["movieId"].tolist(), [1])
        self.assertEqual(popular_fallback(popular)["movieId"].tolist(), [2, 1])

    def test_explore_uses_rating_threshold_match_quality_and_verified_artwork(self):
        movies = pd.DataFrame([
            {"movieId": 1, "title": "The Godfather (1972)", "genres": "Drama", "rating_count": 900, "avg_rating": 4.5, "weighted_rating": 4.4},
            {"movieId": 2, "title": "Godfather Part II (1974)", "genres": "Drama", "rating_count": 90, "avg_rating": 4.2, "weighted_rating": 4.1},
            {"movieId": 3, "title": "Tokyo Godfathers (2003)", "genres": "Animation", "rating_count": 800, "avg_rating": 4.0, "weighted_rating": 4.2},
            {"movieId": 4, "title": "Godfather Part IV (2020)", "genres": "Drama", "rating_count": 0, "avg_rating": 3.8, "weighted_rating": 4.8},
        ])
        default = filter_explore_movies(movies)
        self.assertEqual(default["movieId"].tolist(), [1, 3])
        search = filter_explore_movies(movies, search="godfather")
        self.assertEqual(search["movieId"].tolist(), [1, 3])
        include_rare = filter_explore_movies(movies, search="godfather", include_rare=True)
        self.assertIn(4, include_rare["movieId"].tolist())
        self.assertIn(2, include_rare["movieId"].tolist())
        self.assertEqual(include_rare["movieId"].tolist(), [1, 2, 4, 3])
        art = filter_explore_movies(movies, include_rare=True, only_artwork=True,
                                    media={2: {"poster_status": "verified", "poster_url": "https://example.invalid/poster.jpg"}})
        self.assertEqual(art["movieId"].tolist(), [2])

    def test_display_title_and_genre_cleanup(self):
        self.assertEqual(display_title("Godfather, The (1972)"), "The Godfather")
        self.assertEqual(display_title("Cosmos: A Spacetime Odyssey"), "Cosmos: A Spacetime Odyssey")
        self.assertEqual(display_genres("Unknown|Drama|(no genres listed)"), "Drama")

    def test_unrated_movie_never_displays_any_rating_score(self):
        item = {"rating_count": 0, "avg_rating": 4.9, "weighted_rating": 4.8, "predicted_rating": 4.6}
        self.assertEqual(movie_rating_caption(item, "weighted_rating"), "No ratings yet")
        self.assertEqual(movie_rating_caption(item, "predicted_rating"), "No ratings yet")

    def test_hero_and_hover_embeds_only_use_verified_youtube_urls(self):
        verified = {"trailer_status": "verified", "trailer_url": "https://www.youtube.com/watch?v=c8aFcHFu8QM"}
        candidate = {"trailer_status": "candidate_unverified", "trailer_url": "https://www.youtube.com/watch?v=c8aFcHFu8QM"}
        unsafe = {"trailer_status": "verified", "trailer_url": "https://example.com/watch?v=c8aFcHFu8QM"}
        self.assertEqual(verified_youtube_id(verified), "c8aFcHFu8QM")
        self.assertIsNone(verified_youtube_id(candidate))
        self.assertIsNone(verified_youtube_id(unsafe))

    def test_omdb_poster_credit_and_unverified_trailer_labels(self):
        media = load_media()
        self.assertEqual(media[318]["trailer_status"], "candidate_unverified")
        self.assertEqual(media[858]["trailer_status"], "candidate_unverified")
        self.assertIn(media[202439]["poster_status"], {"verified", "ok"})
        self.assertTrue(media[202439]["poster_source"].startswith("OMDb"))
        self.assertTrue(media[202439].get("poster_url", "").startswith("https://"))
        self.assertEqual(media[968]["poster_status"], "permission_required")

    def test_normalized_trailer_match_requires_a_trailer_title(self):
        self.assertTrue(title_match("Life Is Beautiful (La Vita è bella) (1997)", "Life is Beautiful (1998) Official Trailer - HD"))
        self.assertTrue(title_match("Boot, Das (Boat, The) (1981)", "Das Boot (director's Cut) - Trailer"))
        self.assertTrue(title_match("Lives of Others, The (Das Leben der Anderen)", "The Lives of Others Official Trailer HD"))
        self.assertFalse(title_match("Raiders of the Lost Ark (1981)", "Ti West on RAIDERS OF THE LOST ARK"))
        self.assertFalse(title_match("Up (2009)", "Upp trailer SE I HD"))
        self.assertIn("official+trailer", youtube_search_url("Up", 2009))

    def test_omdb_title_match_fuzzy_yearless_and_statuses(self):
        self.assertTrue(match_omdb_record("Godfather, The (1972)", 1972, "The Godfather", "1972")[0])
        self.assertTrue(match_omdb_record("The Godfather (1972)", 1972, "The Godfather", "1973")[0])
        cosmos = match_omdb_record("Cosmos: A Spacetime Odyssey", None,
                                   "Cosmos: A Spacetime Odissey", "2014")
        self.assertTrue(cosmos[0])
        self.assertTrue(cosmos[1])
        self.assertGreaterEqual(title_similarity("Cosmos: A Spacetime Odyssey", "Cosmos: A Spacetime Odissey"), 0.8)
        self.assertFalse(match_omdb_record("The Godfather (1972)", 1972, "The Godfather", "1980")[0])
        expected = {"ok", "no_imdb_id", "not_found", "poster_na", "mismatch_review", "error"}
        for status in expected:
            self.assertEqual(_canonical_media_status({"poster_status": status, "poster_url": "x" if status == "ok" else ""}), status)

    def test_shared_omdb_budget_and_one_shot_claim(self):
        db_path = self.test_db
        database.initialize(db_path)
        self.assertTrue(database.claim_movie_media_with_budget(7, daily_limit=1, db_path=db_path))
        self.assertEqual(database.omdb_usage_count(db_path=db_path), 1)
        self.assertFalse(database.claim_movie_media_with_budget(8, daily_limit=1, db_path=db_path))
        self.assertFalse(database.claim_movie_media_with_budget(7, daily_limit=10, db_path=db_path))

    def test_unmapped_media_batch_does_not_need_api_credentials(self):
        fixture_dir = Path(__file__).resolve().parents[2] / "database"
        movies_file = fixture_dir / "media_test_movies.csv"
        links_file = fixture_dir / "media_test_links.csv"
        manifest_file = fixture_dir / "media_test_manifest.csv"
        cache_file = fixture_dir / "media_test_cache.json"
        report_file = fixture_dir / "media_test_report.json"
        progress_file = fixture_dir / "media_test_progress.json"
        cleanup = [movies_file, links_file, manifest_file, cache_file, report_file, progress_file, progress_file.with_suffix(".json.tmp")]
        try:
            pd.DataFrame([{"movieId": 1, "title": "Real Test Movie (2000)", "year": 2000}]).to_csv(movies_file, index=False)
            pd.DataFrame([{"movieId": 1, "tmdbId": None}]).to_csv(links_file, index=False)
            with (
                patch.object(acquire_tmdb_media, "MOVIES_FILE", movies_file),
                patch.object(acquire_tmdb_media, "MEDIA_DIR", fixture_dir),
                patch.object(acquire_tmdb_media, "METADATA_FILE", manifest_file),
                patch.object(acquire_tmdb_media, "CACHE_FILE", cache_file),
                patch.object(acquire_tmdb_media, "REPORT_FILE", report_file),
                patch.object(acquire_tmdb_media, "PROGRESS_FILE", progress_file),
            ):
                records = acquire_tmdb_media.acquire(links_file, limit=1)
            self.assertEqual(records[0]["poster_status"], "no_tmdb_id")
            self.assertEqual(records[0]["trailer_status"], "no_tmdb_id")
            self.assertIn("poster_source", pd.read_csv(manifest_file).columns)
            self.assertEqual(json.loads(report_file.read_text(encoding="utf-8"))["catalog_movies"], 1)
        finally:
            for file in cleanup:
                file.unlink(missing_ok=True)

    def test_sqlite_app_ids_are_separate_and_watched_removes_watchlist_entry(self):
        db_path = self.test_db
        database.initialize(db_path)
        app_id = database.register_user("Hari", db_path=db_path)
        other_app_id = database.register_user("Keerthana", db_path=db_path)
        user = database.list_users(db_path)[0]
        self.assertEqual(user["application_user_id"], app_id)
        self.assertIsNone(user["movielens_user_id"])
        self.assertNotEqual(other_app_id, app_id)
        self.assertEqual(user["onboarding_complete"], 0)
        database.save_genres(app_id, ["Drama"], db_path)
        self.assertEqual(database.user_genres(app_id, db_path), ["Drama"])
        self.assertEqual(database.list_users(db_path)[0]["onboarding_complete"], 1)
        database.add_to_watchlist(app_id, 101, db_path)
        database.mark_watched(app_id, 101, 4.5, db_path)
        self.assertEqual(database.watchlist_ids(app_id, db_path), set())
        self.assertEqual(database.watched_ids(app_id, db_path), {101})
        self.assertEqual(database.movie_activity(app_id, "watched_movies", db_path)[0]["rating"], 4.5)


if __name__ == "__main__":
    unittest.main()
