# Movie Recommendation System Using PySpark

## Project Overview
This project develops a personalized movie recommendation system using the MovieLens 32M dataset. It uses Apache Spark for large-scale data processing and the ALS (Alternating Least Squares) collaborative filtering algorithm to generate personalized movie recommendations.

The application provides an interactive Streamlit dashboard where users can register, select their favourite genres, receive movie recommendations, and manage their watched movies and watchlist.

## Technologies Used
- Python
- Apache PySpark
- Spark SQL
- ALS Collaborative Filtering
- Streamlit
- SQLite
- Pandas
- Matplotlib

## Dataset
**MovieLens 32M Dataset**

The dataset contains movie ratings and movie information, which are used for data analysis and building the recommendation model.

Dataset: https://grouplens.org/datasets/movielens/32m/

## Project Modules

### Module 1: Data Processing and Analysis
- Load MovieLens 32M ratings and movies datasets using PySpark.
- Check null values and duplicate records.
- Clean and preprocess the data.
- Perform exploratory data analysis.
- Analyse ratings, genres, users and movie decades.
- Execute Spark DataFrame operations and Spark SQL queries.
- Generate 8 visualizations.
- Export cleaned datasets.

### Module 2: Recommendation Engine
- ALS collaborative filtering.
- Training and testing the recommendation model.
- Model evaluation using RMSE.
- Generating personalized Top-N movie recommendations.

### Module 3: User Interface and Application
- User registration and unique ID generation.
- Favourite genre selection.
- Personalized recommendation dashboard.
- Watched movies management.
- Watchlist management.
- SQLite database integration.

## Generated Outputs
- ratings_clean.parquet (shared separately due to its size)
- movies_clean.parquet
- movies_clean.csv
- genre_stats.csv
- 8 EDA plots

## Team Contributions
- **Anusree S:** Data processing, cleaning, exploratory data analysis, Spark DataFrame operations and Spark SQL.
- **Keerthana V K:** ALS recommendation model, training, evaluation and personalized recommendations.
- **Hari:** Streamlit application, user interface, user management and database integration.

## Project Workflow
1. Load and preprocess the MovieLens 32M dataset.
2. Perform exploratory data analysis.
3. Train the ALS recommendation model.
4. Generate personalized movie recommendations.
5. Integrate the recommendation engine with the Streamlit application.
6. Allow users to manage their movie preferences and watchlists.

## Repository Layout

- `data/raw/` and `data/processed/`: shared dataset locations.
- `member1_data_processing/`: Module 1 workspace.
- `member2_recommendation/`: Module 2 code and recommendation exports.
- `member3_dashboard/`: Module 3 Streamlit and SQLite integration; UI assets go in `frontend/`.
- `models/`, `outputs/`, and `database/`: shared model, generated output, and local database locations.
- `notebooks/`: exploratory notebooks.

Run the dashboard from the repository root with `streamlit run member3_dashboard/app.py`. Local data, model binaries, generated outputs, and SQLite files should not be committed unless the team agrees otherwise.

Member 3's dashboard is implemented in `member3_dashboard/app.py`. Its inputs, account ID rules, media workflow, and run instructions are documented in [member3_dashboard/README.md](member3_dashboard/README.md).

## Project Status
Currently under development.
