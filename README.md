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

## Project Modules

### Module 1: Data Processing and Analysis
- Dataset loading using PySpark.
- Data cleaning and preprocessing.
- Handling missing values and duplicates.
- Exploratory Data Analysis (EDA).
- Movie rating and genre analysis.
- Spark DataFrame operations and Spark SQL queries.
- Data visualization and cleaned dataset export.

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

## Project Status
Currently under development.# Movie Recommendation System Using PySpark
