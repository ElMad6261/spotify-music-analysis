# 🎵 Spotify Music Analysis Platform

A full-stack data science project analyzing 114,000 Spotify tracks to uncover patterns in music popularity, audio features, and genre characteristics.

![Python](https://img.shields.io/badge/Python-3.11-blue?style=flat-square&logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-1.55-red?style=flat-square&logo=streamlit)
![PostgreSQL](https://img.shields.io/badge/PostgreSQL-16-blue?style=flat-square&logo=postgresql)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-ML-orange?style=flat-square&logo=scikit-learn)

---

## 📌 Overview

This project builds a complete data pipeline — from raw data ingestion and cleaning, through exploratory analysis and machine learning, to an interactive dashboard and structured SQL queries.

**Key questions answered:**
- What audio features make a song popular?
- Can we group songs by sound profile without using genre labels?
- How do genres differ in their audio characteristics?

---

## 🏗️ Project Structure

```
spotify-music-analysis/
├── data/
│   ├── raw/                    # Original Kaggle dataset
│   └── processed/              # Cleaned and clustered CSVs
├── notebooks/
│   ├── 01_data_cleaning.ipynb  # Data cleaning & feature engineering
│   ├── 02_eda.ipynb            # Exploratory data analysis
│   ├── 03_clustering.ipynb     # KMeans clustering & PCA
│   └── 04_sql_analysis.ipynb   # SQL queries via SQLAlchemy
├── dashboard/
│   └── app.py                  # Streamlit interactive dashboard
├── src/
│   ├── load_to_postgres.py     # ETL script to PostgreSQL
│   ├── data_processing.py
│   ├── clustering_model.py
│   └── visualization.py
├── sql/
│   └── database_schema.sql     # Table definitions
├── reports/                    # Exported charts and figures
├── .env.example                # Environment variables template
├── requirements.txt
└── README.md
```

---

## 🔧 Tech Stack

| Layer | Tools |
|---|---|
| Language | Python 3.11 |
| Data manipulation | Pandas, NumPy |
| Machine Learning | Scikit-learn (KMeans, PCA, NearestNeighbors) |
| Visualization | Matplotlib, Seaborn, Plotly |
| Dashboard | Streamlit |
| Database | PostgreSQL + SQLAlchemy + psycopg2 |
| Environment | Google Colab, VS Code |
| Version control | Git + GitHub |

---

## 📊 Dataset

**Source:** [Spotify Tracks Dataset](https://www.kaggle.com/datasets/maharshipandya/-spotify-tracks-dataset) — Kaggle

- **113,999 tracks** across **114 genres**
- Audio features: `danceability`, `energy`, `loudness`, `tempo`, `valence`, `acousticness`, `speechiness`, `instrumentalness`, `liveness`
- Target variable: `popularity` (0–100)

---

## 🚀 Pipeline

### Phase 1 — Data Cleaning
- Removed irrelevant columns (`Unnamed: 0`)
- Handled 1 row with nulls across 3 columns
- Converted `duration_ms` → `duration_min`
- Engineered features: `energy_dance_score`, `acoustic_vs_energy`

### Phase 2 — EDA
- Popularity distribution and genre comparison
- Correlation heatmap across all audio features
- Trending songs analysis (`popularity > 80`)
- Duration vs popularity, explicit vs non-explicit

### Phase 3 — Machine Learning
- **Algorithm:** KMeans Clustering
- **Features:** 8 audio features normalized with StandardScaler
- **Optimal k:** 6 (selected via Elbow Method + Silhouette Score = 0.210)
- **Visualization:** PCA 2D projection

**Clusters identified:**

| Cluster | Name | Key Features |
|---|---|---|
| 0 | Hip-Hop / Rap | High speechiness (0.482), high danceability |
| 1 | Hard Rock / Metal | Very high energy (0.815), fast tempo (139 BPM) |
| 2 | Acoustic / Folk | High acousticness (0.675), low energy |
| 3 | Electronic Instrumental | Very high instrumentalness (0.788) |
| 4 | Pop / Dance | High danceability (0.696), high valence (0.700) |
| 5 | Classical / Ambient | High acousticness (0.862), minimal energy |

### Phase 4 — Dashboard
Interactive Streamlit app with 4 pages:
- **Overview:** KPIs, popularity distribution, top genres
- **Feature Analysis:** scatter plots with OLS trendlines, correlation bars
- **Cluster Explorer:** donut chart, radar chart, song tables
- **Song Finder:** search by name or audio parameters using NearestNeighbors

### Phase 5 — SQL
113,999 rows loaded to PostgreSQL via SQLAlchemy. Queries covering genre rankings, cluster profiles, trending songs, and explicit content analysis.

---

## ⚙️ Setup

### 1. Clone the repo
```bash
git clone https://github.com/YOUR_USERNAME/spotify-music-analysis.git
cd spotify-music-analysis
```

### 2. Create virtual environment
```bash
python -m venv venv
venv\Scripts\activate  # Windows
source venv/bin/activate  # Mac/Linux
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Configure environment variables
```bash
cp .env.example .env
# Edit .env with your PostgreSQL credentials
```

### 5. Download dataset
```bash
kaggle datasets download -d maharshipandya/-spotify-tracks-dataset
unzip *.zip -d data/raw/
```

### 6. Run the dashboard
```bash
streamlit run dashboard/app.py
```

---

## 📈 Key Findings

- **Loudness** has the strongest positive correlation with popularity
- **Acousticness** correlates negatively with popularity
- Songs with `popularity > 80` average **0.71 danceability** vs 0.54 dataset-wide
- **Explicit songs** score higher on danceability and energy
- Pop/Dance cluster dominates trending songs (popularity > 80)

---

## 🧠 Skills Demonstrated

`Data Cleaning` · `EDA` · `Feature Engineering` · `KMeans Clustering` · `PCA` · `NearestNeighbors` · `SQL` · `ETL` · `Streamlit` · `Plotly` · `PostgreSQL` · `Python` · `Git`

---

## 👤 Author

**Miguel Arosmena**  
Software Engineering Student — Universidad Tecnológica de Panamá  
