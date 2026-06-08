import pandas as pd
from sqlalchemy import create_engine, text

DB_USER     = "postgres"
DB_PASSWORD = "lol123"
DB_HOST     = "localhost"
DB_PORT     = "5432"
DB_NAME     = "spotify_analysis"

engine = create_engine(
    f"postgresql+psycopg2://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
)

df = pd.read_csv('data/processed/dataset_clustered.csv')

df.to_sql(
    name='spotify_tracks',
    con=engine,
    if_exists='replace',
    index=False,
    chunksize=1000
)

print(f"✅ {len(df):,} filas cargadas en PostgreSQL")

with engine.connect() as conn:
    result = conn.execute(text("SELECT COUNT(*) FROM spotify_tracks"))
    print(f"✅ Filas en DB: {result.fetchone()[0]:,}")