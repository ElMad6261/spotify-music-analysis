import os
import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine, text

load_dotenv()

engine = create_engine(
    f"postgresql+psycopg2://{os.getenv('DB_USER')}:{os.getenv('DB_PASSWORD')}"
    f"@{os.getenv('DB_HOST')}:{os.getenv('DB_PORT')}/{os.getenv('DB_NAME')}"
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