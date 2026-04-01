import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

# ── Config ──────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Spotify Analysis",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── CSS ──────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:ital,opsz,wght@0,9..40,300;0,9..40,400;0,9..40,500;0,9..40,600;0,9..40,700;1,9..40,400&family=Space+Mono:wght@400;700&display=swap');

    /* ── Base ─────────────────────────────────────── */
    html, body, .stApp {
        background-color: #0E1117 !important;
        color: #DDE1EC !important;
        font-family: 'DM Sans', sans-serif !important;
    }
    .block-container {
        padding-top: 1.5rem !important;
        padding-bottom: 2rem !important;
    }

    /* ── Ocultar header y elementos de UI ─────────── */
    [data-testid="stHeader"],
    header[data-testid="stHeader"],
    [data-testid="stToolbar"],
    [data-testid="stDecoration"],
    [data-testid="stStatusWidget"],
    button[aria-label="Keyboard shortcuts"],
    #MainMenu,
    footer {
        display: none !important;
        visibility: hidden !important;
        height: 0 !important;
    }

    /* ── Sidebar ──────────────────────────────────── */
    [data-testid="stSidebar"] {
        display: block !important;
        visibility: visible !important;
        opacity: 1 !important;
        background-color: #13161F !important;
        border-right: 1px solid #21263A !important;
    }
    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] span,
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] div {
        color: #A9AEBE !important;
        font-family: 'DM Sans', sans-serif !important;
    }
    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3 {
        color: #DDE1EC !important;
    }

    /* ── Radio nav ────────────────────────────────── */
    [data-testid="stSidebar"] [data-testid="stRadio"] label {
        border-radius: 8px !important;
        padding: 7px 12px !important;
        transition: background 0.15s ease !important;
        cursor: pointer !important;
        width: 100% !important;
        display: block !important;
    }
    [data-testid="stSidebar"] [data-testid="stRadio"] label:hover {
        background: rgba(30,215,96,0.07) !important;
    }
    [data-testid="stSidebar"] [data-testid="stRadio"] label:has(input:checked) {
        background: rgba(30,215,96,0.13) !important;
    }
    [data-testid="stSidebar"] [data-testid="stRadio"] label:has(input:checked) p {
        color: #1ED760 !important;
        font-weight: 600 !important;
    }

    /* ── Métricas ─────────────────────────────────── */
    [data-testid="stMetric"] {
        background: linear-gradient(145deg, #161B27, #13161F) !important;
        border: 1px solid #21263A !important;
        border-radius: 14px !important;
        padding: 16px 20px !important;
    }
    [data-testid="stMetricValue"] {
        font-family: 'Space Mono', monospace !important;
        color: #1ED760 !important;
        font-size: 24px !important;
        font-weight: 700 !important;
    }
    [data-testid="stMetricLabel"] {
        color: #6B7280 !important;
        font-size: 11px !important;
        text-transform: uppercase !important;
        letter-spacing: 0.09em !important;
        font-weight: 500 !important;
    }
    [data-testid="stMetricDelta"] svg { display: none !important; }

    /* ── Headers ──────────────────────────────────── */
    h1 {
        font-family: 'DM Sans', sans-serif !important;
        font-size: 26px !important;
        font-weight: 700 !important;
        color: #DDE1EC !important;
        letter-spacing: -0.01em !important;
    }
    h2 {
        font-family: 'DM Sans', sans-serif !important;
        font-size: 18px !important;
        font-weight: 600 !important;
        color: #DDE1EC !important;
    }
    h3 {
        font-family: 'DM Sans', sans-serif !important;
        font-size: 15px !important;
        font-weight: 600 !important;
        color: #A9AEBE !important;
    }

    /* ── Section titles ───────────────────────────── */
    .section-title {
        font-family: 'DM Sans', sans-serif;
        font-size: 13px;
        font-weight: 600;
        color: #DDE1EC;
        display: flex;
        align-items: center;
        gap: 10px;
        margin: 24px 0 12px 0;
        padding-bottom: 10px;
        border-bottom: 1px solid #21263A;
        text-transform: uppercase;
        letter-spacing: 0.06em;
    }
    .section-title::before {
        content: '';
        display: inline-block;
        width: 3px;
        height: 14px;
        background: #1ED760;
        border-radius: 2px;
        flex-shrink: 0;
    }

    /* ── Tabs ─────────────────────────────────────── */
    [data-testid="stTabs"] [role="tablist"] {
        border-bottom: 1px solid #21263A !important;
        gap: 4px !important;
    }
    [data-testid="stTabs"] button {
        background: transparent !important;
        color: #6B7280 !important;
        border: none !important;
        border-radius: 0 !important;
        font-family: 'DM Sans', sans-serif !important;
        font-weight: 500 !important;
        font-size: 14px !important;
        padding: 8px 18px !important;
        transition: color 0.15s !important;
    }
    [data-testid="stTabs"] button:hover { color: #DDE1EC !important; }
    [data-testid="stTabs"] button[aria-selected="true"] {
        color: #1ED760 !important;
        border-bottom: 2px solid #1ED760 !important;
        font-weight: 600 !important;
    }

    /* ── Text input ───────────────────────────────── */
    .stTextInput input {
        background: #161B27 !important;
        border: 1px solid #21263A !important;
        border-radius: 10px !important;
        color: #DDE1EC !important;
        font-family: 'DM Sans', sans-serif !important;
        font-size: 14px !important;
        padding: 10px 14px !important;
        transition: border-color 0.2s !important;
    }
    .stTextInput input:focus {
        border-color: #1ED760 !important;
        box-shadow: 0 0 0 3px rgba(30,215,96,0.12) !important;
        outline: none !important;
    }
    .stTextInput input::placeholder { color: #4A5168 !important; }

    /* ── Selectbox ────────────────────────────────── */
    .stSelectbox > div > div {
        background: #161B27 !important;
        border: 1px solid #21263A !important;
        border-radius: 10px !important;
        color: #DDE1EC !important;
    }

    /* ── Botones ──────────────────────────────────── */
    .stButton > button {
        font-family: 'DM Sans', sans-serif !important;
        border-radius: 30px !important;
        font-weight: 600 !important;
        font-size: 14px !important;
        padding: 9px 24px !important;
        transition: all 0.2s ease !important;
        letter-spacing: 0.02em !important;
    }
    .stButton > button[kind="primary"] {
        background: #1ED760 !important;
        color: #0E1117 !important;
        border: none !important;
    }
    .stButton > button[kind="primary"]:hover {
        background: #23F56B !important;
        box-shadow: 0 4px 18px rgba(30,215,96,0.35) !important;
        transform: translateY(-1px) !important;
    }
    .stButton > button[kind="secondary"] {
        background: transparent !important;
        color: #DDE1EC !important;
        border: 1px solid #21263A !important;
    }
    .stButton > button[kind="secondary"]:hover {
        border-color: #1ED760 !important;
        color: #1ED760 !important;
    }

    /* ── Dataframe ────────────────────────────────── */
    [data-testid="stDataFrame"] {
        border: 1px solid #21263A !important;
        border-radius: 12px !important;
        overflow: hidden !important;
    }

    /* ── Alerts ───────────────────────────────────── */
    [data-testid="stAlert"] {
        background: #161B27 !important;
        border: 1px solid #21263A !important;
        border-radius: 10px !important;
        color: #DDE1EC !important;
    }

    /* ── Divider ──────────────────────────────────── */
    hr {
        border: none !important;
        border-top: 1px solid #21263A !important;
        margin: 22px 0 !important;
    }

    /* ── Caption ──────────────────────────────────── */
    .stCaption, [data-testid="stCaptionContainer"] {
        color: #4A5168 !important;
        font-size: 12px !important;
    }

    /* ── Markdown text ────────────────────────────── */
    .stMarkdown p {
        color: #A9AEBE !important;
        font-size: 14px !important;
        line-height: 1.6 !important;
    }

    /* ── Plotly: leyendas y ejes visibles ─────────── */
    .legendtext {
        fill: #DDE1EC !important;
        font-family: 'DM Sans', sans-serif !important;
        font-size: 12px !important;
    }
    .legend .bg {
        fill: #161B27 !important;
        stroke: #21263A !important;
    }
    g.legend text {
        fill: #DDE1EC !important;
    }
    .xtick text, .ytick text,
    .g-xtitle text, .g-ytitle text {
        fill: #DDE1EC !important;
    }
    .angularaxislayer text,
    .radialaxislayer text {
        fill: #DDE1EC !important;
    }
    .pieslice text {
        fill: #DDE1EC !important;
    }
     /* Para cambiar los labels dentro de SongFinder */
[data-testid="stSlider"] label {
    color: #FFFFFF !important;   /* cambia aquí el color */
    font-weight: 500 !important;
    font-size: 14px !important;
}       
            
            /* ── Selectbox labels ──────────────────────────────────── */
[data-testid="stSelectbox"] label {
    color: #FFFFFF !important;
    font-weight: 500 !important;
    font-size: 14px !important;
}
            
            /* ── Selectbox labels ──────────────────────────────────── */
[data-testid="stSelectbox"] label {
    color: #FFFFFF !important;
    font-weight: 500 !important;
    font-size: 14px !important;
}
            /* ── Label del input de búsqueda ───────────────────────────── */
[data-testid="stTextInput"] label {
    color: #FFFFFF !important;
    font-weight: 500 !important;
    font-size: 14px !important;
}
</style>
""", unsafe_allow_html=True)

# ── Chart theme ──────────────────────────────────────────────────────
CHART_BG   = '#161B27'
CHART_GRID = '#21263A'
CHART_FONT = '#DDE1EC'

LEGEND_CFG = dict(
    bgcolor='#161B27',
    bordercolor='#21263A',
    borderwidth=1,
    font=dict(color='#DDE1EC', size=12, family='DM Sans')
)

CHART_THEME = dict(
    paper_bgcolor=CHART_BG,
    plot_bgcolor=CHART_BG,
    font=dict(color=CHART_FONT, family='DM Sans', size=12),
    margin=dict(t=30, b=20, l=10, r=10),
    legend=LEGEND_CFG,
    xaxis=dict(gridcolor=CHART_GRID, zerolinecolor=CHART_GRID,
               color=CHART_FONT, tickfont=dict(color=CHART_FONT)),
    yaxis=dict(gridcolor=CHART_GRID, zerolinecolor=CHART_GRID,
               color=CHART_FONT, tickfont=dict(color=CHART_FONT)),
)

CLUSTER_COLORS = ['#1ED760', '#FF6B6B', '#4EC9F0', '#F7B731', '#A55EEA', '#FD9644']

# ── Datos ────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    return pd.read_csv('data/processed/dataset_clustered.csv')

@st.cache_resource
def build_model(df):
    features = ['danceability', 'energy', 'loudness', 'tempo',
                'valence', 'acousticness', 'speechiness', 'instrumentalness']
    scaler = StandardScaler()
    X = scaler.fit_transform(df[features])
    model = NearestNeighbors(n_neighbors=10, metric='cosine')
    model.fit(X)
    return model, scaler, features

df = load_data()
nn_model, scaler, feature_cols = build_model(df)

# ── Sidebar ──────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🎵 Spotify Analysis")
    st.markdown("---")

    page = st.radio("Navegación", [
        "📊 Overview",
        "🔍 Feature Analysis",
        "🤖 Clusters",
        "🎯 Song Finder"
    ], label_visibility="collapsed")

    st.markdown("---")
    st.markdown("**Filtros globales**")

    genres = ['Todos'] + sorted(df['track_genre'].unique().tolist())
    selected_genre = st.selectbox("Género", genres)
    pop_range = st.slider("Popularidad", 0, 100, (0, 100))

    st.markdown("---")
    st.caption(f"📁 {len(df):,} canciones · {df['track_genre'].nunique()} géneros")

# ── Filtrar ───────────────────────────────────────────────────────────
filtered = df.copy()
if selected_genre != 'Todos':
    filtered = filtered[filtered['track_genre'] == selected_genre]
filtered = filtered[
    (filtered['popularity'] >= pop_range[0]) &
    (filtered['popularity'] <= pop_range[1])
]

# ════════════════════════════════════════════════════════════════════
# PÁGINA 1 — OVERVIEW
# ════════════════════════════════════════════════════════════════════
if page == "📊 Overview":

    st.markdown("## 📊 Overview")
    st.markdown(
        f"<p style='color:#6B7280;font-size:13px;margin-top:-8px'>"
        f"Mostrando <span style='color:#1ED760;font-weight:600'>{len(filtered):,}</span> canciones"
        f"</p>",
        unsafe_allow_html=True
    )

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Canciones",        f"{len(filtered):,}")
    c2.metric("Popularidad avg",  f"{filtered['popularity'].mean():.1f}")
    c3.metric("Danceability avg", f"{filtered['danceability'].mean():.2f}")
    c4.metric("Energy avg",       f"{filtered['energy'].mean():.2f}")
    c5.metric("Géneros",          f"{filtered['track_genre'].nunique()}")

    st.markdown("---")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<div class="section-title">Distribución de Popularidad</div>',
                    unsafe_allow_html=True)
        fig = px.histogram(filtered, x='popularity', nbins=50,
                           color_discrete_sequence=['#1ED760'])
        fig.update_layout(**CHART_THEME, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown('<div class="section-title">Top 10 Géneros por Popularidad</div>',
                    unsafe_allow_html=True)
        top_genres_df = (filtered.groupby('track_genre')['popularity']
                         .mean().sort_values(ascending=True)
                         .tail(10).reset_index())
        fig = px.bar(top_genres_df, x='popularity', y='track_genre',
                     orientation='h',
                     color='popularity',
                     color_continuous_scale=[[0, '#21263A'], [1, '#1ED760']])
        fig.update_layout(**CHART_THEME, showlegend=False, coloraxis_showscale=False)
        st.plotly_chart(fig, use_container_width=True)

    st.markdown('<div class="section-title">🏆 Top 10 Canciones más Populares</div>',
                unsafe_allow_html=True)
    top10 = (filtered[['track_name', 'artists', 'track_genre', 'popularity',
                        'danceability', 'energy', 'cluster_name']]
             .sort_values('popularity', ascending=False)
             .head(10).reset_index(drop=True))
    top10.index += 1
    st.dataframe(top10, use_container_width=True)


# ════════════════════════════════════════════════════════════════════
# PÁGINA 2 — FEATURE ANALYSIS
# ════════════════════════════════════════════════════════════════════
elif page == "🔍 Feature Analysis":

    st.markdown("## 🔍 Feature Analysis")

    col1, col2 = st.columns([1, 3])
    with col1:
        feature = st.selectbox("Feature vs Popularidad", [
            'danceability', 'energy', 'valence',
            'acousticness', 'tempo', 'loudness', 'speechiness'
        ])
    with col2:
        sample_size = st.slider("Muestra de puntos", 1000, 10000, 5000, 500)

    sample = filtered.sample(min(sample_size, len(filtered)), random_state=42)

    fig = px.scatter(sample, x=feature, y='popularity',
                     color='cluster_name', opacity=0.55,
                     trendline='ols',
                     color_discrete_sequence=CLUSTER_COLORS,
                     hover_data=['track_name', 'artists'])
    fig.update_traces(marker=dict(size=4))
    fig.update_layout(**CHART_THEME)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    st.markdown('<div class="section-title">Correlaciones con Popularidad</div>',
                unsafe_allow_html=True)

    features_list = ['danceability', 'energy', 'loudness', 'acousticness',
                     'valence', 'tempo', 'speechiness', 'instrumentalness', 'liveness']
    corrs = (filtered[features_list + ['popularity']]
             .corr()['popularity']
             .drop('popularity')
             .sort_values())

    bar_colors = ['#FF6B6B' if v < 0 else '#1ED760' for v in corrs.values]
    fig2 = go.Figure(go.Bar(
        x=corrs.values,
        y=corrs.index,
        orientation='h',
        marker_color=bar_colors,
        marker_line_width=0,
    ))
    fig2.update_layout(**CHART_THEME)
    st.plotly_chart(fig2, use_container_width=True)


# ════════════════════════════════════════════════════════════════════
# PÁGINA 3 — CLUSTERS
# ════════════════════════════════════════════════════════════════════
elif page == "🤖 Clusters":

    st.markdown("## 🤖 Cluster Explorer")

    col1, col2 = st.columns(2)

    with col1:
        cluster_dist = filtered['cluster_name'].value_counts().reset_index()
        fig = px.pie(cluster_dist, values='count', names='cluster_name',
                     color_discrete_sequence=CLUSTER_COLORS,
                     hole=0.45)
        fig.update_traces(
            textfont=dict(color='#DDE1EC', size=13, family='DM Sans'),
        )
        fig.update_layout(
            paper_bgcolor=CHART_BG,
            font=dict(color='#DDE1EC', family='DM Sans', size=12),
            legend=LEGEND_CFG,
            title=dict(text="Distribución de Clusters",
                       font=dict(color='#DDE1EC', size=14, family='DM Sans')),
            margin=dict(t=50, b=10, l=10, r=10)
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        radar_features = ['danceability', 'energy', 'valence', 'acousticness', 'speechiness']
        cluster_avg = filtered.groupby('cluster_name')[radar_features].mean()
        fig = go.Figure()
        for i, cluster in enumerate(cluster_avg.index):
            fig.add_trace(go.Scatterpolar(
                r=cluster_avg.loc[cluster].values,
                theta=radar_features,
                fill='toself',
                name=cluster,
                line=dict(color=CLUSTER_COLORS[i % len(CLUSTER_COLORS)], width=2),
                opacity=0.75
            ))
        fig.update_layout(
            polar=dict(
                bgcolor='#13161F',
                radialaxis=dict(
                    visible=True, range=[0, 1],
                    gridcolor=CHART_GRID,
                    color='#DDE1EC',
                    tickfont=dict(size=9, color='#DDE1EC')
                ),
                angularaxis=dict(
                    gridcolor=CHART_GRID,
                    color='#DDE1EC',
                    tickfont=dict(size=11, color='#DDE1EC')
                )
            ),
            paper_bgcolor=CHART_BG,
            font=dict(color='#DDE1EC', family='DM Sans', size=12),
            legend=LEGEND_CFG,
            title=dict(text="Perfil de Audio por Cluster",
                       font=dict(color='#DDE1EC', size=14, family='DM Sans')),
            margin=dict(t=50, b=10, l=10, r=10)
        )
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    selected_cluster = st.selectbox("Ver canciones del cluster",
                                    sorted(df['cluster_name'].unique()))
    cluster_songs = (filtered[filtered['cluster_name'] == selected_cluster]
                     [['track_name', 'artists', 'track_genre', 'popularity',
                       'danceability', 'energy', 'valence']]
                     .sort_values('popularity', ascending=False)
                     .head(15).reset_index(drop=True))
    cluster_songs.index += 1
    st.dataframe(cluster_songs, use_container_width=True)


# ════════════════════════════════════════════════════════════════════
# PÁGINA 4 — SONG FINDER
# ════════════════════════════════════════════════════════════════════
elif page == "🎯 Song Finder":

    st.markdown("## 🎯 Song Finder")
    st.markdown("Busca una canción o ajusta los parámetros para encontrar canciones similares.")

    tab1, tab2 = st.tabs(["🔎 Buscar por nombre", "🎛️ Buscar por parámetros"])

    with tab1:
        search = st.text_input("Nombre de la canción", placeholder="Ej: Blinding Lights")

        if search:
            results = df[df['track_name'].str.contains(search, case=False, na=False)]

            if len(results) == 0:
                st.warning("No se encontró ninguna canción con ese nombre.")
            else:
                selected_song = st.selectbox(
                    "Selecciona la canción exacta",
                    results['track_name'] + " — " + results['artists']
                )

                if selected_song:
                    song_idx = results[
                        (results['track_name'] + " — " + results['artists']) == selected_song
                    ].index[0]
                    song = df.loc[song_idx]

                    st.markdown("---")
                    col1, col2, col3 = st.columns(3)
                    col1.metric("Popularidad", song['popularity'])
                    col2.metric("Cluster",     song['cluster_name'])
                    col3.metric("Género",      song['track_genre'])

                    c1, c2, c3, c4 = st.columns(4)
                    c1.metric("Danceability", f"{song['danceability']:.2f}")
                    c2.metric("Energy",       f"{song['energy']:.2f}")
                    c3.metric("Valence",      f"{song['valence']:.2f}")
                    c4.metric("Tempo",        f"{song['tempo']:.0f} BPM")

                    st.markdown("---")
                    st.markdown('<div class="section-title">Canciones más similares</div>',
                                unsafe_allow_html=True)

                    X_song = scaler.transform(df.loc[[song_idx], feature_cols])
                    distances, indices = nn_model.kneighbors(X_song)

                    similar = df.iloc[indices[0][1:]].copy()
                    similar['similitud'] = (1 - distances[0][1:]).round(3)

                    display_cols = ['track_name', 'artists', 'track_genre',
                                    'popularity', 'cluster_name', 'similitud']
                    st.dataframe(similar[display_cols].reset_index(drop=True),
                                 use_container_width=True)

    with tab2:
        st.markdown("Ajusta los parámetros y encuentra canciones con ese perfil de audio:")

        col1, col2 = st.columns(2)
        with col1:
            p_dance            = st.slider("Danceability",     0.0, 1.0,  0.7,  0.01)
            p_energy           = st.slider("Energy",           0.0, 1.0,  0.7,  0.01)
            p_valence          = st.slider("Valence",          0.0, 1.0,  0.6,  0.01)
            p_acousticness     = st.slider("Acousticness",     0.0, 1.0,  0.1,  0.01)
        with col2:
            p_tempo            = st.slider("Tempo (BPM)",      60,  200,  120)
            p_loudness         = st.slider("Loudness (dB)",    -40,   0,   -7)
            p_speechiness      = st.slider("Speechiness",      0.0, 1.0, 0.05, 0.01)
            p_instrumentalness = st.slider("Instrumentalness", 0.0, 1.0,  0.0, 0.01)

        if st.button("🔍 Encontrar canciones similares", type="primary"):
            query = np.array([[p_dance, p_energy, p_loudness, p_tempo,
                               p_valence, p_acousticness,
                               p_speechiness, p_instrumentalness]])
            query_scaled = scaler.transform(query)
            distances, indices = nn_model.kneighbors(query_scaled)

            similar = df.iloc[indices[0]].copy()
            similar['similitud'] = (1 - distances[0]).round(3)

            st.markdown('<div class="section-title">Canciones que coinciden con tu perfil</div>',
                        unsafe_allow_html=True)
            display_cols = ['track_name', 'artists', 'track_genre',
                            'popularity', 'cluster_name', 'similitud']
            st.dataframe(similar[display_cols].reset_index(drop=True),
                         use_container_width=True)
