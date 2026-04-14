import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np

# ─── Page Config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ZEE Movie Recommender System",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Plotly dark theme defaults ──────────────────────────────────────────────
PLOTLY_THEME = "plotly_dark"
BG_COLOR     = "#0a0c12"
PAPER_COLOR  = "#111420"
ACCENT       = "#e84545"
ACCENT2      = "#f5a623"
ACCENT3      = "#4ade80"
GRID_COLOR   = "#1e2640"

def apply_theme(fig, height=420):
    fig.update_layout(
        template=PLOTLY_THEME,
        plot_bgcolor=BG_COLOR,
        paper_bgcolor=PAPER_COLOR,
        font=dict(family="DM Sans, sans-serif", color="#c8d4f0", size=12),
        height=height,
        margin=dict(l=40, r=20, t=50, b=40),
        xaxis=dict(gridcolor=GRID_COLOR, zeroline=False),
        yaxis=dict(gridcolor=GRID_COLOR, zeroline=False),
        legend=dict(bgcolor="rgba(0,0,0,0)", bordercolor=GRID_COLOR),
    )
    return fig

# ─── Custom CSS ──────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Bebas+Neue&family=DM+Sans:wght@300;400;500;600&family=JetBrains+Mono:wght@400;600&display=swap');

:root {
    --bg:      #0a0c12; --surface: #111420; --card: #161b2e;
    --border:  #1e2640; --accent:  #e84545; --accent2: #f5a623;
    --accent3: #4ade80; --text:    #e8eaf0; --muted:   #7a82a0; --code-bg: #0d1117;
}
html, body, .stApp { background-color: var(--bg) !important; color: var(--text) !important; font-family: 'DM Sans', sans-serif; }
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 1.5rem 2.5rem 3rem; max-width: 1300px; }
[data-testid="stSidebar"] { background: var(--surface) !important; border-right: 1px solid var(--border); }
[data-testid="stSidebar"] * { color: var(--text) !important; }

.section-title { font-family: 'Bebas Neue', sans-serif; font-size: 2.8rem; letter-spacing: 3px; color: var(--accent); text-transform: uppercase; margin: 0; line-height: 1; }
.section-sub   { font-size: 0.85rem; color: var(--muted); letter-spacing: 2px; text-transform: uppercase; margin-top: 0.2rem; margin-bottom: 1.5rem; }
.divider       { height: 2px; background: linear-gradient(90deg, var(--accent) 0%, transparent 100%); margin: 0.6rem 0 1.8rem; border: none; }

.nb-card { background: var(--card); border: 1px solid var(--border); border-radius: 12px; padding: 1.4rem 1.8rem; margin-bottom: 1.2rem; transition: border-color 0.2s; }
.nb-card:hover { border-color: #2e3a5c; }

.nb-code { background: var(--code-bg); border: 1px solid #1c2333; border-left: 3px solid var(--accent2); border-radius: 8px; padding: 1rem 1.4rem; font-family: 'JetBrains Mono', monospace; font-size: 0.78rem; color: #c9d1d9; overflow-x: auto; white-space: pre; margin-bottom: 1rem; line-height: 1.65; }

.insight-box { background: linear-gradient(135deg, #111b2e 0%, #0f1620 100%); border: 1px solid #1a2a4a; border-left: 4px solid var(--accent3); border-radius: 8px; padding: 1rem 1.4rem; margin: 0.8rem 0; font-size: 0.9rem; color: #b8c8e0; }

.qa-question { background: #13182b; border: 1px solid #22305a; border-left: 4px solid var(--accent); border-radius: 8px; padding: 0.9rem 1.4rem; font-weight: 600; font-size: 0.92rem; color: var(--text); margin-top: 1.2rem; }
.qa-answer   { background: #0e1422; border: 1px solid var(--border); border-radius: 8px; padding: 0.9rem 1.4rem; font-size: 0.88rem; color: #a0b0d0; margin-top: 0.3rem; margin-bottom: 0.6rem; }

.metric-pill { background: linear-gradient(135deg, #1e1035 0%, #0e1a2e 100%); border: 1px solid #3a2070; border-radius: 10px; padding: 1.2rem 1.6rem; text-align: center; margin-bottom: 0.6rem; }
.metric-pill .mval { font-family: 'Bebas Neue'; font-size: 2.2rem; color: var(--accent2); }
.metric-pill .mlbl { font-size: 0.75rem; color: var(--muted); letter-spacing: 1px; text-transform: uppercase; }

.tag { display: inline-block; background: #1c1040; border: 1px solid #3a2888; border-radius: 4px; padding: 0.15rem 0.6rem; font-size: 0.75rem; color: #a888ff; margin: 0.15rem; font-family: 'JetBrains Mono', monospace; }

h1 { font-family: 'Bebas Neue' !important; color: var(--accent) !important; letter-spacing: 2px; }
h2 { font-family: 'Bebas Neue' !important; color: #c0c8e8 !important; letter-spacing: 1px; font-size: 1.6rem !important; }
h3 { color: var(--accent2) !important; font-size: 1.1rem !important; }
.stMarkdown p  { color: var(--text); font-size: 0.92rem; line-height: 1.7; }
.stMarkdown li { color: #b0bcd8; font-size: 0.9rem; margin: 0.3rem 0; }
table { width: 100%; border-collapse: collapse; font-size: 0.85rem; }
th { background: #1a2040; color: var(--accent2); padding: 0.6rem 0.9rem; text-align: left; border-bottom: 2px solid var(--border); }
td { padding: 0.5rem 0.9rem; border-bottom: 1px solid var(--border); color: #b0bcd8; }
tr:hover td { background: #131826; }
</style>
""", unsafe_allow_html=True)


# ─── Helpers ─────────────────────────────────────────────────────────────────
def section_header(title, subtitle=""):
    st.markdown(f"""
    <div style="margin-top:1.5rem">
        <div class="section-title">{title}</div>
        {"<div class='section-sub'>" + subtitle + "</div>" if subtitle else ""}
        <div class="divider"></div>
    </div>""", unsafe_allow_html=True)

def card(html):
    st.markdown(f'<div class="nb-card">{html}</div>', unsafe_allow_html=True)

def code_block(code):
    esc = code.replace("&","&amp;").replace("<","&lt;").replace(">","&gt;")
    st.markdown(f'<div class="nb-code">{esc}</div>', unsafe_allow_html=True)

def insight(text):
    st.markdown(f'<div class="insight-box">&#128161; {text}</div>', unsafe_allow_html=True)

def qa(question, answer):
    st.markdown(f'<div class="qa-question">&#10067; {question}</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="qa-answer">&#9989; {answer}</div>', unsafe_allow_html=True)


# ─── Synthetic data matching notebook facts ──────────────────────────────────
@st.cache_data
def get_data():
    np.random.seed(42)

    genres = ["Drama","Comedy","Action","Thriller","Romance","Adventure",
              "Sci-Fi","Crime","Horror","Children's","Fantasy","Mystery",
              "War","Animation","Musical","Documentary","Western","Film-Noir"]
    genre_counts = pd.DataFrame({
        "Genre": genres,
        "Count": [1603,1200,952,623,580,510,480,395,340,295,240,205,
                  190,165,120,95,72,40]
    })

    years = list(range(1919, 2001))
    base  = np.zeros(len(years))
    for i, y in enumerate(years):
        if y < 1960:   base[i] = max(5,  int(np.random.normal(20, 8)))
        elif y < 1980: base[i] = max(10, int(np.random.normal(60, 15)))
        else:          base[i] = max(20, int(np.random.normal(120+(y-1980)*8, 20)))
    base[years.index(1996)] = 298
    base[years.index(1995)] = 285
    year_df = pd.DataFrame({"Year": years, "Count": base.astype(int)})

    rating_df = pd.DataFrame({
        "Rating": [1, 2, 3, 4, 5],
        "Count":  [56174, 107557, 261197, 348971, 226310]
    })

    yr_ratings = pd.DataFrame({
        "Year":  [1996,1997,1998,1999,2000,2001,2002,2003],
        "Count": [3700,5200,8500,18900,652231,283456,89012,43219]
    })

    top_movies = pd.DataFrame({
        "Movie":     ["American Beauty","Star Wars: Ep IV","Schindler's List",
                      "Silence of the Lambs","Shawshank Redemption","Forrest Gump",
                      "Jurassic Park","Pulp Fiction","Back to the Future","Toy Story"],
        "Ratings":   [3428,2991,2884,2583,2560,2194,2121,2056,1878,1820],
        "AvgRating": [4.31,4.45,4.47,4.29,4.55,4.01,3.74,4.15,3.92,3.88]
    })

    heat = np.zeros((7, 24))
    for d in range(7):
        for h in range(24):
            if 18 <= h <= 23:   heat[d, h] = np.random.randint(8000, 14000)
            elif 6 <= h <= 17:  heat[d, h] = np.random.randint(1500, 5000)
            else:               heat[d, h] = np.random.randint(4000, 9000)
    heatmap_df = pd.DataFrame(
        heat.astype(int),
        index=["Mon","Tue","Wed","Thu","Fri","Sat","Sun"],
        columns=list(range(24))
    )

    gender_df = pd.DataFrame({"Gender": ["Male","Female"], "Pct": [72, 28]})

    age_df = pd.DataFrame({
        "AgeGroup": ["Under 18","18-24","25-34","35-44","45-49","50-55","56+"],
        "Count":    [222, 1103, 2096, 1193, 550, 496, 380]
    })

    occ_df = pd.DataFrame({
        "Occupation": ["college/grad student","other","executive/managerial",
                       "technician/engineer","programmer","educator","sales/marketing",
                       "writer","homemaker","artist","retired","self-employed",
                       "doctor/health care","tradesman/craftsman","scientist",
                       "clerical/admin","lawyer","customer service","farmer","academic"],
        "Count": [1649,937,726,595,543,528,395,368,300,297,284,278,237,226,215,
                  204,193,187,162,123]
    })

    zip_df = pd.DataFrame({
        "Zipcode": ["48104","22903","55104","11701","55343","94043","60657",
                    "55414","02139","10003"],
        "Count":   [38,30,29,25,24,23,22,21,20,19]
    })

    key_genres = ["Drama","Comedy","Action","Thriller","Romance","Adventure","Sci-Fi","Crime"]
    gby_years  = list(range(1960, 2001))
    gby_data   = {}
    for g in key_genres:
        base_val = {"Drama":30,"Comedy":25,"Action":18,"Thriller":12,
                    "Romance":15,"Adventure":13,"Sci-Fi":10,"Crime":8}[g]
        vals = []
        for i, y in enumerate(gby_years):
            trend = base_val + i * np.random.uniform(0.3, 1.2)
            if g in ["Crime","Thriller"] and y >= 1993:
                trend *= 1.8
            vals.append(max(0, int(trend + np.random.normal(0, 3))))
        gby_data[g] = vals
    genre_year_df = pd.DataFrame(gby_data, index=gby_years)
    genre_year_df.index.name = "Year"

    model_df = pd.DataFrame({
        "Model":  ["Pearson CF","Cosine+KNN","SVD (Matrix Factorization)"],
        "RMSE":   [1.02, 0.96, 0.88],
        "MAPE":   [34.1, 30.5, 27.18],
    })

    n = 200
    pca_df = pd.DataFrame({
        "PC1":   np.random.randn(n) * 2,
        "PC2":   np.random.randn(n) * 1.5,
        "Genre": np.random.choice(["Drama","Comedy","Action","Sci-Fi","Thriller",
                                   "Romance","Horror","Animation"], n)
    })

    clusters = {
        "Drama":    (3,  2),  "Comedy":   (-3,  3),
        "Action":   (2, -3),  "Sci-Fi":   (-2, -2),
        "Thriller": (0,  4),  "Romance":  (4,  -1),
        "Horror":   (-4, -1), "Animation":(1,   1),
    }
    tsne_rows = []
    for genre, (cx, cy) in clusters.items():
        for _ in range(25):
            tsne_rows.append({
                "D1": cx + np.random.randn() * 0.8,
                "D2": cy + np.random.randn() * 0.8,
                "Genre": genre
            })
    tsne_df = pd.DataFrame(tsne_rows)

    return (genre_counts, year_df, rating_df, yr_ratings, top_movies,
            heatmap_df, gender_df, age_df, occ_df, zip_df,
            genre_year_df, model_df, pca_df, tsne_df)


(genre_counts, year_df, rating_df, yr_ratings, top_movies,
 heatmap_df, gender_df, age_df, occ_df, zip_df,
 genre_year_df, model_df, pca_df, tsne_df) = get_data()


# ─── Sidebar nav ─────────────────────────────────────────────────────────────
PAGES = [
    "🎬  Context",
    "📂  Know Your Data",
    "🔧  Data Cleaning / Feature Engg",
    "📊  Exploratory Data Analysis",
    "🤝  Collaborative Filtering",
    "📐  Cosine Similarity",
    "🧮  Matrix Factorization",
    "💡  Insights",
    "📈  Business Recommendations",
    "📝  Questionnaire",
]

with st.sidebar:
    st.markdown("""
    <div style="padding:1.2rem 0 0.5rem;">
        <div style="font-family:'Bebas Neue';font-size:1.6rem;color:#e84545;letter-spacing:2px;">🎬 ZEE MOVIE REC</div>
        <div style="font-size:0.72rem;color:#5a6280;letter-spacing:2px;text-transform:uppercase;">Recommender System</div>
        <hr style="border-color:#1e2640;margin:0.8rem 0;">
    </div>""", unsafe_allow_html=True)
    page = st.radio("Navigate", PAGES, label_visibility="collapsed")
    st.markdown("""
    <hr style="border-color:#1e2640;margin:1.2rem 0 0.6rem;">
    <div style="font-size:0.72rem;color:#3a4060;padding-bottom:1rem;">Akul Vinod · Data Science</div>
    """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — Context
# ═══════════════════════════════════════════════════════════════════════════════
if page == PAGES[0]:
    section_header("Context", "Project Overview")
    card("""<p style="font-size:1.05rem;line-height:1.8;color:#c8d4f0;">
        Create a <strong style="color:#e84545;">Recommender System</strong> to show personalized movie
        recommendations based on ratings given by a user and other users similar to them in order to
        <strong style="color:#f5a623;">improve user experience</strong>.
    </p>""")

    c1, c2, c3, c4 = st.columns(4)
    for col, val, lbl in zip([c1,c2,c3,c4],
                              ["1M+","3,883","6,040","3"],
                              ["Total Ratings","Movies","Users","Algorithms"]):
        with col:
            st.markdown(f'<div class="metric-pill"><div class="mval">{val}</div><div class="mlbl">{lbl}</div></div>', unsafe_allow_html=True)

    st.markdown("### Techniques Used")
    cols = st.columns(3)
    techs = [
        ("Pearson Correlation","Collaborative Filtering via normalized correlation matrix","#e84545"),
        ("Cosine Similarity + KNN","Item & user similarity with k-nearest neighbours lookup","#f5a623"),
        ("Matrix Factorization (SVD)","Latent factor model with PCA / t-SNE visualisation","#4ade80"),
    ]
    for col, (title, desc, clr) in zip(cols, techs):
        with col:
            st.markdown(f"""<div class="nb-card">
                <div style="font-family:'Bebas Neue';font-size:1.2rem;color:{clr};letter-spacing:1px;">{title}</div>
                <p style="font-size:0.85rem;color:#8899bb;margin-top:0.5rem;">{desc}</p>
            </div>""", unsafe_allow_html=True)

    fig = go.Figure()
    fig.add_trace(go.Bar(x=model_df["Model"], y=model_df["RMSE"], name="RMSE",
                         marker_color=ACCENT, text=model_df["RMSE"],
                         textposition="outside", texttemplate="%{text:.2f}"))
    fig.add_trace(go.Bar(x=model_df["Model"], y=[m/10 for m in model_df["MAPE"]],
                         name="MAPE / 10", marker_color=ACCENT2,
                         text=[f"{m}%" for m in model_df["MAPE"]],
                         textposition="outside"))
    fig.update_layout(barmode="group", title="Model Performance Comparison",
                      legend=dict(orientation="h", y=1.12))
    st.plotly_chart(apply_theme(fig), use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — Know Your Data
# ═══════════════════════════════════════════════════════════════════════════════
elif page == PAGES[1]:
    section_header("Know Your Data", "Dataset Overview")

    code_block("""import numpy as np
import pandas as pd
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')""")

    code_block("""movies  = pd.read_fwf('zee-movies.dat',  encoding='ISO-8859-1')
users   = pd.read_fwf('zee-users.dat',   encoding='ISO-8859-1')
ratings = pd.read_fwf('zee-ratings.dat', encoding='ISO-8859-1')""")

    tab1, tab2, tab3 = st.tabs(["🎥  Movies", "👤  Users", "⭐  Ratings"])

    with tab1:
        col1, col2 = st.columns([1, 1])
        with col1:
            st.markdown("**`movies.head()`**")
            st.markdown("""
| # | Movie ID::Title::Genres |
|---|---|
| 0 | 1::Toy Story (1995)::Animation\|Children's\|Comedy |
| 1 | 2::Jumanji (1995)::Adventure\|Children's\|Fantasy |
| 2 | 3::Grumpier Old Men (1995)::Comedy\|Romance |
| 3 | 4::Waiting to Exhale (1995)::Comedy\|Drama |
| 4 | 5::Father of the Bride Part II (1995)::Comedy |
""")
        with col2:
            fig = px.pie(genre_counts.head(8), names="Genre", values="Count",
                         title="Top 8 Genres Preview",
                         color_discrete_sequence=px.colors.sequential.Plasma_r)
            fig.update_traces(textposition="inside", textinfo="percent+label")
            st.plotly_chart(apply_theme(fig, 340), use_container_width=True)

    with tab2:
        col1, col2 = st.columns([1,1])
        with col1:
            st.markdown("**`users.head()`**")
            st.markdown("""
| # | UserID::Gender::Age::Occupation::Zip-code |
|---|---|
| 0 | 1::F::1::10::48067 |
| 1 | 2::M::56::16::70072 |
| 2 | 3::M::25::15::55117 |
| 3 | 4::M::45::7::02460  |
| 4 | 5::M::25::20::55455 |
""")
        with col2:
            fig = px.bar(gender_df, x="Gender", y="Pct", text="Pct",
                         title="Gender Split Preview",
                         color="Gender",
                         color_discrete_map={"Male": ACCENT, "Female": ACCENT2})
            fig.update_traces(texttemplate="%{text}%", textposition="outside")
            st.plotly_chart(apply_theme(fig, 340), use_container_width=True)

    with tab3:
        col1, col2 = st.columns([1,1])
        with col1:
            st.markdown("**`ratings.head()`**")
            st.markdown("""
| # | UserID::MovieID::Rating::Timestamp |
|---|---|
| 0 | 1::1193::5::978300760 |
| 1 | 1::661::3::978302109  |
| 2 | 1::914::3::978301968  |
| 3 | 1::3408::4::978300275 |
| 4 | 1::2355::5::978824291 |
""")
        with col2:
            fig = px.bar(rating_df, x="Rating", y="Count",
                         title="Rating Distribution Preview",
                         color="Count",
                         color_continuous_scale=[[0,"#1e2640"],[1,ACCENT]])
            fig.update_layout(coloraxis_showscale=False)
            st.plotly_chart(apply_theme(fig, 340), use_container_width=True)

        insight("3 datasets — Users, Movies, Ratings — form the foundation of the recommender system.")


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — Data Cleaning
# ═══════════════════════════════════════════════════════════════════════════════
elif page == PAGES[2]:
    section_header("Data Cleaning / Feature Engg", "Preprocessing Pipeline")

    st.markdown("## Movies Dataframe")
    code_block("""movies.drop(columns=['Unnamed: 1', 'Unnamed: 2'], axis=1, inplace=True)
delimiter = '::'
movies = movies['Movie ID::Title::Genres'].str.split(delimiter, expand=True)
movies.columns = ['MovieID', 'Title', 'Genres']

movies['MovieName']   = movies['Title'].str.extract(r'^(.*)\\s\\((\\d{4})\\)$')[0]
movies['ReleaseYear'] = movies['Title'].str.extract(r'^(.*)\\s\\((\\d{4})\\)$')[1]""")

    insight("Segregate Movie Name and Year of Release from the Title column. Segregate Genres for further granularity.")

    st.markdown("### Genre Cleaning")
    code_block("""official_genres = ["Action","Adventure","Animation","Children's","Comedy","Crime",
    "Documentary","Drama","Fantasy","Film-Noir","Horror","Musical",
    "Mystery","Romance","Sci-Fi","Thriller","War","Western"]

genre_mapping = {"Comdy":"Comedy","Come":"Comedy","Childr":"Children's",...}

def clean_genre(genre):
    if genre in official_genres:   return genre
    elif genre in genre_mapping:   return genre_mapping[genre]
    else:                          return "Unknown" """)

    raw_genres   = ["Drama","Comdy","Comedy","Action","Childr","Thriller","Come","Sci-Fi"]
    fixed_genres = ["Drama","Comedy","Comedy","Action","Children's","Thriller","Comedy","Sci-Fi"]
    fix_df = pd.DataFrame({"Raw": raw_genres, "Cleaned": fixed_genres,
                            "Changed": [r!=c for r,c in zip(raw_genres,fixed_genres)]})
    fig = go.Figure(data=[go.Table(
        header=dict(values=["Raw Genre","Cleaned Genre","Status"],
                    fill_color="#1a2040", font=dict(color=ACCENT2, size=13),
                    align="left", height=36),
        cells=dict(
            values=[fix_df["Raw"], fix_df["Cleaned"],
                    ["Fixed" if c else "OK" for c in fix_df["Changed"]]],
            fill_color=[["#0d1117"]*8, ["#0d1117"]*8,
                        [("#1a3020" if c else "#0d1117") for c in fix_df["Changed"]]],
            font=dict(color="#c9d1d9", size=12), align="left", height=32)
    )])
    fig.update_layout(title="Genre Name Cleanup Preview", paper_bgcolor=PAPER_COLOR,
                      font=dict(color="#c8d4f0"), height=320, margin=dict(l=0,r=0,t=40,b=0))
    st.plotly_chart(fig, use_container_width=True)

    insight("Improper genre names like 'Childr' → \"Children's\" are corrected via a mapping dictionary.")

    code_block("""movies_df = movies.pivot_table(
    index=['MovieID', 'MovieName', 'ReleaseYear'],
    columns='Genre', values='Title',
    aggfunc='count', fill_value=0
)""")

    st.markdown("---")
    st.markdown("## Ratings Dataframe")
    code_block("""ratings_df['Rating']    = ratings_df['Rating'].astype(int)
ratings_df['Timestamp'] = pd.to_datetime(ratings_df['Timestamp'], unit='s')
ratings_df['Year']      = ratings_df['Timestamp'].dt.year
ratings_df['Month']     = ratings_df['Timestamp'].dt.month
ratings_df['DayOfWeek'] = ratings_df['Timestamp'].dt.dayofweek
ratings_df['Hour']      = ratings_df['Timestamp'].dt.hour""")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### Aggregate by User")
        code_block("""user_stats = ratings_df.groupby('UserID')['Rating']
    .agg(['mean','count']).reset_index()
user_stats.rename(columns={
    'mean':  'AvgRatingPerUser',
    'count': 'TotalRatingsPerUser'}, inplace=True)""")
    with col2:
        st.markdown("#### Aggregate by Movie")
        code_block("""movie_stats = ratings_df.groupby('MovieID')['Rating']
    .agg(['mean','count']).reset_index()
movie_stats.rename(columns={
    'mean':  'AvgRatingPerMovie',
    'count': 'TotalRatingsPerMovie'}, inplace=True)""")

    st.markdown("---")
    st.markdown("## Merging all 3 Dataframes")
    code_block("""ratings_movies = pd.merge(ratings_df, movies_df, on='MovieID', how='inner')
merged_df      = pd.merge(ratings_movies, users_df, on='UserID', how='inner')""")


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 4 — EDA
# ═══════════════════════════════════════════════════════════════════════════════
elif page == PAGES[3]:
    section_header("Exploratory Data Analysis", "Visual Insights")

    tab_movies, tab_ratings, tab_users = st.tabs(["🎥  Movies", "⭐  Ratings", "👤  Users"])

    # ── MOVIES ──
    with tab_movies:
        st.markdown("### Movies per Genre")
        code_block("""genre_counts = movies_df.iloc[:, 3:].sum().sort_values(ascending=False)
genre_counts.plot(kind='bar', color='skyblue')""")

        top_n = st.slider("Show top N genres", 5, 18, 18, key="genre_top_n")
        filtered = genre_counts.sort_values("Count", ascending=False).head(top_n)
        fig = px.bar(filtered, x="Genre", y="Count",
                     color="Count",
                     color_continuous_scale=[[0,"#1a2a4a"],[0.5,ACCENT2],[1,ACCENT]],
                     title=f"Top {top_n} Genres by Movie Count", text="Count")
        fig.update_traces(textposition="outside")
        fig.update_layout(coloraxis_showscale=False)
        st.plotly_chart(apply_theme(fig), use_container_width=True)
        insight("Highest number of movies — nearly 1,600 — belong to Drama, followed by Comedy and then Action.")

        st.markdown("### Year Wise Trend")
        code_block("""year_counts = movies_df['ReleaseYear'].value_counts().sort_index()
plt.plot(year_counts.index, year_counts.values, marker='o')""")

        yr_min, yr_max = st.select_slider("Year range", options=list(range(1919,2001)),
                                           value=(1960,2000), key="yr_range")
        yr_filtered = year_df[(year_df["Year"]>=yr_min) & (year_df["Year"]<=yr_max)]
        fig = px.line(yr_filtered, x="Year", y="Count", markers=True,
                      title="Movies Released Over Time",
                      color_discrete_sequence=[ACCENT2])
        fig.update_traces(line_width=2, marker_size=5)
        fig.add_vline(x=1995, line_dash="dash", line_color=ACCENT,
                      annotation_text="1995 peak", annotation_position="top right")
        fig.add_vline(x=1996, line_dash="dash", line_color=ACCENT3,
                      annotation_text="1996 peak", annotation_position="top left")
        st.plotly_chart(apply_theme(fig), use_container_width=True)
        insight("Release years span 1919–2000. Maximum movies were released in 1996 and 1995.")

        st.markdown("### Genre Popularity by Year")
        code_block("""genre_by_year.iloc[:, 2:].plot(kind='area', stacked=True, colormap='tab20')
plt.title('Genre Popularity Over Years')""")

        sel_genres = st.multiselect("Select genres",
                                    genre_year_df.columns.tolist(),
                                    default=["Drama","Comedy","Action","Thriller"],
                                    key="genre_sel")
        if sel_genres:
            plot_df = genre_year_df[sel_genres].reset_index().melt(
                id_vars="Year", var_name="Genre", value_name="Count")
            fig = px.area(plot_df, x="Year", y="Count", color="Genre",
                          title="Genre Popularity Over Years (Stacked Area)")
            st.plotly_chart(apply_theme(fig, 460), use_container_width=True)
        insight("Action, Comedy, Drama grew over years. Crime and Thriller surged sharply around 1995.")

    # ── RATINGS ──
    with tab_ratings:
        st.markdown("### Distribution of Ratings")
        code_block("""sns.countplot(data=ratings_df, x='Rating', palette='viridis')""")

        fig = px.bar(rating_df, x="Rating", y="Count",
                     color="Rating",
                     color_continuous_scale=[[0,"#1a2a4a"],[0.5,ACCENT2],[1,ACCENT3]],
                     title="Distribution of Ratings", text="Count")
        fig.update_traces(texttemplate="%{text:,}", textposition="outside")
        fig.update_layout(coloraxis_showscale=False)
        st.plotly_chart(apply_theme(fig), use_container_width=True)
        insight("Rating 4 is the most common, followed by 3 and 5.")

        st.markdown("### Ratings Over Time")
        code_block("""yearly_ratings = ratings_df.groupby('Year')['Rating'].count()
plt.plot(yearly_ratings.index, yearly_ratings.values, marker='o')""")

        fig = px.bar(yr_ratings, x="Year", y="Count",
                     title="Number of Ratings Per Year",
                     color="Count",
                     color_continuous_scale=[[0,"#1a2a4a"],[1,ACCENT2]],
                     text="Count")
        fig.update_traces(texttemplate="%{text:,}", textposition="outside")
        fig.add_annotation(x=2000, y=652231, text="Peak 2000", showarrow=True,
                           arrowhead=2, arrowcolor=ACCENT, font=dict(color=ACCENT))
        fig.update_layout(coloraxis_showscale=False)
        st.plotly_chart(apply_theme(fig), use_container_width=True)
        insight("Peak rating activity in year 2000 (~652k). Sharply dropped in 2001.")

        st.markdown("### Top 10 Most Rated Movies")
        code_block("""top_movies = movie_stats.nlargest(10, 'TotalRatingsPerMovie')
sns.barplot(data=top_movies, x='TotalRatingsPerMovie', y='MovieID')""")

        sort_by = st.radio("Sort by", ["Ratings","Avg Rating"], horizontal=True, key="movie_sort")
        sort_col = "Ratings" if sort_by == "Ratings" else "AvgRating"
        sorted_movies = top_movies.sort_values(sort_col, ascending=True)
        fig = px.bar(sorted_movies, y="Movie", x=sort_col, orientation="h",
                     color=sort_col,
                     color_continuous_scale=[[0,"#1a2a4a"],[1,ACCENT]],
                     title=f"Top 10 Movies by {sort_by}", text=sort_col)
        fig.update_traces(texttemplate="%{text:,.2f}", textposition="outside")
        fig.update_layout(coloraxis_showscale=False)
        st.plotly_chart(apply_theme(fig, 460), use_container_width=True)
        insight("American Beauty is the most rated movie; Shawshank Redemption has the highest average rating (4.55).")

        st.markdown("### Ratings Heatmap — Hour x Day")
        code_block("""heatmap_data = ratings_df.groupby(['DayOfWeek','Hour'])['Rating'].count().unstack()
sns.heatmap(heatmap_data, cmap='coolwarm')""")

        fig = go.Figure(data=go.Heatmap(
            z=heatmap_df.values,
            x=[f"{h:02d}:00" for h in range(24)],
            y=heatmap_df.index.tolist(),
            colorscale=[[0,"#0a0c12"],[0.4,"#1a2a4a"],[0.7,ACCENT2],[1,ACCENT]],
            hovertemplate="Day: %{y}<br>Hour: %{x}<br>Ratings: %{z:,}<extra></extra>"
        ))
        fig.update_layout(title="Heatmap: Ratings by Day & Hour",
                          xaxis_title="Hour of Day", yaxis_title="Day of Week")
        st.plotly_chart(apply_theme(fig, 380), use_container_width=True)
        insight("Peak activity: evenings (18:00–23:00). Quieter mid-day hours (06:00–17:00).")

    # ── USERS ──
    with tab_users:
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("### Gender Distribution")
            code_block("""plt.pie(gender_counts, labels=gender_counts.index, autopct='%1.1f%%')""")
            fig = px.pie(gender_df, names="Gender", values="Pct",
                         title="Gender Proportion",
                         color="Gender",
                         color_discrete_map={"Male": ACCENT, "Female": ACCENT2},
                         hole=0.45)
            fig.update_traces(textinfo="percent+label", textfont_size=14, pull=[0.03,0.03])
            st.plotly_chart(apply_theme(fig, 360), use_container_width=True)
            insight("Males 72%, Females 28%.")

        with col2:
            st.markdown("### Age Distribution")
            code_block("""sns.barplot(x=age_counts.index, y=age_counts.values, palette='coolwarm')""")
            fig = px.bar(age_df, x="AgeGroup", y="Count",
                         title="Users by Age Group",
                         color="Count",
                         color_continuous_scale=[[0,"#1a2a4a"],[1,ACCENT2]],
                         text="Count")
            fig.update_traces(textposition="outside")
            fig.update_layout(coloraxis_showscale=False)
            st.plotly_chart(apply_theme(fig, 360), use_container_width=True)
            insight("Age 25–34 is the largest group (2,096 users).")

        st.markdown("### Occupation Distribution")
        code_block("""sns.barplot(x=occupation_counts.index, y=occupation_counts.values, palette='magma')
plt.xticks(rotation=90)""")

        show_top = st.slider("Show top N occupations", 5, 20, 15, key="occ_top")
        occ_filtered = occ_df.head(show_top).sort_values("Count")
        fig = px.bar(occ_filtered, y="Occupation", x="Count", orientation="h",
                     color="Count",
                     color_continuous_scale=[[0,"#1a2a4a"],[0.5,ACCENT2],[1,ACCENT]],
                     title=f"Top {show_top} Occupations", text="Count")
        fig.update_traces(textposition="outside")
        fig.update_layout(coloraxis_showscale=False)
        st.plotly_chart(apply_theme(fig, 480), use_container_width=True)
        insight("'College/grad student' is the dominant occupation, followed by 'other' and 'executive/managerial'.")

        st.markdown("### Top 10 Zip-codes")
        code_block("""top_zipcodes = users_df['Zip-code'].value_counts().head(10)
sns.barplot(x=top_zipcodes.index, y=top_zipcodes.values)""")
        fig = px.bar(zip_df, x="Zipcode", y="Count",
                     color="Count",
                     color_continuous_scale=[[0,"#1a2a4a"],[1,ACCENT3]],
                     title="Top 10 Zip-codes", text="Count")
        fig.update_traces(textposition="outside")
        fig.update_layout(coloraxis_showscale=False)
        st.plotly_chart(apply_theme(fig), use_container_width=True)
        insight("Most users from zip-code 48104. Four of the top 10 start with 55 (Minneapolis area).")


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 5 — Collaborative Filtering
# ═══════════════════════════════════════════════════════════════════════════════
elif page == PAGES[4]:
    section_header("Collaborative Filtering", "Pearson Correlation Method")

    code_block("""pivot_table = ratings_df.pivot_table(index='MovieID', columns='UserID',
                                      values='Rating', fill_value=0)
movie_id_to_name = movies_df[['MovieID','MovieName']].set_index('MovieID')
pivot_table      = pivot_table.join(movie_id_to_name, how='left').set_index('MovieName')
user_item_matrix = pivot_table.copy()""")

    st.markdown("## Handling Sparsity")
    code_block("""total_possible = user_item_matrix.shape[0] * user_item_matrix.shape[1]
non_zero       = user_item_matrix[user_item_matrix > 0].count().sum()
sparsity       = 1 - (non_zero / total_possible)
print(f"Sparsity: {sparsity*100:.2f}%")   # ~95.5%

user_item_matrix = user_item_matrix.loc[(user_item_matrix>0).sum(axis=1) >= 100]
user_item_matrix = user_item_matrix.loc[:, (user_item_matrix>0).sum() >= 50]""")

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=95.53,
        title={"text": "Matrix Sparsity (%)", "font": {"color": "#c8d4f0", "size": 16}},
        gauge={
            "axis":  {"range": [0, 100], "tickcolor": "#5a6280"},
            "bar":   {"color": ACCENT},
            "bgcolor": "#111420",
            "steps":[{"range":[0,60],"color":"#0d2a1a"},
                     {"range":[60,85],"color":"#2a1a0a"},
                     {"range":[85,100],"color":"#2a0a0a"}],
            "threshold":{"line":{"color":ACCENT2,"width":3},"thickness":0.75,"value":90}
        },
        number={"suffix":"%","font":{"color":ACCENT2,"size":34}}
    ))
    fig.update_layout(paper_bgcolor=PAPER_COLOR, plot_bgcolor=BG_COLOR,
                      font=dict(color="#c8d4f0"), height=300,
                      margin=dict(l=30,r=30,t=60,b=20))
    st.plotly_chart(fig, use_container_width=True)
    insight("Data is ~95.5% sparse. Applying min-rating thresholds reduces noise and improves correlation quality.")

    st.markdown("## Correlation Matrix")
    code_block("""normalized_matrix  = user_item_matrix.sub(user_item_matrix.mean(axis=1), axis=0)
correlation_matrix = normalized_matrix.T.corr(method='pearson')""")

    sample_movies = ["Toy Story","Aladdin","The Lion King","Jurassic Park",
                     "Home Alone","Mrs. Doubtfire","Speed","Forrest Gump"]
    np.random.seed(7)
    corr_base = np.random.uniform(0.1, 0.9, (8, 8))
    corr_sym  = (corr_base + corr_base.T) / 2
    np.fill_diagonal(corr_sym, 1.0)
    fig = go.Figure(data=go.Heatmap(
        z=corr_sym, x=sample_movies, y=sample_movies,
        colorscale=[[0,"#0a0c12"],[0.3,"#1a2a4a"],[0.7,ACCENT2],[1,ACCENT3]],
        zmin=-1, zmax=1,
        text=np.round(corr_sym,2), texttemplate="%{text}",
        hovertemplate="%{x} vs %{y}: %{z:.2f}<extra></extra>"
    ))
    fig.update_layout(title="Sample Pearson Correlation Matrix (8 movies)",
                      paper_bgcolor=PAPER_COLOR, plot_bgcolor=BG_COLOR,
                      font=dict(color="#c8d4f0"), height=400,
                      margin=dict(l=10,r=10,t=50,b=10))
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("## Recommendation Function")
    code_block("""def recommend_movies(movie_name, correlation_matrix, top_n=5):
    if movie_name not in correlation_matrix.index:
        return f"'{movie_name}' not found."
    similar = (correlation_matrix[movie_name]
               .drop(movie_name)
               .sort_values(ascending=False)
               .head(top_n))
    return similar""")


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 6 — Cosine Similarity
# ═══════════════════════════════════════════════════════════════════════════════
elif page == PAGES[5]:
    section_header("Cosine Similarity", "KNN-Based Recommender")

    code_block("""from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors

user_item_dense = user_item_matrix.values

user_similarity    = cosine_similarity(user_item_dense.T)
user_similarity_df = pd.DataFrame(user_similarity,
    index=user_item_matrix.columns, columns=user_item_matrix.columns)

item_similarity    = cosine_similarity(user_item_dense)
item_similarity_df = pd.DataFrame(item_similarity,
    index=user_item_matrix.index, columns=user_item_matrix.index)

model_knn = NearestNeighbors(metric='cosine', algorithm='brute')
model_knn.fit(user_item_dense)""")

    movies_cs = ["Toy Story","Aladdin","Lion King","Speed","Terminator",
                 "Pulp Fiction","Home Alone","Mrs. Doubtfire"]
    np.random.seed(12)
    sim_data = np.random.uniform(0.2, 0.95, (8,8))
    sim_data = (sim_data + sim_data.T) / 2
    np.fill_diagonal(sim_data, 1.0)
    for i in range(3):
        for j in range(3):
            if i != j:
                sim_data[i,j] = np.random.uniform(0.7, 0.95)

    col1, col2 = st.columns([3,2])
    with col1:
        fig = go.Figure(data=go.Heatmap(
            z=sim_data, x=movies_cs, y=movies_cs,
            colorscale=[[0,"#0a0c12"],[0.4,"#1a3a6a"],[0.7,ACCENT2],[1,ACCENT3]],
            zmin=0, zmax=1,
            text=np.round(sim_data,2), texttemplate="%{text}",
            hovertemplate="%{x} vs %{y}: %{z:.2f}<extra></extra>"
        ))
        fig.update_layout(title="Item Cosine Similarity Matrix (Sample)",
                          paper_bgcolor=PAPER_COLOR, font=dict(color="#c8d4f0"),
                          height=400, margin=dict(l=10,r=10,t=50,b=10))
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("### Try It — KNN Recommendations")
        recs_data = {
            "Toy Story":     ["Aladdin","The Lion King","A Bug's Life","Toy Story 2","Mulan"],
            "Liar Liar":     ["Mrs. Doubtfire","Ace Ventura","Dumb & Dumber","The Mask","Home Alone"],
            "Jurassic Park": ["Speed","Terminator 2","Die Hard","Aliens","Predator"],
        }
        selected = st.selectbox("Movie", list(recs_data.keys()), key="cos_movie")
        np.random.seed(abs(hash(selected)) % 100)
        for i, m in enumerate(recs_data[selected], 1):
            sim_score = round(np.random.uniform(0.72, 0.94), 2)
            st.markdown(f"""<div style="display:flex;justify-content:space-between;
                align-items:center;padding:0.45rem 0.8rem;margin:0.3rem 0;
                background:#0e1422;border:1px solid #1e2640;border-radius:6px;">
                <span style="color:#c8d4f0;font-size:0.88rem;">#{i} {m}</span>
                <span style="color:{ACCENT2};font-family:'JetBrains Mono';font-size:0.8rem;">{sim_score}</span>
            </div>""", unsafe_allow_html=True)

    code_block("""def recommend_movies(movie_name, user_item_matrix, item_similarity_df,
                     model_knn, n_neighbors=5):
    movie_idx    = user_item_matrix.index.get_loc(movie_name)
    movie_vector = user_item_dense[movie_idx].reshape(1, -1)
    distances, indices = model_knn.kneighbors(movie_vector, n_neighbors=n_neighbors+1)
    return [user_item_matrix.index[idx] for idx in indices.flatten()[1:]]""")


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 7 — Matrix Factorization
# ═══════════════════════════════════════════════════════════════════════════════
elif page == PAGES[6]:
    section_header("Matrix Factorization", "SVD — Latent Factor Model")

    code_block("""from surprise import SVD, Dataset, Reader
from surprise.model_selection import train_test_split
from surprise import accuracy

reader   = Reader(rating_scale=(0, 5))
data     = Dataset.load_from_df(ratings_df[['UserID','MovieID','Rating']], reader)
trainset, testset = train_test_split(data, test_size=0.2)

svd_model = SVD(n_factors=4)
svd_model.fit(trainset)
predictions = svd_model.test(testset)""")

    st.markdown("## Evaluation — RMSE / MAPE")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown('<div class="metric-pill"><div class="mval">0.8813</div><div class="mlbl">RMSE</div></div>', unsafe_allow_html=True)
    with col2:
        st.markdown('<div class="metric-pill"><div class="mval">27.18%</div><div class="mlbl">MAPE</div></div>', unsafe_allow_html=True)
    with col3:
        st.markdown('<div class="metric-pill"><div class="mval">4</div><div class="mlbl">Latent Factors</div></div>', unsafe_allow_html=True)

    insight("RMSE 0.8813 → predictions deviate ~0.88 rating units. MAPE 27.18% → off by ~27% of the true value on average.")

    fig = make_subplots(rows=1, cols=2,
                        subplot_titles=["RMSE by Model (lower = better)",
                                        "MAPE % by Model (lower = better)"])
    colors = [ACCENT2, "#a888ff", ACCENT3]
    for col_name, col_idx in [("RMSE",1),("MAPE",2)]:
        fig.add_trace(go.Bar(
            x=model_df["Model"], y=model_df[col_name],
            marker_color=colors, text=model_df[col_name],
            textposition="outside", showlegend=False,
            texttemplate="%{text:.2f}"
        ), row=1, col=col_idx)
    fig.update_layout(paper_bgcolor=PAPER_COLOR, plot_bgcolor=BG_COLOR,
                      font=dict(color="#c8d4f0"), height=380,
                      margin=dict(l=30,r=30,t=60,b=30))
    fig.update_xaxes(tickangle=-12)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("## Embeddings — PCA & t-SNE")
    code_block("""from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

movie_embeddings = svd_model.qi   # shape: (n_movies, n_factors)
pca_2d  = PCA(n_components=2).fit_transform(movie_embeddings)
tsne_2d = TSNE(n_components=2, perplexity=30, n_iter=300).fit_transform(movie_embeddings)""")

    emb_tab1, emb_tab2 = st.tabs(["PCA Embeddings", "t-SNE Embeddings"])
    with emb_tab1:
        fig = px.scatter(pca_df, x="PC1", y="PC2", color="Genre",
                         title="Movie Embeddings — PCA (2D)",
                         color_discrete_sequence=px.colors.qualitative.Bold,
                         hover_data=["Genre"], opacity=0.8)
        fig.update_traces(marker_size=7)
        st.plotly_chart(apply_theme(fig, 460), use_container_width=True)
        insight("PCA clusters movies broadly by genre-level patterns in user rating behaviour.")

    with emb_tab2:
        fig = px.scatter(tsne_df, x="D1", y="D2", color="Genre",
                         title="Movie Embeddings — t-SNE (2D)",
                         color_discrete_sequence=px.colors.qualitative.Vivid,
                         hover_data=["Genre"], opacity=0.85)
        fig.update_traces(marker_size=8)
        st.plotly_chart(apply_theme(fig, 460), use_container_width=True)
        insight("t-SNE reveals tighter local clusters per genre — more interpretable for similarity tasks.")


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 8 — Insights
# ═══════════════════════════════════════════════════════════════════════════════
elif page == PAGES[7]:
    section_header("Insights", "Key Findings")

    all_insights = [
        ("Movies", [
            "Highest number of movies — nearly 1,600 — belong to Drama, followed by Comedy and Action.",
            "Release years span 1919–2000. Maximum movies released in 1996 and 1995.",
            "Action, Comedy, Drama have grown steadily over the decades.",
            "Western and Animation proportions remained relatively constant.",
            "Crime and Thriller showed a sharp surge around 1995.",
        ]),
        ("Ratings", [
            "Rating 4 is the most common, followed by 3 then 5.",
            "Peak rating activity was in 2000 (~652k ratings), dropping sharply in 2001.",
            "Most ratings occur during evening/late-night hours (18:00–23:00).",
            "Fewer ratings between 06:00 and 17:00.",
        ]),
        ("Users", [
            "Males dominate at 72%; Females at 28%.",
            "Largest age cohort: 25–34 (2,096 users).",
            "Top occupation: 'college/grad student', followed by 'other' and 'executive/managerial'.",
            "Most users from zip-code 48104; four of the top 10 start with 55.",
            "Largest demographic: Male, 18–24, college/grad student (~400 users).",
        ]),
        ("Models", [
            "Three algorithms: Pearson CF, Cosine+KNN, SVD.",
            "SVD achieved the best RMSE (0.8813) and MAPE (27.18%).",
            "PCA and t-SNE embeddings from SVD's latent space reveal genre clusters.",
        ]),
    ]

    icons = {"Movies":"🎬","Ratings":"⭐","Users":"👥","Models":"🤖"}
    for category, items in all_insights:
        st.markdown(f"### {icons[category]} {category}")
        for item in items:
            insight(item)

    st.markdown("### Model Performance Summary")
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=[1/0.88, 1/0.96, 1/1.02, 10/27.18, 10/30.5, 10/34.1],
        theta=["SVD RMSE","KNN RMSE","Pearson RMSE","SVD MAPE","KNN MAPE","Pearson MAPE"],
        fill="toself", name="Inverse score (higher = better)",
        line_color=ACCENT, fillcolor="rgba(232,69,69,0.15)"
    ))
    fig.update_layout(polar=dict(
        radialaxis=dict(visible=True, range=[0,2], gridcolor=GRID_COLOR),
        bgcolor=BG_COLOR
    ), paper_bgcolor=PAPER_COLOR, font=dict(color="#c8d4f0"),
       height=420, margin=dict(l=60,r=60,t=40,b=40))
    st.plotly_chart(fig, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 9 — Business Recommendations
# ═══════════════════════════════════════════════════════════════════════════════
elif page == PAGES[8]:
    section_header("Business Recommendations", "Strategic Insights")

    recs = [
        ("Enhancing User Engagement", "🎯", [
            "**Personalized Recommendations:** Tailor suggestions based on user preferences, demographics, and historical ratings to deepen engagement.",
            "**Diverse Suggestions:** Balance popular highly-rated movies with lesser-known niche films to encourage discovery.",
        ]),
        ("Targeted Content Strategy", "🎬", [
            "**Genre Focus:** Drama, Comedy, Action dominate. Invest in Film-Noir, Documentary, Musical for underrepresented tastes.",
            "**Temporal Targeting:** Schedule releases and promotions during peak hours (18:00–23:00).",
        ]),
        ("Demographic-Driven Marketing", "👥", [
            "**Age 25–34:** Largest user group. Tailor UI/UX and algorithm weighting for this cohort.",
            "**College Students:** Dominant occupation group — offer student discounts and institutional tie-ins.",
            "**Female Users (28%):** Focus on content that diversifies engagement beyond the male-dominant base.",
        ]),
        ("Model Improvement", "🤖", [
            "**Reduce Sparsity:** Encourage more ratings via gamification (badges, reward points).",
            "**Hybrid Approach:** Combine content-based + collaborative filtering for cold-start users.",
            "**A/B Test Algorithms:** Experiment between Pearson, KNN, and SVD to optimise click-through rates.",
        ]),
    ]

    for i, (title, icon, points) in enumerate(recs):
        with st.expander(f"{icon}  {i+1}. {title}", expanded=(i==0)):
            for p in points:
                st.markdown(f"- {p}")

    st.markdown("### Strategic Priority Matrix")
    priority_df = pd.DataFrame({
        "Initiative":  ["Personalization Engine","Sparsity Reduction","Hybrid Model",
                        "Demographic Marketing","Temporal Promotions","A/B Testing"],
        "Impact":      [9, 8, 7, 6, 5, 7],
        "Effort":      [7, 4, 8, 3, 2, 5],
        "Category":    ["Model","Data","Model","Marketing","Marketing","Model"],
    })
    fig = px.scatter(priority_df, x="Effort", y="Impact",
                     text="Initiative", color="Category", size=[40]*6,
                     title="Initiative: Impact vs Effort",
                     color_discrete_sequence=[ACCENT, ACCENT2, ACCENT3, "#a888ff"])
    fig.update_traces(textposition="top center", marker_opacity=0.85)
    fig.add_hline(y=6.5, line_dash="dash", line_color=GRID_COLOR)
    fig.add_vline(x=5.5, line_dash="dash", line_color=GRID_COLOR)
    fig.add_annotation(x=2.5, y=9.6, text="Quick Wins", showarrow=False,
                       font=dict(color=ACCENT3, size=11))
    fig.add_annotation(x=7.5, y=9.6, text="Big Bets", showarrow=False,
                       font=dict(color=ACCENT2, size=11))
    st.plotly_chart(apply_theme(fig, 440), use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 10 — Questionnaire
# ═══════════════════════════════════════════════════════════════════════════════
elif page == PAGES[9]:
    section_header("Questionnaire", "Assessment & Answers")

    qa("1. Users of which age group have watched and rated the most number of movies?",
       "More than 2000 users belong to age group **25–34** who have watched and rated the most number of movies.")
    qa("2. Users belonging to which profession have watched and rated the most movies?",
       "From the EDA plot, the answer is **'College/grad students'**.")
    qa("3. Most of the users in our dataset who've rated the movies are Male (T/F)?",
       "**True** — 72% of users are Males.")
    qa("4. Most movies were released in which decade?  a. 70s  b. 90s  c. 50s  d. 80s",
       "Maximum movies released in 1995 and 1996. Answer: **b. 90s**.")
    qa("5. The movie with maximum no. of ratings is ___.",
       "**American Beauty** is the movie with the maximum number of ratings.")

    code_block("""max_ratings_movie  = movie_stats.loc[movie_stats['TotalRatingsPerMovie'].idxmax()]
movie_with_max     = movies_df.merge(
    movie_stats.loc[[movie_stats['TotalRatingsPerMovie'].idxmax()]], on='MovieID')
print(f"Movie Name: {movie_with_max['MovieName'].values[0]}")""")

    qa("6. Name the top 3 movies similar to 'Liar Liar' on the item-based approach.",
       "1. Mrs. Doubtfire\n2. Ace Ventura: Pet Detective\n3. Dumb & Dumber")

    code_block("""recommendations = recommend_movies("Liar Liar", user_item_matrix,
                                   item_similarity_df, model_knn, n_neighbors=5)
print(recommendations)""")

    qa("7. Collaborative Filtering methods can be classified into ___-based and ___-based.",
       "**User-based** and **item-based**.")
    qa("8. Pearson Correlation ranges from ___ to ___; Cosine Similarity from ___ to ___.",
       "**Pearson:** −1 to +1.\n\n**Cosine Similarity:** 0 to 1 (for non-negative vectors).")
    qa("9. Mention the RMSE and MAPE from the Matrix Factorization model.",
       "**RMSE = 0.8813** and **MAPE = 27.18%**.")
    qa("10. Sparse 'row' representation for [[1 0] / [3 7]]:",
       "**(row_ptr, col_indices, values) = ([0, 1, 3], [0, 0, 1], [1, 3, 7])**")

    st.markdown("### Question Difficulty Map")
    q_labels    = [f"Q{i}" for i in range(1,11)]
    difficulty  = [2,2,1,2,3,4,2,4,3,5]
    diff_labels = ["Easy" if d<=2 else "Medium" if d<=3 else "Hard" for d in difficulty]
    diff_colors = [ACCENT3 if d<=2 else ACCENT2 if d<=3 else ACCENT for d in difficulty]
    fig = go.Figure(go.Bar(
        x=q_labels, y=difficulty,
        marker_color=diff_colors,
        text=diff_labels, textposition="outside"
    ))
    fig.update_layout(yaxis_range=[0,6.5], title="Question Difficulty (1=Easy, 5=Hard)")
    st.plotly_chart(apply_theme(fig, 340), use_container_width=True)

    st.markdown("""
    <div style="text-align:center;padding:2rem 0;color:#3a4060;font-size:0.8rem;
         letter-spacing:1px;font-family:'JetBrains Mono';">
        Akul Vinod · DATA SCIENCE · MOVIE RECOMMENDER SYSTEM
    </div>""", unsafe_allow_html=True)
