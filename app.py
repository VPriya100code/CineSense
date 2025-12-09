import streamlit as st
import pandas as pd
from recommender import MovieRecommender

# ---------------------- PAGE CONFIG ---------------------- #
st.set_page_config(
    page_title="CineSense - Movie Recommender",
    page_icon="🎬",
    layout="wide",
)

# ---------------------- CUSTOM CSS ----------------------- #
st.markdown(
    """
    <style>

    /* MAIN APP BG */
    [data-testid="stAppViewContainer"] {
        background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
        font-family: 'Inter', sans-serif;
    }

    /* Prevent global recoloring of toolbar */
    [data-testid="stAppViewContainer"] * {
        color: #e8eaf6 !important;
    }

    /* Sidebar styling */
    section[data-testid="stSidebar"] {
        background: rgba(15, 23, 42, 0.60);
        backdrop-filter: blur(20px);
        border-right: 2px solid rgba(255, 255, 255, 0.1);
    }

    section[data-testid="stSidebar"] * {
        color: #e0e7ff !important;
    }

    /* Inputs */
    div[data-baseweb="select"] > div {
        background-color: rgba(255,255,255,0.07) !important;
        border-radius: 6px !important;
        color: white !important;
    }

    .stTextInput input {
        background-color: rgba(255,255,255,0.07) !important;
        color: white !important;
    }

    /* Title styling */
    .main-title {
        font-size: 3.2rem;
        font-weight: 900;
        margin-bottom: 0;
        display: flex;
        align-items: center;
        gap: 10px;
    }

    /* Subtitle */
    .subtitle {
        color: #e2e8f0 !important;
        opacity: 0.9;
    }

    /* Movie card */
    .movie-card {
        background: rgba(255, 255, 255, 0.07);
        border: 1px solid rgba(255, 255, 255, 0.15);
        border-radius: 18px;
        padding: 1.3rem;
        transition: 0.3s;
        backdrop-filter: blur(12px);
        box-shadow: 0 10px 30px rgba(0,0,0,0.25);
    }

    .movie-card:hover {
        transform: translateY(-6px) scale(1.02);
        box-shadow: 0 20px 40px rgba(0,0,0,0.45);
    }

    .movie-title {
        font-size: 1.3rem;
        font-weight: 700;
        color: #f8fafc !important;
    }

    .badge {
        display: inline-block;
        padding: 5px 11px;
        border-radius: 999px;
        background: rgba(96,165,250,0.22);
        border: 1px solid rgba(96,165,250,0.35);
        color: #93c5fd;
        font-size: 0.75rem;
        margin-right: 5px;
    }

    /* Footer */
    .footer {
        width: 100%;
        text-align: center;
        padding: 12px 0;
        color: #c7d2fe;
        font-size: 14px;
        margin-top: 40px;
    }

    /* Header Toolbar - Gradient match */
    [data-testid="stToolbar"] {
        background: linear-gradient(90deg, #0f0c29, #302b63, #24243e) !important;
        backdrop-filter: blur(8px);
        border-bottom: 1px solid rgba(255,255,255,0.15);
    }

    [data-testid="stToolbar"] * {
        color: #e8eaf6 !important;
        visibility: visible !important;
        opacity: 1 !important;
    }

    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------- LOAD MODEL ----------------------- #
@st.cache_resource(show_spinner=True)
def load_recommender():
    rec = MovieRecommender("movies.csv")
    rec.fit()
    return rec

recommender = load_recommender()
movies_df = recommender.movies

# ---------------------- NAVIGATION ----------------------- #
st.sidebar.title("📌 Navigation")
page = st.sidebar.radio("Go to:", ["🏠 Home", "🎬 Recommendations"])

# ---------------------- HEADER --------------------------- #
col1, col2 = st.columns([2.5, 1.3])

with col1:
    st.markdown(
    """
    <div class="main-title">
        🎬 <span style="background: linear-gradient(90deg, #60a5fa, #93c5fd);
                       -webkit-background-clip: text;
                       color: transparent;">CineSense</span>
    </div>
    """,
    unsafe_allow_html=True
)

    st.markdown(
        '<div class="subtitle">Your personalized movie recommendation engine powered by AI & content-based filtering.</div>',
        unsafe_allow_html=True
    )

with col2:
    st.metric("Movies Loaded", f"{len(movies_df):,}")

# ---------------------- HOME PAGE ------------------------ #
if page == "🏠 Home":
    st.markdown("""
    <div style='margin-top: 20px; padding: 40px; 
                background: rgba(255,255,255,0.05); 
                border-radius: 22px; 
                backdrop-filter: blur(12px);
                border: 1px solid rgba(255,255,255,0.08);'>

    <h1 style='font-size: 3rem; font-weight: 900;
               background: linear-gradient(90deg, #38bdf8, #c084fc);
               -webkit-background-clip: text; color: transparent;'>
        Welcome to CineSense
    </h1>

    <p style='color:#cbd5e1; font-size: 1.25rem; margin-top: -10px;'>
        Your intelligent AI-powered movie recommendation assistant.
    </p>

    <h2 style='color:#a5b4fc; margin-top: 30px;'>🚀 Features</h2>

    <ul style='color:#dbeafe; font-size: 1.1rem; line-height: 1.8;'>
        <li><b>🎯 Similar Movie Recommendations</b> – Find movies similar to any title.</li>
        <li><b>❤️ Recommendations Based on Favourites</b> – AI learns your taste profile.</li>
        <li><b>🎭 Genre-Based Filtering</b> – Explore movies by selected genres.</li>
        <li><b>🤖 Smart AI Engine</b> – TF-IDF + Cosine Similarity for content understanding.</li>
        <li><b>⚡ Fast Processing</b> – Search across 10,000+ movies instantly.</li>
    </ul>

    <h2 style='color:#a5b4fc; margin-top: 30px;'>✨ How It Works</h2>

    <ol style='color:#dbeafe; font-size: 1.1rem; line-height: 1.8;'>
        <li>Select <b>Recommendations</b> from the navigation menu.</li>
        <li>Choose your method: Similar Movie, Favourites, or Genres.</li>
        <li>Get personalized, AI-powered movie suggestions!</li>
    </ol>

    <p style='color:#94a3b8; margin-top: 25px; font-size: 1rem;'>
        Start your cinematic journey using the sidebar 🚀🍿
    </p>

    </div>
    """, unsafe_allow_html=True)
    st.markdown("""
    <div style="
        margin-top: 50px;
        padding: 16px;
        text-align: center;
        font-size: 0.95rem;
        color: #d1d5e1;
        background: rgba(255,255,255,0.04);
        border-top: 1px solid rgba(255,255,255,0.12);
        backdrop-filter: blur(6px);
        border-radius: 12px;
    ">
         Built with ❤️ by Priya<br> Built for movie lovers, by a movie lover.<br>© 2025 <b>CineSense</b>
    </div>
    """, unsafe_allow_html=True)


    st.stop()

# ---------------------- RECOMMENDATIONS PAGE ------------- #
if page == "🎬 Recommendations":

    # Sidebar Filters
    st.sidebar.title("🎛 Recommendation Controls")

    mode = st.sidebar.radio(
        "How do you want recommendations?",
        ["Similar to a movie", "Based on favourites", "Just by genres"],
    )

    all_titles = movies_df["title"].astype(str).sort_values().tolist()

    if mode == "Similar to a movie":
        selected_title = st.sidebar.selectbox("Choose a movie you like", all_titles)

    elif mode == "Based on favourites":
        selected_titles = st.sidebar.multiselect(
            "Pick a few of your favourite movies",
            all_titles
        )

    genres_available = recommender.get_genres()
    selected_genres = st.sidebar.multiselect(
        "Filter by genres (optional)",
        genres_available,
    )

    top_n = st.sidebar.slider("Number of recommendations", 5, 20, 10)

    # Main area
    st.subheader("Recommendations")

    # Helper: Genre filter
    def apply_genre_filter(df):
        if not selected_genres or df is None:
            return df

        def match(movie_genres):
            mg = {g.strip().lower() for g in str(movie_genres).split(",")}
            return all(g.lower() in mg for g in selected_genres)

        return df[df["genre"].apply(match)]

    # ---------- Recommendation Modes ----------
    if mode == "Similar to a movie":
        st.subheader(f"Because you liked: **{selected_title}**")

        results, err = recommender.recommend_similar_by_title(selected_title, top_n=top_n*2)
        if err:
            st.error(err)
        else:
            results = apply_genre_filter(results).head(top_n)
            cols = st.columns(2)

            for i, (_, row) in enumerate(results.iterrows()):
                with cols[i % 2]:
                    st.markdown(
                        f"""
                        <div class="movie-card">
                            <div class="movie-title">{row['title']}</div>
                            <div class="movie-meta">{str(row['release_date'])[:4]} • ⭐ {row['vote_average']}</div>
                            <div class="movie-meta"><b>Genres:</b> {row['genre']}</div>
                            <div style="margin:0.4rem 0;">
                                <span class="badge">Similarity: {row['similarity']:.2f}</span>
                                <span class="badge">Lang: {row['original_language']}</span>
                            </div>
                            <div class="movie-overview">{row['overview'][:280]}...</div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

    elif mode == "Based on favourites":
        st.subheader("Because of your favourite movies:")

        if not selected_titles:
            st.info("Choose at least one movie from favourites.")
        else:
            results, err = recommender.recommend_from_favourites(selected_titles, top_n=top_n*2)
            if err:
                st.error(err)
            else:
                results = apply_genre_filter(results).head(top_n)
                cols = st.columns(2)

                for i, (_, row) in enumerate(results.iterrows()):
                    with cols[i % 2]:
                        st.markdown(
                            f"""
                            <div class="movie-card">
                                <div class="movie-title">{row['title']}</div>
                                <div class="movie-meta">{str(row['release_date'])[:4]} • ⭐ {row['vote_average']}</div>
                                <div class="movie-meta"><b>Genres:</b> {row['genre']}</div>
                                <div style="margin:0.4rem 0;">
                                    <span class="badge">Similarity: {row['similarity']:.2f}</span>
                                    <span class="badge">Lang: {row['original_language']}</span>
                                </div>
                                <div class="movie-overview">{row['overview'][:280]}...</div>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )

    elif mode == "Just by genres":
        st.subheader("Genre-based Recommendations")

        results = recommender.filter_by_genres(selected_genres).head(top_n)
        cols = st.columns(2)

        for i, (_, row) in enumerate(results.iterrows()):
            with cols[i % 2]:
                st.markdown(
                    f"""
                    <div class="movie-card">
                        <div class="movie-title">{row['title']}</div>
                        <div class="movie-meta">{str(row['release_date'])[:4]} • ⭐ {row['vote_average']}</div>
                        <div class="movie-meta"><b>Genres:</b> {row['genre']}</div>
                        <div class="movie-overview">{row['overview'][:280]}...</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

# ---------------------- FOOTER --------------------------- #
st.markdown(
    """
    <div class="footer">
        Built with TF-IDF, cosine similarity & Streamlit • Crafted by Priya 🌟
    </div>
    """,
    unsafe_allow_html=True
)
