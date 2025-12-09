import pandas as pd
import numpy as np
import re
import string
from typing import List, Optional, Tuple

STOP_WORDS = {
    "the","a","an","and","or","of","in","on","for","to","with","at","by","is",
    "it","this","that","from","as","are","be","was","were","but","not","so",
    "if","their","they","them","he","she","his","her"
}

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors


def clean_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(f"[{re.escape(string.punctuation)}]", " ", text)
    tokens = [t for t in text.split() if t not in STOP_WORDS]
    return " ".join(tokens)


class MovieRecommender:
    def __init__(self, csv_path: str):
        self.csv_path = csv_path
        self.movies: Optional[pd.DataFrame] = None
        self.vectorizer: Optional[TfidfVectorizer] = None
        self.feature_matrix = None
        self.nn_model: Optional[NearestNeighbors] = None

    def load_and_prepare(self):
        df = pd.read_csv(self.csv_path)

        # Keep only useful columns if they exist
        keep_cols = [
            "id",
            "title",
            "genre",
            "original_language",
            "overview",
            "popularity",
            "release_date",
            "vote_average",
            "vote_count",
        ]
        present_cols = [c for c in keep_cols if c in df.columns]
        df = df[present_cols]

        # Basic cleaning: drop rows without title
        df = df.dropna(subset=[c for c in ["title"] if c in df.columns]).reset_index(drop=True)

        # Fill missing text fields safely
        if "genre" in df.columns:
            df["genre"] = df["genre"].fillna("")
        else:
            df["genre"] = ""

        if "overview" in df.columns:
            df["overview"] = df["overview"].fillna("")
        else:
            df["overview"] = ""

        # Combine genre + overview into a single "document"
        df["combined_text"] = (df["genre"].astype(str) + " " + df["overview"].astype(str)).apply(clean_text)

        self.movies = df

    def build_vectorizer(self):
        # Safe defaults — can be tuned
        self.vectorizer = TfidfVectorizer(
            max_features=20000,
            ngram_range=(1, 2),
            min_df=2,
        )
        self.feature_matrix = self.vectorizer.fit_transform(self.movies["combined_text"])

    def build_nn_model(self):
        # cosine distance = 1 - cosine similarity
        self.nn_model = NearestNeighbors(
            metric="cosine", algorithm="brute", n_neighbors=21
        )
        self.nn_model.fit(self.feature_matrix)

    def fit(self):
        self.load_and_prepare()
        self.build_vectorizer()
        self.build_nn_model()

    def _get_index_from_title(self, title: str) -> Optional[int]:
        mask = self.movies["title"].astype(str).str.lower() == title.lower()
        matches = self.movies[mask]
        if matches.empty:
            return None
        return matches.index[0]

    def recommend_similar_by_title(self, title: str, top_n: int = 10) -> Tuple[Optional[pd.DataFrame], Optional[str]]:
        idx = self._get_index_from_title(title)
        if idx is None:
            return None, f"Movie '{title}' not found in the dataset."

        # Use +1 to include the movie itself and then skip it
        distances, indices = self.nn_model.kneighbors(
            self.feature_matrix[idx], n_neighbors=top_n + 1
        )
        indices = indices[0][1:]
        distances = distances[0][1:]

        results = self.movies.iloc[indices].copy()
        results["similarity"] = 1 - distances
        return results, None

    def recommend_from_favourites(self, titles: List[str], top_n: int = 10) -> Tuple[Optional[pd.DataFrame], Optional[str]]:
        # Collect indices for known titles
        indices = []
        for t in titles:
            idx = self._get_index_from_title(t)
            if idx is not None:
                indices.append(idx)

        if not indices:
            return None, "None of the selected movies were found."

        # Create a mean user profile vector (sparse matrix)
        user_profile = self.feature_matrix[indices].mean(axis=0)

        # Convert to dense array usable by kneighbors
        if hasattr(user_profile, "toarray"):
            user_profile = user_profile.toarray()
        else:
            user_profile = np.asarray(user_profile)

        # Ensure shape is (1, n_features)
        if user_profile.ndim == 1:
            user_profile = user_profile.reshape(1, -1)

        # Ask for more neighbors than top_n to allow removal of the favourites themselves
        distances, neighbour_idxs = self.nn_model.kneighbors(
            user_profile, n_neighbors=min(self.nn_model.n_neighbors, top_n + len(indices) + 5)
        )

        neighbour_idxs = neighbour_idxs[0]
        distances = distances[0]

        # Remove the favourite indices from the results
        filtered = []
        filtered_distances = []
        for nid, dist in zip(neighbour_idxs, distances):
            if nid not in indices:
                filtered.append(nid)
                filtered_distances.append(dist)
            if len(filtered) >= top_n:
                break

        results = self.movies.iloc[filtered].copy()
        results["similarity"] = 1 - np.array(filtered_distances)
        return results, None

    def get_genres(self):
        genres = set()
        for g in self.movies["genre"].dropna():
            for part in str(g).split(","):
                part = part.strip()
                if part:
                    genres.add(part)
        return sorted(genres)

    def filter_by_genres(self, selected_genres: List[str]):
        if not selected_genres:
            return self.movies

        def has_all(genres_str: str) -> bool:
            movie_genres = {g.strip().lower() for g in str(genres_str).split(",") if g.strip()}
            return all(g.lower() in movie_genres for g in selected_genres)

        mask = self.movies["genre"].apply(has_all)
        return self.movies[mask]
