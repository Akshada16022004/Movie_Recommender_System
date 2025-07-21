import streamlit as st
import pandas as pd
import requests
import scipy.sparse
import time
from functools import lru_cache

TMDB_API_KEY = "8265bd1679663a7ea12ac168da84d2e8"
DEFAULT_POSTER_URL = "https://via.placeholder.com/500x750?text=No+Poster"  # Fallback image
REQUEST_TIMEOUT = 10  # seconds
MAX_RETRIES = 3

# 🔹 Fetch TMDb movie ID from name with retry logic
def get_movie_id(movie_name):
    search_url = f"https://api.themoviedb.org/3/search/movie"
    params = {
        "api_key": TMDB_API_KEY,
        "query": movie_name
    }
    
    for attempt in range(MAX_RETRIES):
        try:
            response = requests.get(search_url, params=params, timeout=REQUEST_TIMEOUT)
            response.raise_for_status()  # Raises HTTPError for bad responses
            break
        except (requests.exceptions.RequestException, requests.exceptions.HTTPError) as e:
            if attempt == MAX_RETRIES - 1:
                print(f"❌ Failed to fetch movie ID for '{movie_name}' after {MAX_RETRIES} attempts: {str(e)}")
                return None
            wait_time = 2 ** attempt  # Exponential backoff
            print(f"⚠️ Retry {attempt + 1}/{MAX_RETRIES} for '{movie_name}'. Waiting {wait_time}s...")
            time.sleep(wait_time)
    if response.status_code == 200:
        results = response.json().get("results")
        if results:
            return results[0]["id"]
    return None

@lru_cache(maxsize=128)
def fetch_poster(movie_name):
    movie_id = get_movie_id(movie_name)
    print(f"🔍 Movie ID for '{movie_name}': {movie_id}")
    if movie_id:
        movie_url = f"https://api.themoviedb.org/3/movie/{movie_id}"
        params = {"api_key": TMDB_API_KEY}
        
        for attempt in range(MAX_RETRIES):
            try:
                response = requests.get(movie_url, params=params, timeout=REQUEST_TIMEOUT)
                response.raise_for_status()
                break
            except Exception as e:
                print(f"❌ Poster fetch failed: {e}")
                return DEFAULT_POSTER_URL
        
        if response.status_code == 200:
            data = response.json()
            poster_path = data.get("poster_path")
            print(f"🖼️ Poster path: {poster_path}")
            if poster_path:
                return f"https://image.tmdb.org/t/p/w500{poster_path}"
    
    print(f"⚠️ No valid poster found for: {movie_name}")
    return DEFAULT_POSTER_URL


# 🔹 Load data
movies_df = pd.read_csv("imdb_top_250.csv")
print("DEBUG: movies_df columns:", movies_df.columns)  # Debug print to check columns
similarity = scipy.sparse.load_npz("similarity.npz")


# 🔹 Recommendation engine
def recommend(movie_name):
    if movie_name not in movies_df["Title"].values:
        return [], []

    movie_index = movies_df[movies_df["Title"] == movie_name].index[0]
    distances = similarity[movie_index].toarray().flatten()
    similar_movies = sorted(enumerate(distances), key=lambda x: x[1], reverse=True)[1:6]

    recommended_movies = []
    recommended_posters = []
    
    for idx, _ in similar_movies:
        movie_title = movies_df.iloc[idx]["Title"]
        print(f"🔍 Fetching poster for: {movie_title}")  # Debug line
        recommended_movies.append(movie_title)
        poster_url = fetch_poster(movie_title)
        print(f"🖼️ Poster URL: {poster_url}")  # Debug line
        recommended_posters.append(poster_url)
    
    return recommended_movies, recommended_posters

# 🔹 Streamlit UI
st.title("🎬 Movie Recommender System")

# 🧪 TEST BLOCK
import requests
from PIL import Image
from io import BytesIO

# Test poster fetch with SSL verification disabled
# test_url = fetch_poster("Inception")
# st.markdown(f"🔍 Poster URL: `{test_url}`")
# headers = {
#     "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"
# }

# try:
#     response = requests.get(test_url, headers=headers, timeout=10, verify=False)
#     response.raise_for_status()
#     img = Image.open(BytesIO(response.content))
#     st.image(img, caption="Test Poster: Inception", width=250)
# except Exception as e:
#     st.error(f"❌ Failed to load image manually: {e}")






selected_movie_name = st.selectbox("Select a movie:", movies_df ["Title"].values)

if st.button("Recommend"):
    recommended_names, recommended_posters = recommend(selected_movie_name)

    for name, poster_url in zip(recommended_names, recommended_posters):
        st.markdown(f"### 🎬 {name}")
        if poster_url.startswith("http"):
            st.image(poster_url, width=250)
        else:
            st.warning("⚠️ No poster available.")
        st.markdown("---")





