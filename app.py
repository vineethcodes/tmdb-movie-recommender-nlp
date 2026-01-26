import streamlit as st
import pickle
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import difflib

# --- PAGE SETUP ---
st.set_page_config(page_title="Movie Matcher AI", page_icon="🎬")

# --- LOAD DATA ---
@st.cache_data # This makes the app super fast by keeping data in memory
def load_data():
    movies = pd.DataFrame(pickle.load(open('movie_list.pkl', 'rb')))
    tfidf_matrix = pickle.load(open('tfidf_matrix.pkl', 'rb'))
    return movies, tfidf_matrix

movies, tfidf_matrix = load_data()

# --- UI ELEMENTS ---
st.title("🎬 AI Movie Recommender")
st.markdown("Enter a movie you like, and I'll find similar ones based on story and genre.")

# Search box
user_input = st.text_input("Search for a movie (e.g., The Godfather, Iron Man):", "")

if st.button('Get Recommendations'):
    if user_input:
        # 1. Precise Fuzzy Matching
        all_titles = movies['title'].tolist()
        # cutoff=0.6 means it won't guess wildly; it needs to be 60% similar
        close_matches = difflib.get_close_matches(user_input, all_titles, n=1, cutoff=0.5)
        
        if close_matches:
            selected_movie = close_matches[0]
            st.success(f"Showing results for: **{selected_movie}**")
            
            # 2. Recommendation Logic
            idx = movies[movies['title'] == selected_movie].index[0]
            similarity_scores = cosine_similarity(tfidf_matrix[idx], tfidf_matrix).flatten()
            
            # Get top 6 (excluding the movie itself)
            similar_indices = similarity_scores.argsort()[-7:-1][::-1]
            
            # 3. Display Results
            st.divider()
            for i in similar_indices:
                movie_data = movies.iloc[i]
                col1, col2 = st.columns([3, 1])
                
                with col1:
                    st.subheader(f"{movie_data['title']}")
                    st.caption(f"Genre: {movie_data['genres']}")
                
                with col2:
                    # Professional formatting for ratings
                    st.metric("Rating", f"⭐ {movie_data['vote_average']}")
                st.divider()
        else:
            st.warning("No close match found. Please check your spelling and try again!")
    else:
        st.error("Please type a movie name first.")

