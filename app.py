import streamlit as st
import pickle
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# --- 1. SETUP ---
st.set_page_config(page_title="Movie Matcher", page_icon="🎬")

# --- 2. LOAD DATA ---
@st.cache_data
def load_data():
    df = pd.DataFrame(pickle.load(open('movie_list.pkl', 'rb')))
    tfidf_matrix = pickle.load(open('tfidf_matrix.pkl', 'rb'))
    return df, tfidf_matrix

df, tfidf_matrix = load_data()

# --- 3. UI ---
st.title("🎬 AI Movie Recommender")

movie_input = st.text_input("Type a movie name (e.g., 'batman'):")

if st.button('Recommend'):
    if movie_input:
        # --- YOUR EXACT FUNCTION LOGIC ---
        try:
            # 1. Fuzzy Search: Look for movies that contain the user's text
            matches = df[df['title'].str.contains(movie_input, case=False, na=False)]
            
            if matches.empty:
                st.warning("⚠️ Movie not found. Please try a different name!")
            else:
                # 2. Pick the most popular (Safety check: uses index 0 if popularity isn't in file)
                if 'popularity' in matches.columns:
                    idx = matches.sort_values(by='popularity', ascending=False).index[0]
                else:
                    idx = matches.index[0]
                
                # 3. Calculate similarity for just this one movie
                target_vector = tfidf_matrix[idx]
                scores = cosine_similarity(target_vector, tfidf_matrix).flatten()
                
                # 4. Get the Top 5 most similar (excluding itself)
                top_indices = np.argsort(scores)[-6:-1][::-1]
                recommendations = df.iloc[top_indices]

                # --- DISPLAY ---
                st.success(f"Results for: **{df.iloc[idx]['title']}**")
                st.divider()

                for _, row in recommendations.iterrows():
                    col1, col2 = st.columns([3, 1])
                    with col1:
                        st.subheader(row['title'])
                        st.write(f"**Genres:** {row['genres']}")
                    with col2:
                        # Professional star rating display
                        st.metric("Rating", f"⭐ {row['vote_average']}")
                    st.divider()

        except Exception as e:
            st.error("An error occurred. Please try a more specific title.")
    else:
        st.info("Please enter a movie title to see recommendations.")

