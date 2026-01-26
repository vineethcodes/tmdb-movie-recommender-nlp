import streamlit as st
import pickle
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

# --- 1. LOAD DATA ---
movies = pd.DataFrame(pickle.load(open('movie_list.pkl', 'rb')))
tfidf_matrix = pickle.load(open('tfidf_matrix.pkl', 'rb'))

# --- 2. APP UI ---
st.set_page_config(page_title="Movie Matcher", page_icon="🎬")
st.title("🎬 Movie Recommender System")
st.write("Find movies similar to your favorites using AI.")

# Dropdown for user to select a movie
selected_movie = st.selectbox("Type or select a movie:", movies['title'].values)

# --- 3. RECOMMENDATION LOGIC ---
if st.button('Show Recommendations'):
    try:
        # Find index of the movie
        idx = movies[movies['title'] == selected_movie].index[0]
        
        # Calculate similarity
        vector = tfidf_matrix[idx]
        scores = cosine_similarity(vector, tfidf_matrix).flatten()
        
        # Get top 5
        distances = sorted(list(enumerate(scores)), reverse=True, key=lambda x: x[1])
        
        st.subheader(f"Because you liked {selected_movie}:")
        
        # Display results in columns
        for i in distances[1:6]:
            rec_movie = movies.iloc[i[0]]
            st.write(f"**{rec_movie['title']}** | {rec_movie['genres']} | ⭐ {rec_movie['vote_average']}")
            
    except Exception as e:
        st.error("Something went wrong. Please try another movie.")
