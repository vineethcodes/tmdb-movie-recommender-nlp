import streamlit as st
import pickle
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# --- 1. PAGE CONFIG ---
st.set_page_config(page_title="MovieMatch AI", page_icon="🎬")

# --- 2. LOAD DATA ---
@st.cache_data
def load_data():
    # Loading the files you created in your Notebook
    df = pd.DataFrame(pickle.load(open('movie_list.pkl', 'rb')))
    tfidf_matrix = pickle.load(open('tfidf_matrix.pkl', 'rb'))
    return df, tfidf_matrix

df, tfidf_matrix = load_data()

# --- 3. UI HEADER ---
st.title("🎬 MovieMatch AI")
st.markdown("### Intelligent Recommendation System")
st.write("Enter a movie you like, and I'll find similar ones based on genres and descriptions.")

# --- 4. SEARCH INPUT ---
movie_input = st.text_input("Type a movie name (e.g., 'batman'):")

if st.button('Get Recommendations'):
    if movie_input:
        try:
            # 1. FIND THE MOVIE: Search for the title in our 25k records
            matches = df[df['title'].str.contains(movie_input, case=False, na=False)]
            
            if matches.empty:
                st.warning("⚠️ Movie not found. Please try another name!")
            else:
                # 2. SELECT BEST MATCH: Pick the most popular one if multiple exist
                idx = matches.sort_values(by='popularity', ascending=False).index[0]
                
                # 3. CALCULATE SIMILARITY: Compare this movie vector to all others
                target_vector = tfidf_matrix[idx]
                scores = cosine_similarity(target_vector, tfidf_matrix).flatten()
                
                # 4. GET TOP 5: Sort scores and pick the best 5 (excluding the movie itself)
                top_indices = np.argsort(scores)[-6:-1][::-1]
                recommendations = df.iloc[top_indices]

                # --- 5. DISPLAY RESULTS ---
                st.success(f"Because you liked: **{df.iloc[idx]['title']}**")
                st.divider()

                for _, row in recommendations.iterrows():
                    col1, col2 = st.columns([3, 1])
                    
                    with col1:
                        st.subheader(row['title'])
                        st.write(f"**Genres:** {row['genres']}")
                    
                    with col2:
                        # --- THE DATA INTEGRITY LOGIC ---
                        rating = row['vote_average']
                        
                        if rating == 0:
                            display_rating = "Rating Not Available"
                        else:
                            display_rating = f"⭐ {rating:.1f}/10"
                        
                        st.metric("Score", display_rating)
                    
                    st.divider()

        except Exception as e:
            st.error("Error processing request. Please try a more specific title.")
    else:
        st.info("Please enter a movie title first.")

