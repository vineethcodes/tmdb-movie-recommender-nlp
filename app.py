import streamlit as st
import pickle
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import difflib 

# Load data
movies = pd.DataFrame(pickle.load(open('movie_list.pkl', 'rb')))
tfidf_matrix = pickle.load(open('tfidf_matrix.pkl', 'rb'))

st.title("🎬 Movie Recommender")

# 1. Get all movie titles as a list
list_of_all_titles = movies['title'].tolist()

# 2. Input box
movie_name = st.text_input('Type a movie name (it’s okay if you misspell it!):')

if st.button('Recommend'):
    # 3. Find the "Closest Match" to what the user typed
    find_close_match = difflib.get_close_matches(movie_name, list_of_all_titles)
    
    if not find_close_match:
        st.error("Sorry! We couldn't find a match. Try checking your spelling.")
    else:
        close_match = find_close_match[0]
        st.success(f"Finding recommendations for: **{close_match}**")
        
        # 4. Normal recommendation logic using the "Corrected" title
        index_of_the_movie = movies[movies.title == close_match].index[0]
        similarity_score = list(enumerate(cosine_similarity(tfidf_matrix[index_of_the_movie], tfidf_matrix).flatten()))
        sorted_similar_movies = sorted(similarity_score, key=lambda x: x[1], reverse=True)[1:7]

        for i, movie in enumerate(sorted_similar_movies):
            index = movie[0]
            title_from_index = movies[movies.index==index]['title'].values[0]
            st.write(f"{i+1}. {title_from_index}")

