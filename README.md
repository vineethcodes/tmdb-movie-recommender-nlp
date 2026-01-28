# 🎬 MovieMatch: Content-Based Recommendation System
📝 Project Overview
I developed an end-to-end recommendation engine that analyzes a dataset of 25,000+ movies to suggest films based on story themes, genres, and "vibes." This project bridges the gap between raw data analysis in Kaggle and a live, user-facing product, allowing users to discover films using Natural Language Processing (NLP).

🔗 [View Live App Here](https://vineeth-movie-recommender.streamlit.app/)

# 🚀 Key Findings
NLP Vectorization: By combining movie overviews and genres into a single "Tags" column, I used TF-IDF to turn text into mathematical vectors. This allows the system to recognize that Interstellar and The Martian are similar beyond just being in the "Sci-Fi" category.

Smart Search Logic: I found that exact title matching leads to a poor user experience. I implemented a str.contains search logic that handles partial titles and case-sensitivity, making the search feel intuitive.

High Efficiency: By utilizing Cosine Similarity on a pre-calculated matrix, the system retrieves the top 5 matches from a library of 25,000+ titles in under 0.5 seconds.

# 🛠️ Tools Used
Python (Pandas, NumPy)

Machine Learning (Scikit-Learn, TF-IDF, Cosine Similarity)

Deployment (Streamlit Cloud, GitHub, VS Code)

# 💡 My Process & AI Oversight
In this project, I used AI to help optimize the math behind the Cosine Similarity matrix. However, I applied a "Human Filter" to the deployment phase—when local firewall restrictions prevented the app from running on my machine, I pivoted to a GitHub-to-Cloud workflow. This ensured the final product was a live, accessible URL rather than just static code on a hard drive.
# Feature Update
Smart Data Handling: Custom logic to identify and label unreleased or unrated titles ("Rating NA") to ensure data integrity and a clean user experience.
