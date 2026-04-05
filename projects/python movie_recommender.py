import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load dataset
movies = pd.read_csv("movies.csv")

# Replace NaN genres
movies['genres'] = movies['genres'].fillna('')

# Convert genres to vectors using TF-IDF (better for large data)
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(movies['genres'])

# Function to recommend based on category
def recommend_by_category(category, top_n=10):
    category_vector = tfidf.transform([category])
    similarity = cosine_similarity(category_vector, tfidf_matrix)
    
    similar_indices = similarity[0].argsort()[::-1][:top_n]
    
    return movies.iloc[similar_indices][['title', 'genres']]

# Example
print(recommend_by_category("Action", 5))