import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Sample dataset
movies = pd.DataFrame({
    'title': ['Inception', 'Interstellar', 'The Dark Knight', 'Avengers', 'Titanic'],
    'genres': ['Sci-Fi Thriller', 'Sci-Fi Drama', 'Action Crime', 'Action Fantasy', 'Romance Drama']
})

# Convert text to vectors
vectorizer = CountVectorizer()
genre_matrix = vectorizer.fit_transform(movies['genres'])

# Calculate similarity
cosine_sim = cosine_similarity(genre_matrix, genre_matrix)

# Recommendation function
def recommend_movie(title):
    if title not in movies['title'].values:
        return "Movie not found"
    
    idx = movies[movies['title'] == title].index[0]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:3]
    movie_indices = [i[0] for i in sim_scores]
    
    return movies['title'].iloc[movie_indices]

print("Recommended Movies:")
print(recommend_movie('Inception'))