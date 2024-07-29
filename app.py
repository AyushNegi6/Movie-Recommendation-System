import pandas as pd
import difflib
import requests
from flask import Flask, request, render_template
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

# OMDB API key
OMDB_API_KEY = "431eab79"

def fetch_poster_url(movie_title, api_key):
    base_url = "http://www.omdbapi.com/"
    params = {
        "apikey": api_key,
        "t": movie_title
    }
    response = requests.get(base_url, params=params)
    if response.status_code == 200:
        data = response.json()
        poster_url = data.get("Poster")
        if poster_url != "N/A":
            return poster_url
    return None

# Load and preprocess data
movies_data = pd.read_csv("./web/movies.csv")
selected_features = ["genres", "keywords", "tagline", "cast", "director"]
for feature in selected_features:
    movies_data[feature] = movies_data[feature].fillna("")

combined_features = movies_data["genres"] + " " + movies_data["keywords"] + " " + movies_data["tagline"] + " " + movies_data["cast"] + " " + movies_data["director"]
vectorizer = TfidfVectorizer()
feature_vectors = vectorizer.fit_transform(combined_features)
similarity = cosine_similarity(feature_vectors)

def recommend_movies(movie_name):
    list_all_movies = movies_data["title"].tolist()
    find_close_match = difflib.get_close_matches(movie_name, list_all_movies)
    if not find_close_match:
        return [("No matching movies found.", None)]
    close_match = find_close_match[0]
    index_of_movie = movies_data[movies_data.title == close_match]['index'].values[0]
    similarity_score = list(enumerate(similarity[index_of_movie]))
    sorted_similar_movies = sorted(similarity_score, key=lambda x: x[1], reverse=True)
    recommended_movies = []
    for i, movie in enumerate(sorted_similar_movies):
        if i > 0 and i < 13:
            index = movie[0]
            title_movie = movies_data[movies_data.index == index]['title'].values[0]
            recommended_movies.append((title_movie, fetch_poster_url(title_movie, OMDB_API_KEY)))
    return recommended_movies

@app.route("/", methods=["GET", "POST"])
def index():
    recommendations = []
    movie_name = ""
    if request.method == "POST":
        movie_name = request.form["movie_name"]
        recommendations = recommend_movies(movie_name)
    return render_template("index.html", movie_name=movie_name, recommendations=recommendations)

if __name__ == "__main__":
    app.run(debug=True)
