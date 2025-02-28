import numpy as np
import pandas as pd
from flask import Flask, render_template, request
import requests
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__, static_url_path='/static', static_folder='static')
# Load the dataset
data = pd.read_csv('main_data.csv')

# TMDb API key
API_KEY = "3ee83e8a1334118dcde20c0d6634ffe7"

# Preprocess text for cosine similarity
def preprocess_text(text):
    # Implement any text preprocessing steps here
    return text.lower()

# Apply text preprocessing to 'comb' column
data['preprocessed_comb'] = data['comb'].apply(preprocess_text)

# Initialize CountVectorizer to convert text data into vectors
count_vectorizer = CountVectorizer(stop_words='english')
plot_matrix = count_vectorizer.fit_transform(data['preprocessed_comb'])

# Calculate cosine similarity matrix for movie plots
cosine_sim_plots = cosine_similarity(plot_matrix, plot_matrix)

def rcmd(movie_title):
    movie_title = movie_title.lower()
    if movie_title not in data['movie_title'].str.lower().values:
        return 'Sorry! The movie you requested is not in our database. Please check the spelling or try with some other movies'
    else:
        # Get the index of the movie in the dataset
        idx = data[data['movie_title'].str.lower() == movie_title].index[0]
        # Get the pairwise similarity scores with other movies
        sim_scores = list(enumerate(cosine_sim_plots[idx]))
        # Sort the movies based on similarity scores
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        # Get the top 10 most similar movies
        sim_scores = sim_scores[1:11]
        # Get the movie indices
        movie_indices = [i[0] for i in sim_scores]
        # Return the movie titles of the top 10 similar movies
        return data['movie_title'].iloc[movie_indices].tolist()

def get_movie_poster(movie_title):
    url = f"https://api.themoviedb.org/3/search/movie?api_key={API_KEY}&query={movie_title}"
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        if data['results']:
            # Extract the poster path for the first movie in the search results
            poster_path = data['results'][0].get('poster_path')
            if poster_path:
                # Construct the full poster URL using the base URL provided by TMDb
                poster_url = f"https://image.tmdb.org/t/p/w500/{poster_path}"
                return poster_url
    return None

@app.route("/")
def home():
    return render_template('home.html')

@app.route("/recommend", methods=["GET", "POST"])
def recommend():
    if request.method == "POST":
        movie_title = request.form['movie_title']
        recommended_movies = rcmd(movie_title)
        searched_movie_poster = get_movie_poster(movie_title)
        movie_posters = []  # Collect posters here
        for movie in recommended_movies:
            # Retrieve the poster URL for each movie from TMDb API
            poster_url = get_movie_poster(movie)
            movie_posters.append((movie, poster_url))
        return render_template('recommend.html', recommended_movies=movie_posters, searched_movie=movie_title, searched_movie_poster=searched_movie_poster)
    else:
        return render_template('recommend.html', recommended_movies=None)

# Genre-based recommendations route
@app.route("/recmd", methods=["GET", "POST"])
def recmd():
    if request.method == "POST":
        movie_title = request.form['movie_title']
        recommended_movies = rcmd(movie_title)
        movie_posters = []  # Collect posters here
        for movie in recommended_movies:
            # Retrieve the poster URL for each movie from TMDb API
            poster_url = get_movie_poster(movie)
            movie_posters.append((movie, poster_url))
        # If this POST request does not use genre, ensure genre is None or similar handling
        return render_template('rcmdgenres.html', recommended_movies=movie_posters, genre=None)
    elif request.method == "GET" and 'genre' in request.args:
        genre = request.args['genre']
        recommended_movies = rcmd_genre(genre)
        movie_posters = []  # Collect posters here
        for movie in recommended_movies:
            # Retrieve the poster URL for each movie from TMDb API
            poster_url = get_movie_poster(movie)
            movie_posters.append((movie, poster_url))
        # Pass 'genre' to the template to use in the <h1> tag
        return render_template('recmdgenres.html', recommended_movies=movie_posters, genre=genre.title())
    else:
        # Handle case when no genre or movie is specified
        return render_template('recmdgenres.html', recommended_movies=None, genre=None)

def rcmd_genre(genre):
    genre = genre.lower()
    # Example filtering mechanism - adjust according to your DataFrame structure
    recommended_movies = data[data['genres'].str.lower().str.contains(genre)]['movie_title'].tolist()
    return recommended_movies

if __name__ == '__main__':
    app.run(debug=True)




#Access key:AKIAWPPO6N3GAJ2BONU3
#secret key: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