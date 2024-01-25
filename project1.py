import streamlit as st
import numpy as np
import pandas as pd
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import ast

# Load the datasets
movies = pd.read_csv('C:/Users/HP/OneDrive/Desktop/python/tmdb_5000_movies.csv')
credits = pd.read_csv('C:/Users/HP/OneDrive/Desktop/python/tmdb_5000_credits.csv')

# Merge datasets on the 'title' column
movies = movies.merge(credits, on='title')

# Select relevant columns
movies = movies[['id', 'title', 'overview', 'genres', 'keywords', 'cast', 'crew']]

# Drop rows with missing values
movies.dropna(inplace=True)

# Function to convert strings with dictionary-like structures to lists of names
def convert(obj):
    L = []
    for i in ast.literal_eval(obj):
        L.append(i['name'])
    return L

# Function to convert the first 3 elements of a list of dictionary-like structures to a list of names
def convert3(obj):
    L = []
    counter = 0
    for i in ast.literal_eval(obj):
        if counter != 3:
            L.append(i['name'])
            counter += 1
        else:
            break
    return L

# Function to fetch the director from a list of dictionary-like structures
def fetch_director(obj):
    L = []
    for i in ast.literal_eval(obj):
        if i['job'] == 'Director':
            L.append(i['name'])
            break
    return L

# Applying functions to the respective columns
movies['genres'] = movies['genres'].apply(convert)
movies['keywords'] = movies['keywords'].apply(convert)
movies['cast'] = movies['cast'].apply(convert3)
movies['crew'] = movies['crew'].apply(fetch_director)

# Applying lambda functions to tokenize and clean strings in the specified columns
movies['overview'] = movies['overview'].apply(lambda x: x.split())
movies['genres'] = movies['genres'].apply(lambda x: [i.replace(" ", "") for i in x])
movies['keywords'] = movies['keywords'].apply(lambda x: [i.replace(" ", "") for i in x])
movies['cast'] = movies['cast'].apply(lambda x: [i.replace(" ", "") for i in x])
movies['crew'] = movies['crew'].apply(lambda x: [i.replace(" ", "") for i in x])

# Concatenate columns into a new 'tags' column
movies['tags'] = movies['overview'] + movies['genres'] + movies['keywords'] + movies['cast'] + movies['crew']

# Create a new DataFrame with selected columns
new_df = movies[['id', 'title', 'tags']]

# Apply join operation to concatenate list elements in the 'tags' column
new_df['tags'] = new_df['tags'].apply(lambda x: " ".join(x))
new_df['tags'] = new_df['tags'].apply(lambda x: x.lower())

# Stemming function
ps = PorterStemmer()
def stem(text):
    y = []
    for i in text.split():
        y.append(ps.stem(i.lower()))
    return " ".join(y)

new_df['tags'] = new_df['tags'].apply(stem)

# Vectorization using CountVectorizer
cv = CountVectorizer(max_features=5000, stop_words='english')
vectors = cv.fit_transform(new_df['tags']).toarray()

# Cosine similarity matrix
similarity = cosine_similarity(vectors)

# Streamlit App
st.title("Movie Recommendation System")

# Display the list of movies
st.sidebar.title("Movie List")
selected_movie = st.sidebar.selectbox("Select a Movie", new_df['title'])

# Display selected movie information
st.subheader("Selected Movie:")
st.write(new_df[new_df['title'] == selected_movie])

# Recommendation logic
movie_index = new_df[new_df['title'] == selected_movie].index[0]
distances = similarity[movie_index]
movies_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1:6]

# Display recommended movies
st.header("Recommended Movies:")
for i in movies_list:
    st.write(new_df.iloc[i[0]].title)
