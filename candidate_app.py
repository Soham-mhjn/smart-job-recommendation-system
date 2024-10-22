# # [theme]
# # primaryColor="#87cefa"
# # backgroundColor="#f0f8ff"
# # secondaryBackgroundColor="#a0acf1"
# # textColor="#101010"



import pdfplumber
import streamlit as st
import pandas as pd
import numpy as np
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords 
from nltk.stem import WordNetLemmatizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.neighbors import NearestNeighbors
from geopy.geocoders import Nominatim
from geopy.extra.rate_limiter import RateLimiter
from sklearn.preprocessing import MinMaxScaler
import folium
from streamlit_folium import folium_static
import os
from string import punctuation

# Set up NLTK data path and downloads
custom_path = 'c:\\Users\\FX506LHB\\Desktop\\New folder\\Job-and-Talent-Recommendation-System\\.venv\\nltk_data'
if not os.path.exists(custom_path):
    os.makedirs(custom_path)
nltk.data.path.append(custom_path)
nltk.download('punkt', download_dir=custom_path)
nltk.download('stopwords', download_dir=custom_path)
nltk.download('wordnet', download_dir=custom_path)

# Ignore warning and set wide layout
st.set_page_config(layout="wide")
st.title('Job Recommendation System')

# File upload
cv = st.file_uploader('Upload your CV (PDF)', type='pdf')

# Career Level selection
levels = ["Entry Level", "Middle", "Senior", "Top", "Not Specified"]
career_level = st.multiselect('Career Level', levels, levels)

# Number of job recommendations
no_of_jobs = st.slider('Number of Job Recommendations:', min_value=20, max_value=100, step=10)

# OCR function to extract text from PDF
def extract_data(feed):
    text = ''
    with pdfplumber.open(feed) as pdf:
        for page in pdf.pages:
            text += page.extract_text()
    return text

# Load additional stopwords and location data
@st.cache_data
def get_location():
    with open('./districts.txt', 'r', errors='ignore') as f:
        return word_tokenize(f.read().replace("\n", " "))

@st.cache_data
def get_stopwords():
    with open('./stopwords.txt', 'r', errors='ignore') as f:
        return word_tokenize(f.read().replace("\n", " "))

locations = get_location()
stopwords_additional = get_stopwords()

# NLP processing function
@st.cache_data
def nlp(x):
    word_sent = word_tokenize(x.lower().replace("\n", ""))
    _stopwords = set(stopwords.words('english') + list(punctuation) + locations + stopwords_additional)
    word_sent = [word for word in word_sent if word not in _stopwords]
    lemmatizer = WordNetLemmatizer()
    nlp_processed_cv = [lemmatizer.lemmatize(word) for word in word_sent]
    return " ".join(nlp_processed_cv)

# Process the uploaded CV
if cv:
    cv_text = extract_data(cv)
    try:
        nlp_processed_cv = nlp(cv_text)
    except NameError:
        st.error('Error in processing CV. Please upload a valid PDF.')

    # DataFrame for processed CV
    df2 = pd.DataFrame()
    df2['All'] = [nlp_processed_cv]

    # Load job data
    @st.cache_data
    def get_job_data():
        return pd.read_csv('data_nlp.csv')

    df = get_job_data()

    # Combine columns into the 'All' field for job data
    df['All'] = (df['title'].fillna('') + ' at ' + df['company'].fillna('') +
                 '. Location: ' + df['location'].fillna('') +
                 '. Experience: ' + df['career level'].fillna(''))

    # Recommendation function
    @st.cache_data
    def get_recommendations(top, df, scores):
        recommendations = pd.DataFrame(columns=['JobID', 'title', 'career level', 'company', 'industry', 'salary', 'location', 'webpage', 'score'])
        for count, idx in enumerate(top):
            recommendations.loc[count] = {
                'JobID': df['JobID'][idx],
                'title': df['title'][idx],
                'career level': df['career level'][idx],
                'company': df['company'][idx],
                'industry': df['industry'][idx],
                'salary': df['salary'][idx],
                'location': df['location'][idx],
                'webpage': df['webpage'][idx],
                'score': scores[count]
            }
        return recommendations

    # TF-IDF function
    @st.cache_data
    def tfidf(scraped_data, cv):
        vectorizer = TfidfVectorizer(stop_words='english')
        tfidf_jobid = vectorizer.fit_transform(scraped_data)
        user_tfidf = vectorizer.transform(cv)
        return cosine_similarity(user_tfidf, tfidf_jobid)

    output2 = tfidf(df['All'], df2['All'])

    # Show top job recommendations using TF-IDF
    top = sorted(range(len(output2[0])), key=lambda i: output2[0][i], reverse=True)[:no_of_jobs]
    list_scores = [output2[0][i] for i in top]
    tfidf_recommendations = get_recommendations(top, df, list_scores)

    st.dataframe(tfidf_recommendations[['title', 'career level', 'company', 'industry', 'salary', 'location']])

    # Count Vectorizer function
    @st.cache_data
    def count_vectorize(scraped_data, cv):
        count_vectorizer = CountVectorizer(stop_words='english')
        count_jobid = count_vectorizer.fit_transform(scraped_data)
        user_count = count_vectorizer.transform(cv)
        return cosine_similarity(user_count, count_jobid)

    output3 = count_vectorize(df['All'], df2['All'])

    # Show top job recommendations using Count Vectorizer
    top = sorted(range(len(output3[0])), key=lambda i: output3[0][i], reverse=True)[:no_of_jobs]
    list_scores = [output3[0][i] for i in top]
    count_recommendations = get_recommendations(top, df, list_scores)

    # KNN function
    @st.cache_data
    def knn(scraped_data, cv):
        vectorizer = TfidfVectorizer(stop_words='english')
        job_tfidf = vectorizer.fit_transform(scraped_data)
        user_tfidf = vectorizer.transform(cv)
        knn_model = NearestNeighbors(n_neighbors=no_of_jobs, metric='cosine')
        knn_model.fit(job_tfidf)
        distances, indices = knn_model.kneighbors(user_tfidf)
        return indices[0], distances[0]

    knn_top_indices, knn_distances = knn(df['All'], df2['All'])
    knn_recommendations = get_recommendations(knn_top_indices, df, 1 - knn_distances)

    # Combine results from TF-IDF, Count Vectorizer, and KNN
    combined_recommendations = pd.concat([tfidf_recommendations, count_recommendations, knn_recommendations]).drop_duplicates(subset='JobID')

    # Normalize the scores and compute the final recommendation score
    scaler = MinMaxScaler()
    combined_recommendations[['score']] = scaler.fit_transform(combined_recommendations[['score']])
    combined_recommendations['Final Score'] = combined_recommendations['score'] / 3

    # Filter by career level
    filtered_jobs = combined_recommendations[combined_recommendations['career level'].isin(career_level)]

    # Display filtered recommendations
    st.dataframe(filtered_jobs[['title', 'career level', 'company', 'industry', 'salary', 'location']])

    # Map creation for job locations
    filtered_jobs['location'] = filtered_jobs['location'].fillna('India') + ', India'
    locator = Nominatim(user_agent="myGeocoder")
    geocode = RateLimiter(locator.geocode, min_delay_seconds=1)

    # Create a folium map
    folium_map = folium.Map(location=[20.5937, 78.9629], zoom_start=5)

    # Create a dictionary to count job occurrences by location
    location_counts = filtered_jobs['location'].value_counts().to_dict()

    for location, count in location_counts.items():
        loc_geo = geocode(location)
        if loc_geo:
            folium.CircleMarker(
                location=[loc_geo.latitude, loc_geo.longitude],
                radius=10,
                popup=f"{location}: {count} jobs",
                color='cadetblue',
                fill=True,
                fill_color='lightblue'
            ).add_to(folium_map)

    # Render the folium map
    folium_static(folium_map, width=1250)

    st.balloons()
