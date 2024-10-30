import streamlit as st
import pandas as pd
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
from src.pipeline.recommend_pipeline import RecommendPipeline  # Ensure correct import

# Set up Spotify API credentials
sp = spotipy.Spotify(auth_manager=SpotifyClientCredentials(
    client_id="0c64bcdd0c2a4358bc87d3bcc131ead0",
    client_secret="21dbcb4e2819426f84774f107e3afd30"
))

@st.cache_data
def load_data():
    data_path = "data/taylor_swift_tracks_dataset.csv"
    df = pd.read_csv(data_path)
    return df

# Function to fetch album art URL from Spotify
def get_album_art_url(track_id):
    try:
        track_info = sp.track(track_id)
        album_art_url = track_info['album']['images'][0]['url']  # Get the largest album image
        return album_art_url
    except Exception:
        return None

# Apply custom CSS for dark background and white text
st.markdown(
    """
    <style>
        /* Set background color to black */
        .reportview-container {
            background-color: black;  /* This affects the whole app */
        }
        .stApp {
            background-color: black;  /* This affects the app's main area */
        }
        /* Text colors */
        h1, h2, h3, p {
            color: white;  /* Title and text color */
        }
        .custom-song-box {
            border: 2px solid white;  /* White border for the song box */
            border-radius: 8px;
            margin-bottom: 15px;
            color: white;  /* Text color inside the box */
        }
        .custom-song-content {
            background-color: black;  /* Box background color */
            border-radius: 8px;  /* Match the border-radius of the outer box */
            padding: 15px;  /* Padding inside the box */
        }
        .custom-song-content img {
            border-radius: 5px;  /* Optional: round corners for the album art */
        }
        .custom-button {
            background-color: white;
            color: black;
            border: none;
            border-radius: 8px;
            padding: 10px 20px;
            font-size: 16px;
            font-weight: bold;
            cursor: pointer;
            transition: background-color 0.3s ease, color 0.3s ease;
            margin-top: 10px;
        }
        .custom-button:hover {
            background-color: #e0e0e0;  /* Darker background on hover */
        }
    </style>
    """,
    unsafe_allow_html=True
)

# Load dataset
df = load_data()
recommend_pipeline = RecommendPipeline()

# Streamlit UI setup
st.title("Taylor Swift Song Recommender")
st.write("Choose a song from the dropdown to get recommendations based on your selection.")

# Searchable dropdown with album art
song_names = df['track_name'].unique()
selected_song = st.selectbox("Search and Select a Song:", song_names)

# Display album art beside dropdown
if selected_song:
    song_data = df[df['track_name'].str.lower() == selected_song.lower()]  # Case-insensitive match
    if not song_data.empty:
        song_data = song_data.iloc[0]
        album_name = song_data['album_name']
        album_art_url = get_album_art_url(song_data['track_id'])

        col1, col2 = st.columns([1, 4])
        
        with col1:
            if album_art_url:
                st.image(album_art_url, width=100)  # Adjust width for a smaller image
            else:
                st.write("Album art not available")

        with col2:
            st.write(f"Album: {album_name}")
            st.write(f"Track: {selected_song}")
    else:
        st.write("Song data not found in the dataset.")

# Custom HTML button to get recommendations
if st.markdown('<button class="custom-button" onclick="getRecommendations()">Get Recommendations</button>', unsafe_allow_html=True):
    recommendations = recommend_pipeline.recommend(song_name=selected_song)
    st.write("Recommended Songs:")

    for rec in recommendations:
        # Fetch each recommended song's album art
        rec_data = df[df['track_name'].str.lower() == rec['track_name'].lower()]  # Case-insensitive match

        if not rec_data.empty:
            rec_data = rec_data.iloc[0]
            rec_album_art_url = get_album_art_url(rec_data['track_id'])
            
            # Display each recommendation in a styled box with album art and details
            with st.container():
                st.markdown('<div class="custom-song-box">', unsafe_allow_html=True)  # Outer box with border
                st.markdown('<div class="custom-song-content">', unsafe_allow_html=True)  # Inner box for padding
                col1, col2 = st.columns([1, 4])
                
                with col1:
                    if rec_album_art_url:
                        st.image(rec_album_art_url, width=50)  # Adjust width for smaller thumbnails
                    else:
                        st.write("No album art")
                
                with col2:
                    st.write(f"**{rec['track_name']}**")
                    st.write(f"Album: {rec['album_name']}")
                    st.write(f"Mood: {rec['mood']} - Sentiment: {rec['sentiment']:.2f}")
                
                st.markdown('</div>', unsafe_allow_html=True)  # Close inner box
                st.markdown('</div>', unsafe_allow_html=True)  # Close outer box
        else:
            st.write(f"{rec['track_name']} - Song data not found")