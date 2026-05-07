import streamlit as st
import pandas as pd
import pickle
import numpy as np
import requests
from sklearn.metrics.pairwise import linear_kernel

API_KEY = "7ba158b7018d8a861d327e0e314ef65b"

def fetch_poster(title):
    url = f"https://api.themoviedb.org/3/search/movie?api_key={API_KEY}&query={str(title).replace(' ', '+')}"
    try:
        data = requests.get(url).json()
        if data.get('results') and data['results'][0].get('poster_path'):
            return f"https://image.tmdb.org/t/p/w500/{data['results'][0]['poster_path']}"
        return None
    except:
        return None

st.set_page_config(page_title="Movie Matcher", layout="wide")
st.markdown("<h1 style='text-align: center; color: #E50914;'>🎬 Movie Matcher</h1>", unsafe_allow_html=True)
st.write("---") 

@st.cache_data
# def load_data():
#     return pickle.load(open('df.pkl', 'rb')), pickle.load(open('indices.pkl', 'rb')), pickle.load(open('tfidfMatrix.pkl', 'rb'))
def load_data():
    # Use pandas to read the compressed file
    df = pd.read_pickle('df.pkl.gz')
    indices = pickle.load(open('indices.pkl', 'rb'))
    tfidf_matrix = pickle.load(open('tfidfMatrix.pkl', 'rb'))
    return df, indices, tfidf_matrix

df, indices, tfidf_matrix = load_data()

st.subheader("What are we watching today?")
selected_movie = st.selectbox("Type or select a movie from the dropdown", df['title'].values)

if st.button('Get Recommendations'):
    with st.spinner("Populating the popcorn..."):
        
        idx = indices[selected_movie]
        if hasattr(idx, '__iter__'):
            idx = idx.iloc[0] if hasattr(idx, 'iloc') else idx[0]
            
        idx = int(idx)

        top_indices = linear_kernel(tfidf_matrix[idx], tfidf_matrix).flatten().argsort()[-6:-1][::-1] 
        
        st.success(f"Because you liked **{selected_movie}**, you might enjoy these:")
        
        cols = st.columns(5)
        for col, i in zip(cols, top_indices):
            title = df['title'].iloc[i]
            overview = df['overview'].iloc[i]
            
            if pd.isna(overview) or str(overview).strip() == "" or str(overview).lower() == "nan":
                overview = "Plot overview is currently not available for this movie."
            
            with col:
                st.markdown(f"**{title}**")
                
                poster_url = fetch_poster(title)
                
                if poster_url:
                    st.image(poster_url, use_container_width=True)
                else:
                    st.markdown(
                        "<div style='height: 225px; display: flex; align-items: center; justify-content: center; border: 1px solid #555; border-radius: 5px; text-align: center; padding: 10px;'>"
                        "<span>🚫<br>Poster Not Available</span>"
                        "</div>", 
                        unsafe_allow_html=True
                    )
                
                with st.expander("Read Overview"):
                    st.caption(overview)