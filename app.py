import pickle
import streamlit as st
import numpy as np
import dask
import dask.array as da

st.header('Distributed Collaborative Filtering Books Recommendation System')
model = pickle.load(open('Exports/model.pkl', 'rb'))
books_name = pickle.load(open('Exports/book_names.pkl', 'rb'))
final_rating = pickle.load(open('Exports/final_rating.pkl', 'rb'))
book_pivot = pickle.load(open('Exports/book_pivot.pkl', 'rb'))

def fetch_poster(suggestion):
    book_name = []
    ids_index = []
    poster_url = []
    
    for book_id in suggestion:
        book_name.append(book_pivot.index[book_id])
    
    for name in book_name[0]:
        ids = np.where(final_rating['title'] == name)[0][0]
        ids_index.append(ids)
        
    for idx in ids_index:
        url = final_rating.iloc[idx]['image_url']
        poster_url.append(url)
        
    return poster_url

def recommend_books(book_name, book_pivot, model):
    book_list = []
    book_pivot_dask = da.from_array(book_pivot.values, chunks=(100, len(book_pivot.columns)))
    book_id = dask.delayed(np.where)(book_pivot.index == book_name)[0][0]
    
    @dask.delayed
    def compute_recommendations(book_id):
        distance, suggestion = model.kneighbors(book_pivot_dask[book_id, :].reshape(1, -1), n_neighbors=6)
        return suggestion
    
    suggestions = dask.compute(compute_recommendations(book_id))[0]
    
    poster_url = fetch_poster(suggestions)
    
    for i in range(len(suggestions)):
        books = book_pivot.index[suggestions[i]]
        for j in books:
            book_list.append(j)
            
    return book_list, poster_url


selected_books = st.selectbox (
    'Type or select a book',
    books_name
)

if st.button('Show Recommendation'):
    recommendation_books, poster_url = recommend_books(selected_books, book_pivot, model)
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.text(recommendation_books[1])
        st.image(poster_url[1])
    
    with col2:
        st.text(recommendation_books[2])
        st.image(poster_url[2])
        
    with col3:
        st.text(recommendation_books[3])
        st.image(poster_url[3])

    with col4:
        st.text(recommendation_books[4])
        st.image(poster_url[4])
        
    with col5:
        st.text(recommendation_books[5])
        st.image(poster_url[5])