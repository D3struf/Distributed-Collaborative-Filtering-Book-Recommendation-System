# Distributed-Collaborative-Filtering-Book-Recommendation-System
It uses Dask as a Distributed Framework with Website Application using Streamlit. Inspired by the work of https://github.com/entbappy/ML-Based-Book-Recommender-System

 <img src="https://img.shields.io/badge/Made%20with-Jupyter-orange?style=for-the-badge&logo=Jupyter" width="240" alt="Made with Jupyter"> [![forthebadge made-with-python](http://ForTheBadge.com/images/badges/made-with-python.svg)](https://www.python.org/)

## Collaborative Recommendation System
 - Collaborative filtering systems rely on user-item interactions.

 - Users with similar ratings form clusters, facilitating the recommendation process.

 - When recommending books, the system employs a cluster-based mechanism.

- The system considers either ratings or comments as its sole parameter.

- In essence, collaborative filtering assumes that if one user likes item A and another user likes both item A and another item, B, the first user may also be interested in item B.

- Challenges include:

   - The computational expense of managing a user-item nXn matrix.

   - Preferential recommendation for only popular items.

   - Potential neglect of recommending new items.

## Data Used
We used the data from Kaggle that contains the Book names, User Ids and their Ratings. \
Link to data: [https://www.kaggle.com/datasets/arashnic/book-recommendation-dataset?resource=download&select=Ratings.csv](https://www.kaggle.com/datasets/arashnic/book-recommendation-dataset?resource=download&select=Ratings.csv)

## Algorithm Used
In training the Model, we used the KNN algorithm to cluster their ratings and in finding the suitable books to be recommended.

1. Load the dataset.

2. Set the value of k.

3. Iterate through the total number of training data points to obtain the predicted class.

4. Compute the Euclidean distance between the test data and each row of the training data, as it is a widely used distance metric.

5. Arrange the calculated distances in ascending order.

6. Extract the top k rows from the sorted array.

## Distributed Framework - Dask
Dask is a parallel computing library designed to seamlessly scale and handle larger-than-memory computations in a distributed environment.

1. Convert the Pandas DataFrame to a Dask DataFrame

2. Find the index of the target book in a distributed manner

3. Compute the distances and suggestions in a distributed manner

4. Schedule the computation and gather results

5. Append the Book list into the array

## Usage

### Clone the Repository 
```
git clone https://github.com/D3struf/Distributed-Collaborative-Filtering-Book-Recommendation-System.git
```
### Open Anaconda Command Prompt and Create a conda environment inside the repository's directory
```
conda create -n books python=3.7.10 -y
```
```
conda activate books
```
### Install the requirements
```
pip install -r requirements.txt
```
### Now run the app.py
```
streamlit run app.py
```
