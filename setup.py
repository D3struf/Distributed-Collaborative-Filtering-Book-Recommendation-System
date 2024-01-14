from setuptools import setup

with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()
    
REPO_NAME = "Distributed Collaborative Filtering Book Recommendation System"
AUTHOR_USER_NAME = 'd3struf'
SRC_REPO = 'src'
LIST_OF_REQUIREMENTS = ['streamlit', 'numpy', 'scikit-learn', 'pyspark', 'pyspark[sql]']

setup (
    name = SRC_REPO,
    version = '0.0.1',
    author = AUTHOR_USER_NAME,
    description = 'A small package for Movie Recommender System',
    long_description = long_description,
    long_description_content_type = 'text/markdown',
    url = f"https://github.com/{AUTHOR_USER_NAME}/{REPO_NAME}",
    author_email = "johnpaul.monter@tup.edu.ph",
    packages = [SRC_REPO],
    license = "MIT",
    python_requires = ">=3.7",
    install_requires = LIST_OF_REQUIREMENTS
)