#!/usr/bin/env python3
""" Statistical Language Processing (SNLP), Assignment 4
    See <https://snlp2020.github.io/a4/> for detailed instructions.
    Course:      Statistical Language processing - SS2020
    Assignment:  a4
    Author(s):   Jinghua Xu
    Description: Unsupervised learning with sklearn, an experiment with dimensionality reduction with PCA and clustering using k-means.
    
    Honor Code:  I pledge that this program represents my own work.
"""

import numpy as np
import gzip

import itertools
from sklearn.feature_extraction.text import CountVectorizer


def read_data(fname):
    """ Read the tab-separated data.

    Parameters:
    -----------
    filename:  The name of the file to read.

    Returns: (a tuple)
    -----------
    names:      Screen names, a sequence texts with (repeated) screen
                names in the input file. The length of the sequence
                should be equal to the number of instances (Tweets) in
                the data.
    texts:      The corresponding texts (a sequence, e.g., a list)
    """

    names, texts = list(), list()
    with gzip.open(fname, 'rt') as fp:
        header = next(fp)
        for line in fp:
            data = line.split('\t', 1)

            if len(data) == 1:
                texts[-1] += data[0]
                continue

            names.append(data[0])
            texts.append(data[1])

    names = np.array(names)
    texts = np.array(texts)

    return names, texts


def encode_people(names, texts):
    """ Encode each person in the data as a (sparse) vector.

    The input to this function are the texts associated with screen
    names in the data. You are required to encode the texts using
    CountVectorizer (from sklearn), and take the average of all
    vectors belonging to the same person.

    You are free to use either sparse or dense matrices for the
    output.

    Parameters:
    -----------
    names       A sequence of length n_texts with (repeated) screen names
    texts       A sequence of texts

    Returns: (a tuple)
    -----------
    nameset:    (Unique) set of screen names 
    avg_vectors:    Corresponding average word-count vectors
    """
    nameset = []
    avg_vectors = []

    # vectorize the texts
    vectorizer = CountVectorizer()
    vectorized_texts = vectorizer.fit_transform(texts).todense()

    # dictionary stores the mapping from each name to its corpus
    name2vectors = dict()

    for (name, vector) in zip(names, vectorized_texts):
        if name not in name2vectors:
            name2vectors[name] = [vector]
        else:
            name2vectors[name].append(vector)

    for name, vectors in name2vectors.items():
        nameset.append(name)
        # average over these document vectors for each person
        vectors = np.array(vectors)
        avg_vector = np.mean(vectors, axis=0)
        avg_vectors.append(avg_vector)

    return nameset, avg_vectors


def most_similar(name, names, vectors, n=10):
    """ Print out most similar and most-dissimilar screen names for a screen name.

    Based on the vectors provided, print out most similar (according to
    cosine similarity) and most dissimilar 'n' people.

    Parameters:
    -----------
    name        The screen name for which we calculate the similarities
    names       The full set of names
    vectors     The vector representations corresponding to names
                (e.g., output of encode_people() above)
    n           Number of (dis)similar screen names to print

    Returns: None
    """


def reduce_dimensions(vectors, explained_var=0.99):
    """ Reduce the dimensionality with PCA (technique, not necessarily the implementation).

    Transform 'vectors' to a lower dimensional space with PCA and return
    the lower dimensional vectors. You can use any PCA implementation,
    e.g., PCA or TruncatedSVD from sklearn (scipy also has PCA/SVD
    implementations).

    The number of dimensions to return should be determined by the
    parameter explained_var, such that the total variance
    explained by the lower dimensional representation should be the
    minimum dimension that explains at least explained_var.

    Parameters:
    -----------
    vectors      Original high-dimensional (n_names, n_features) vectors
    explaind_var The amount of variance explained by the resulting
                 low-dimensional representation.

    Returns: 
    -----------
    lowdim_vectors  Vectors of shape (n_names, n_dims) where n_dims is
                    (much) lower than original n_features.
    """


def plot(names, vec, xi=0, yi=1, filename='plot-2d.pdf'):
    """ Plot the names on a 2D graph.

    This function should plot the screen names (the text) at
    coordinates of vec[i][xi] and vec[i][yi] where 'i' is the index of
    the screen name being plotted.

    Parameters:
    -----------
    names       Screen names
    vectors     Corresponding vectors (n_names, n_features)
    xi,yi       The dimensions to plot
    filename    The output file name

    Returns: None
    """


def cluster_kmeans(names, vectors, k=5):
    """ Cluster given data using k-means, print the resulting clusters of names.

    Parameters:
    -----------
    names       Screen names
    vectors     Corresponding vectors (n_names, n_dims)
    k           Number of clusters
    filename    The output file name

    Returns: None
    """


def plot_scree(vectors, max_k=20, filename="scree-plot.pdf"):
    """ Plot a scree plot of silhouette score of k-means clustering.

    This function should cluster the given data multiple times from k=2
    to k=max_k, calculate the silhouette score, and plot a scree plot
    to the indicated file name.

    Parameters:
    -----------
    vectors     The data coded as (dense) vectors.
    max_k       The maximum k to try.
    filename    The output file name

    Returns: None
    """


if __name__ == "__main__":
    # 4.1
    usernames, texts = read_data(
        '/Users/xujinghua/a4-jinhxu-and-beabea1234/a4-corpus-small.tsv.gz')

    # tests
    '''
    for i in range(0, 50000):
        print((usernames[i], texts[i]))

    print(usernames.size)
    print(texts.size)
    '''

    # 4.2
    nameset, vectors = encode_people(usernames, texts)

    # tests
    print(nameset)
    print(vectors)
