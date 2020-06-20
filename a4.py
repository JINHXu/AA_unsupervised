#!/usr/bin/env python3
""" Statistical Language Processing (SNLP), Assignment 4
    See <https://snlp2020.github.io/a4/> for detailed instructions.
    Course:      Statistical Language processing - SS2020
    Assignment:  a4
    Author(s):   Jinghua Xu
    Description: Unsupervised learning with sklearn, an experiment with dimensionality reduction with PCA and clustering using k-means.
    
    Honor Code:  I pledge that this program represents my own work.
"""
import time

import numpy as np

import gzip
import itertools

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA


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
    '''
    nameset = []

    # vectorize the texts
    vectorizer = CountVectorizer()
    vectorized_texts = vectorizer.fit_transform(texts).toarray()

    n = len(vectorized_texts[0])
    avg_vectors = np.empty((0, n), int)

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
        avg_vector = np.mean(vectors, axis=0)
        avg_vector = np.reshape(avg_vector, (1, avg_vector.size))
        avg_vectors = np.append(avg_vectors, avg_vector, axis=0)

    return nameset, avg_vectors
    '''
    nameset = []
    avg_vectors = []

    # vectorize the texts
    vectorizer = CountVectorizer()
    vectorized_texts = vectorizer.fit_transform(texts).toarray()

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
        avg_vector = np.mean(vectors, axis=0)
        avg_vectors.append(avg_vector)

    avg_vectors = np.array(avg_vectors)

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
    # stores the n most similar usernames to name, in tuple(name, score)
    similar_n = []
    dissimilar_n = []

    cosine_similarities = []

    # the vector of name
    name_vector = vectors[names.index(name)]

    for vector in vectors:
        # array.reshape(1, -1) if it contains a single sample
        name_vector = name_vector.reshape(1, -1)
        vector = vector.reshape(1, -1)
        cosine_similarities.append(
            cosine_similarity(name_vector, vector)[0][0])
    # test
    # print(cosine_similarities)

    # prepare the copies as we remove from the list for max and min n times
    cs_cp4max = cosine_similarities.copy()
    cs_cp4min = cosine_similarities.copy()

    # paralel names lists
    names_cp4max = names.copy()
    names_cp4min = names.copy()

    # most similar n and most dissimilar n
    for _ in range(n):
        # find, store and remove max
        maxi = max(cs_cp4max)
        user = names_cp4max[cs_cp4max.index(maxi)]
        similar_n.append((user, maxi))
        cs_cp4max.remove(maxi)
        names_cp4max.remove(user)

        # find, store and remove min
        mini = min(cs_cp4min)
        user = names_cp4min[cs_cp4min.index(mini)]
        dissimilar_n.append((user, mini))
        cs_cp4min.remove(mini)
        names_cp4min.remove(user)

    # print to screen
    print(f'Most similar {n} users to {name}:')
    for idx in range(n):
        print(
            f'name: {similar_n[idx][0]}, score(cosine similarity): {similar_n[idx][1]}')

    print(f'Most dissimilar {n} users to {name}:')
    for idx in range(n):
        print(
            f'name: {dissimilar_n[idx][0]}, score(cosine similarity): {dissimilar_n[idx][1]}')


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
    n_components = 1

    n_samples = len(vectors)
    n_features = len(vectors[0])
    max_components = min(n_samples, n_features)

    # while True:
    while n_components < max_components:
        pca = PCA(n_components=n_components)
        pca.fit(vectors)

        if sum(pca.explained_variance_) > explained_var:
            return pca.transform(vectors)

        n_components += 1

    # in the rare case when the demensions of the vectors cannot be reduced by PCA with our setting(explained_var, svd_solver='full')
    # this may happen when explained_var is set overly large

    # or raise an error or throw an exception?

    print("Failed in PCA with our setting!")
    return vectors


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

    '''start_time = time.time()'''

    # 4.2
    nameset, vectors = encode_people(usernames, texts)

    # tests
    '''
    print(nameset)
    print(vectors)
    print(len(nameset))
    print(len(vectors))
    print(type(vectors))
    print(type(vectors[0]))
    print("--- %s seconds ---" % (time.time() - start_time))
    '''

    # 4.3
    # most_similar('Lagarde', nameset, vectors)
    '''print(vectors.shape)'''

    # 4.4
    reduced_vectors = reduce_dimensions(vectors)
    '''
    print(reduced_vectors)
    print(reduced_vectors.shape)
    '''
