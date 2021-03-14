# [Clutering Tweets](https://snlp2020.github.io/a4/)
Experiment with unsupervised methods.

[Data Collection](https://github.com/JINHXu/TwitterCrawler)<br>
[smaller sample](a4-corpus-small.tsv.gz)
(in accordance with Twitter's developer policy).<br>
[Full data set available only in private repo](https://github.com/snlp2020/snlp).

Libraries:
[scikit-learn](https://scikit-learn.org), [numpy](https://numpy.org), [scipy](https://scipy.org/)


### 4.1 Read the data from `csv` file.
### 4.2 Create "person vectors" 
The `encode_people()` function takes the data read in, and return a (sparse) vector of per-document average word or n-gram counts for each person in the data set.
Used scikit-learn's `CountVectorizer` for counting the words per document.
You should write your own code to
average over these document vectors for each person.
For the purposes of this exercise,
simple bag-of-words features are sufficient.
However, you are recommended to experiment
with  higher level n-grams as well
(see below for more possible additional exercises).

This function returns a list of unique screen names,
and a numpy or scipy (sparse) array of shape `(n_names, n_words)`,
where n_words is the number of word types (or n-gram types)
in the whole corpus.

  
### 4.3 Similarities between people 

The function `most_similar()` printS top-n most similar and most dissimilar people
according to cosine similarity between the vectors representing them.

The output is a sorted list of most similar people with the
similarity score printed next to them in a row.
Similarly, output should contain another list with most dissimilar people,
again with the similarity metric printed next to them.

### 4.4 Dimensionality reduction (2p)

Reduce the dimensionality of the "person vectors" using PCA
(`reduce_dimensions()` in the template) with implementations from scikit-learn.


### 4.5 Plotting the low-dimensional data (1p)

The `plot()` function in the template plots the screen names of people on coordinates 
of two dimensions of the (dense) vectors indicated given as arguments,
and writes the output to an external file.

### 4.6 Clustering with k-means 
The function `cluster_kmeans()`, clusters the (dense) vectors obtained after dimensiionality reduction
into pre-specified number of clusters,
and prints out the screen names of the people in each cluster.

The output should look like (for 8 clusters)
```
0: f b c g e
1: a d
...
7: h x y
```
the numbers are the cluster numbers and letters are
the screen names.

### 4.7 Picking the number of clusters 

The function `plot_scree()` plots a scree plot of k-means clusters
from `k=2` to number of clusters specified by `max_k`.
The score plotted on the scree plots should be
the silhouette score of the predicted clusters.

Determine the number of clusters based on this graph

