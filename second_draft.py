# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.2'
#       jupytext_version: 1.0.5
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# ## Second draft of Contrast algorithm

# %% [markdown]
# ## To do
# - Inverse law when seeing duplicates
# - Implement context
# - Recollection: random updates vs complete updates?
#
# ## Not to do
#
# ## Done
# - ✔ Forgetting data
# - ✔ Compute standard_distances for each cluster
# - ✔ Clustering validity checking methods (Note: not necessarily relevant
#   though)

# %% [markdown]
# ## Imports

# %%
# import time
# import csv
import matplotlib.pyplot as plt
# import matplotlib.cm as cm
import numpy as np
import seaborn as sns
import pandas as pd
# from scipy.stats import chi2
from scipy.stats import special_ortho_group
# from scipy.spatial.distance import mahalanobis
# from sklearn import metrics
# from utils import plot_confusion_matrix
import collections
import functools
import itertools
# from sklearn.utils.multiclass import unique_labels
from sklearn.datasets import load_wine

np.random.seed(1)

# %% [markdown]
# ## Global variables and functions

# %%
# --- Information for generating dataset ---
NB_FEATURES = 5
NB_GROUPS = 5
N = 300
DOMAIN_LENGTH = 200
DEV_MAX = 20

COLORS = ['r', 'g', 'b', 'y', 'c', 'm']

# stocks metadata as (DATASET_NAME, DATASET_EXTENSION, DATASET_PATH,
#     \ DATASET_CLUSTER_COLUMN_INDEX, DATASET_DATA_COLUMNS_INDICES)
METADATA = {
    'Cards': ('Cards', '.csv', 'data/', 1, (2, None)),
    'Cards_truncated': ('Cards', '.csv', 'data/', 1, (2, 7))
}

SHOULD_LOAD_DATASET = 1  # 0 to generate, 1 to load csv, 2 to load sklearn

if SHOULD_LOAD_DATASET == 1:
    NAME = 'Cards'
    DATASET_NAME, DATASET_EXTENSION, DATASET_PATH, \
        DATASET_CLUSTER_COLUMN_INDEX, \
        DATASET_DATA_COLUMNS_INDICES = METADATA[NAME]
    DATASET_PATH_FULL = DATASET_PATH + DATASET_NAME + DATASET_EXTENSION
elif SHOULD_LOAD_DATASET == 0:
    DATASET_PATH = 'data/'
    DATASET_NAME = 'dummy'
elif SHOULD_LOAD_DATASET == 2:
    load_dataset = load_wine
    DATASET_PATH = 'data/'
    DATASET_NAME = 'wine'

PRINT_METRICS_DEFINED = False


def generate_cluster(n_cluster, nb_features=NB_FEATURES, d=DOMAIN_LENGTH,
                     dev_max=DEV_MAX, method='byhand'):
    """
    The 'method' argument can be one of the following:
    - 'byhand'
    - 'multinormal'
    """
    mean = np.random.random(nb_features) * d
    if method == 'multinormal':
        # /!\ does not work: data does not seem random at all, covariance is
        # always positive...!
        raise DeprecationWarning("beware, 'multinormal' method does not seem"
                                 "to work")
        cov = np.tril(np.random.random((nb_features, n_cluster)) *
                      np.random.random() * dev_max)
        cov = cov @ cov.transpose()  # a covariance matrix
        return(np.random.multivariate_normal(mean, cov, n_cluster))
    else:
        if method != 'byhand':
            print("generate_cluster: method unknown, using 'byhand'")
        st_devs = dev_max * np.random.random(nb_features)
        # holds the st_devs of each feature
        cluster_points = np.zeros((n_cluster, nb_features))
        for i in range(n_cluster):
            for j in range(nb_features):
                cluster_points[i][j] = np.random.normal(loc=mean[j],
                                                        scale=st_devs[j])
        cluster_points = cluster_points @ special_ortho_group.rvs(nb_features)
        return(cluster_points)


def generate_dataset(nb_groups=NB_GROUPS, n=N, nb_features=NB_FEATURES,
                     d=DOMAIN_LENGTH, dev_max=DEV_MAX):
    group_sizes = np.random.random(nb_groups)
    group_sizes *= n / np.sum(group_sizes)
    group_sizes = np.trim_zeros(np.round(group_sizes)).astype(int)
    data = [generate_cluster(n_cluster) for n_cluster in group_sizes]
    data = np.vstack(data)
    clusters_true = np.concatenate([n_cluster * [i] for i, n_cluster in
                                    enumerate(group_sizes)])
    np.save(DATASET_PATH + DATASET_NAME + '_data.npy', data)
    np.save(DATASET_PATH + DATASET_NAME + '_clusters_true.npy', clusters_true)


def shuffle(data_to_shuffle):
    new_permutation = np.random.permutation(len(data_to_shuffle))
    return(data_to_shuffle[new_permutation])


# %% [markdown]
# ## Generate or load dataset

# %%
if SHOULD_LOAD_DATASET == 0:
    generate_dataset()
    data = np.load(DATASET_PATH + DATASET_NAME + '_data.npy')
    clusters_true = np.load(DATASET_PATH + DATASET_NAME +
                            '_clusters_true.npy').astype(int)
elif SHOULD_LOAD_DATASET == 1:
    start, end = DATASET_DATA_COLUMNS_INDICES
    df1 = pd.read_csv(DATASET_PATH_FULL)
    df1_np = df1.to_numpy(copy=True)
    data = df1_np[:, start:end].astype('float')
    clusters_true = df1_np[:, DATASET_CLUSTER_COLUMN_INDEX]
elif SHOULD_LOAD_DATASET == 2:
    dataset = load_wine()
    data = dataset["data"]
    clusters_true = dataset["target"]


assert data is not None, 'data is None'
assert clusters_true is not None, 'clusters_true is None'

# print(data)
# print(clusters_true)

# %% [markdown]
# ## Preview dataset

# %%
columns = ["d" + str(i) for i in range(data.shape[1])] + ['true cluster']
df = pd.DataFrame(np.hstack((data, np.reshape([clusters_true],
                                              (data.shape[0], 1)))),
                  columns=columns)
true_data_plot = sns.pairplot(df, kind="scatter", hue='true cluster',
                              vars=columns[:-1])
true_data_plot.savefig(DATASET_PATH + DATASET_NAME + '_true.png')


# %% [markdown]
# ## The agent

# %%
class ContrastAgent(object):
    def __init__(self,
                 eps=0.01,
                 maxi=2.5,
                 memory_size=50,
                 mini=1,
                 nb_closest=8,
                 nb_winners=1,
                 update_method=2):

        self.deviations = []
        self.eps = eps
        self.maxi = maxi
        self.memory = []
        self.memory_size = memory_size
        self.mini = mini
        self.nb_closest = nb_closest
        self.nb_winners = nb_winners
        self.update_method = update_method
        self.weights = []

    def adjusted_threshold(self, x, maxi=2.5, mini=1, a=0.2, b=5):
        """ https://www.desmos.com/calculator/rydyha6kmb """
        return maxi - ((maxi - mini) / (1 + np.exp(-(a*(x-1) - b))))

    def cluster_battles(self, obj, indices, nb_winners=1):
        winner_indices = []
        while len(indices) > nb_winners:
            if len(indices) % 2 == 1:
                winner_indices.append(indices.pop(
                    np.random.randint(len(indices))))
            for i in range(0, len(indices), 2):
                dim_a, dim_b = 0, 0
                a = indices[i]
                b = indices[i+1]
                for j in range(len(obj)):
                    if np.abs(self.memory[a][j] - obj[j]) > \
                            np.abs(self.memory[b][j] - obj[j]):
                        dim_a += 1
                    else:
                        dim_b += 1
                if dim_a > dim_b:
                    winner_indices.append(a)
                else:
                    winner_indices.append(b)
            indices = winner_indices
            winner_indices = []
        return indices

    def feed_data_online(self, new_data):
        if len(self.memory) == 0:
            self.update_clusters(new_data[0])
        for obj in new_data[1:]:
            self.find_cluster(obj, update=True, nb_winners=self.nb_winners)

    def find_cluster(self, obj, update=True, nb_winners=1):
        n_closest_indices = self.find_n_closest_prototypes(obj,
                                                           self.nb_closest)
        winner_indices = self.cluster_battles(obj, n_closest_indices,
                                              nb_winners)
        if update:
            self.update_clusters(obj, winner_indices,
                                 method=self.update_method)
            return
        else:
            return winner_indices

    def find_n_closest_prototypes(self, obj, n=4):
        matching_dims = np.zeros(len(self.memory))
        for i in range(len(self.memory)):
            for j in range(len(obj)):
                if self.is_in_cluster_dimension(obj, i, j):
                    matching_dims[i] += 1
#         unique_dims = np.unique(matching_dims)
#         closest_indices = np.concatenate(
#             [np.flatnonzero(matching_dims == i) for i in unique_dims])
        max_dims = np.max(matching_dims)
#         print("max_dims: {}".format(max_dims))
        closest_indices = \
            np.flatnonzero(matching_dims == max_dims)
        # print("closest_indices: {}".format(closest_indices))
        if len(closest_indices) > n:
            # closest_indices = np.random.choice(closest_indices, 4,
            #                                    replace=False)
            closest_indices = closest_indices[-n:]
        return closest_indices.tolist()

    def forget_if_needed(self):
        while len(self.memory) > self.memory_size:
            self.memory.pop(0)
            self.weights.pop(0)
            self.deviations.pop(0)

    def is_in_cluster_dimension(self, obj, i, j,):
        return np.abs(obj[j] - self.memory[i][j]) <= self.deviations[i][j] * \
            self.adjusted_threshold(self.weights[i], self.maxi, self.mini)

    def update_clusters(self, obj, winner_indices=None, method=2):
        """
            method 1: weighted averaged standard deviations
            method 2: weighted averaged distances
        """
        if winner_indices is not None:

            for cluster_index in winner_indices:
                old_weight = self.weights.pop(cluster_index)
                old_deviations = self.deviations.pop(cluster_index)
                old_prototype = self.memory.pop(cluster_index)

                new_weight = old_weight + 1
                if method == 1:
                    temp = np.repeat(np.array([old_prototype, obj]),
                                     [old_weight, 1],
                                     axis=0)
                    deviation_with_obj = np.std(temp, axis=0)
                    new_deviations = (old_deviations * old_weight +
                                      deviation_with_obj) / new_weight
                else:
                    if method != 2:
                        print("update_clusters: "
                              "unknown method, using method 2")
                    new_deviations = (old_deviations * old_weight +
                                      np.abs(old_prototype - obj)) / new_weight

                new_prototype = (old_weight * old_prototype + obj) / new_weight

                self.weights.append(new_weight)
                self.deviations.append(new_deviations)
                self.memory.append(new_prototype)

        self.weights.append(1)
        self.deviations.append(np.abs(self.eps * obj))
        self.memory.append(obj)

        self.forget_if_needed()

    # --- Plot functions ---

    def print_confusion_matrix(self, data, clusters_true,
                               cmap=plt.cm.Blues,
                               cmap2=plt.cm.Reds,
                               method="closest",
                               nb_winners=4):
        """
            method can be either "all" or "closest"
        """
        nb_clusters_true = np.max(clusters_true) + 1
        cluster_true_sizes = [np.count_nonzero(clusters_true == i) for i in
                              range(nb_clusters_true)]
        cm = np.zeros((len(self.memory), nb_clusters_true))

        if method == "all":
            for obj_i, obj in enumerate(data):
                for i in range(len(self.memory)):
                        if all([self.is_in_cluster_dimension(obj, i, j) for j in
                                range(len(obj))]):
                            cm[i][clusters_true[obj_i]] += 1
        else:
            if method != "closest":
                print("print_confusion_matrix: unknown"
                      "is_in_cluster_method, using 'closest'")
            for obj_i, obj in enumerate(data):
                winner_indices = self.find_cluster(obj, update=False, nb_winners=nb_winners)
                for i in winner_indices:
                    cm[i][clusters_true[obj_i]] += 1

        cm /= cluster_true_sizes

        # print(cm)

        title = 'Normalized confusion matrix'
        fig, (ax1, ax2) = plt.subplots(
                              1, 2, sharey=True,
                              figsize=(4+nb_clusters_true, 4+len(self.memory)/5)
                              )
        im1 = ax1.imshow(cm, interpolation='nearest', cmap=cmap, aspect='auto')
        ax1.figure.colorbar(im1, ax=ax1)
        # We want to show all ticks...
        ax1.set(xticks=np.arange(cm.shape[1]),
                yticks=np.arange(cm.shape[0]),
                # ... and label them with the respective list entries
                xticklabels=np.arange(cm.shape[1]),
                yticklabels=np.arange(cm.shape[0]),
                title=title,
                ylabel='Predicted label',
                xlabel='True label')

        im2 = ax2.imshow(np.reshape(self.weights, (-1, 1)),
                         interpolation='nearest',
                         cmap=cmap2)
        ax2.figure.colorbar(im2, ax=ax2)
        ax2.set(xticks=np.arange(0),
                yticks=np.arange(len(self.weights)),
                yticklabels=np.arange(len(self.weights)),
                title="Weights")

        # Rotate the tick labels and set their alignment.
        # plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
        #          rotation_mode="anchor")

        # Loop over data dimensions and create text annotations.
        fmt = '.2f'
        fmt2 = 'd'
        thresh = cm.max() / 2.
        thresh2 = np.max(self.weights) / 2
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax1.text(j, i, format(cm[i, j], fmt),
                         ha="center", va="center",
                         color="white" if cm[i, j] > thresh else "black")
            ax2.text(0, i, format(self.weights[i], fmt2),
                     ha="center", va="center",
                     color="white" if self.weights[i] > thresh2 else "black")
        fig.tight_layout()
        plt.savefig(DATASET_PATH + DATASET_NAME + '_confusion_matrix.png')
        return fig


# %% [markdown]
# ## Data manipulation

# %%
new_data = np.copy(data)


def data_add(j, x):
    new_data[:, j] += x


def data_mult(j, x):
    new_data[:, j] *= x


def data_func(j, f):
    new_data[:, j] = f(new_data[:, j])


# %% [markdown]
# ## Clusterize

# %%
ca = ContrastAgent(eps=0.01,
                   maxi=2.5,
                   memory_size=30,
                   mini=1,
                   nb_closest=4,
                   nb_winners=1,
                   update_method=2)

SHUFFLE_DATA_ENABLED = 1
if SHUFFLE_DATA_ENABLED:
    shuffled_data = shuffle(new_data)
    ca.feed_data_online(shuffled_data)
else:
    ca.feed_data_online(new_data)


def print_metrics(method="closest", nb_winners=4):
    if clusters_true is not None:
        # clusters_true2 will contain ids instead of labels
        ids = collections.defaultdict(functools.partial(next,
                                                        itertools.count()))
        clusters_true2 = np.array([ids[label] for label in clusters_true])
    ca.print_confusion_matrix(new_data, clusters_true2, method=method,
                              nb_winners=nb_winners)


PRINT_METRICS_DEFINED = True
print_metrics(method="closest", nb_winners=2)
print_metrics(method="all")

# %%
