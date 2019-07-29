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
# ## Third draft of Contrast algorithm

# %% [markdown]
# - Rapport update
# - Seed random

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
from utils import plot_confusion_matrix
import collections
import functools
import itertools
import tkinter as tk
from tkinter.font import Font
# from sklearn.utils.multiclass import unique_labels

np.random.seed(1)

# %% [markdown]
# ## Global variables and functions

# %%
# --- Information for generating dataset ---
NB_FEATURES = 15
NB_GROUPS = 10
N = 300
DOMAIN_LENGTH = 200
DEV_MAX = 20

COLORS = ['r', 'g', 'b', 'y', 'c', 'm']

# stocks metadata as (DATASET_NAME, DATASET_EXTENSION, DATASET_PATH,
#     \ DATASET_CLUSTER_COLUMN_INDEX, DATASET_DATA_COLUMNS_INDICES)
METADATA = {
    'Cards': ('Cards', '.csv', 'data/', 1, (2, None), ["points", "bulges", "petals", "direction", "queue", "red", "green", "blue", "width", "height"]),
    'Cards2': ('Cards2', '.csv', 'data/', 1, (2, None), ["points", "bulges", "petals", "direction", "queue", "symmetries", "creases", "curviness", "red", "green", "blue", "width", "height"]),
    'Cards_truncated': ('Cards', '.csv', 'data/', 1, (2, 7)),
    'Zoo': ('zoo', '.csv', 'data/', 1, (2, None))
}

SHOULD_LOAD_DATASET = 1  # 0 to generate, 1 to load csv, 2 to load sklearn

if SHOULD_LOAD_DATASET == 1:
    NAME = 'Cards2'
    DATASET_NAME, DATASET_EXTENSION, DATASET_PATH, \
        DATASET_CLUSTER_COLUMN_INDEX, \
        DATASET_DATA_COLUMNS_INDICES, ADJECTIVES = METADATA[NAME]
    DATASET_PATH_FULL = DATASET_PATH + DATASET_NAME + DATASET_EXTENSION
elif SHOULD_LOAD_DATASET == 0:
    DATASET_PATH = 'data/'
    DATASET_NAME = 'dummy'
elif SHOULD_LOAD_DATASET == 2:
    from sklearn.datasets import load_iris
    load_dataset = load_iris
    DATASET_PATH = 'data/'
    DATASET_NAME = 'iris'

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

def color_converter(r, g, b):
    return('#%02x%02x%02x' % (int(r), int(g), int(b)))


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
    dataset = load_dataset()
    data = dataset["data"]
    clusters_true = dataset["target"]


assert data is not None, 'data is None'
assert clusters_true is not None, 'clusters_true is None'

# clusters_true2 will contain ids instead of labels
ids = collections.defaultdict(functools.partial(next,
                                                itertools.count()))
clusters_true2 = np.array([ids[label] for label in clusters_true])
nb_clusters_true = np.max(clusters_true2) + 1

# print(data)
# print(clusters_true)

# %% [markdown]
# ## Preview dataset

# %%
# columns = ["d" + str(i) for i in range(data.shape[1])] + ['true cluster']
# df = pd.DataFrame(np.hstack((data, np.reshape([clusters_true],
#                                               (data.shape[0], 1)))),
#                   columns=columns)
# true_data_plot = sns.pairplot(df, kind="scatter", hue='true cluster',
#                               vars=columns[:-1])
# true_data_plot.savefig(DATASET_PATH + DATASET_NAME + '_true.png')

# %% [markdown]
# ## The agent

# %%
class ContrastAgent(object):
    def __init__(self,
                 cmemory_size=50,
                 divide_by_dev=False,
                 eps=0.01,
                 maxi=2.5,
                 memory_size=50,
                 mini=1,
                 nb_closest=8,
                 nb_winners=1,
                 update_method=2):

        self.deviations = {1: [], 2: []}
        self.divide_by_dev = divide_by_dev
        self.eps = eps
        self.maxi = maxi
        self.memories = {1: [], 2: []}
        self.memory_sizes = {1: memory_size, 2: cmemory_size}
        self.mini = mini
        self.names_contrasts_neg = []
        self.names_contrasts_pos = []
        self.names_prototypes = []
        self.nb_closest = nb_closest
        self.nb_winners = nb_winners
        self.update_method = update_method
        self.weights = {1: [], 2: []}

    def adjusted_threshold(self, x, maxi, mini, a=0.2, b=5):
        """ https://www.desmos.com/calculator/rydyha6kmb """
        return (maxi - ((maxi - mini) / (1 + np.exp(-(a*(x-1) - b)))))

    def cluster_battles(self, obj, indices, data_order, nb_winners=1):
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
                    if np.abs(self.memories[data_order][a][j] - obj[j]) > \
                            np.abs(self.memories[data_order][b][j] - obj[j]):
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

    def extract_contrasts(self, obj, contrast_indices, data_order, fill_with_zeros=False, objects=[], use_objects=False):
        contrasts = []
        inversions = []
        for obj_i, i in enumerate(contrast_indices):
            # dim_contrast = 0  # TEST
            new_contrast = np.zeros(len(obj))
            # print("should be a new prototype: {}".format(self.memories[data_order][i]))  # TEST
            for j in range(len(obj)):
                if not self.is_in_cluster_dimension(obj, i, j, data_order, tolerance=1):
                    # dim_contrast += 1 # TEST
                    new_contrast[j] = (obj[j] - self.memories[data_order][i][j])
                    if use_objects:
                        new_contrast[j] = (obj[j] - objects[obj_i][j])
                    if self.divide_by_dev:
                        new_contrast /= self.deviations[data_order][i][j]
            # if new_contrasts is not zero and has at least one zero
            if np.any(new_contrast) and not np.all(new_contrast):
                if np.all(new_contrast <= 0):  # if everything negative, change to positive
                    new_contrast *= -1
                    inversions.append(True)
                else:
                    inversions.append(False)
                contrasts.append(new_contrast)
            elif fill_with_zeros:
                contrasts.append(np.zeros(len(obj)))
                inversions.append(False)
            # print("dim_contrast: {}".format(dim_contrast)) # TEST
        return (np.array(contrasts), inversions)

    def feed_data_online(self, new_data, data_order):
        is_empty = len(self.memories[data_order]) == 0
        if is_empty:
            self.update_clusters(new_data[0], data_order)
        for obj in new_data[is_empty:]:
            nb_winners = self.nb_winners
            if data_order == 2:
                nb_winners = 1
            # find_cluster updates clusters
            nb = len(self.find_cluster(
                obj, data_order,
                update=True, nb_winners=nb_winners))
            if data_order == 1:
#                 new_contrasts = np.abs(self.memories[data_order][-(nb+1):-1] - obj)
                new_contrasts = self.extract_contrasts(obj, list(range(-(nb+1),-1)), data_order)[0]
                if len(new_contrasts) > 0:
                    self.feed_data_online(new_contrasts, data_order=2)
            self.forget_if_needed(data_order)
    
    def find_closest_object(self, proto_i, data, data_order=1):
        """ finds the closest object to the prototype (relevant dimensions then distance)"""
        matching_dims = np.zeros(len(data))
        for i, obj in enumerate(data):
            for j in range(len(obj)):
                if self.is_in_cluster_dimension(obj, proto_i, j, data_order):
                    matching_dims[i] += 1
        max_dims = np.max(matching_dims)
        closest_indices = \
            np.flatnonzero(matching_dims == max_dims)
        closest_objects = [data[i] for i in closest_indices]
        best_i = 0
        best_d = np.inf
        proto_vec = self.memories[data_order][proto_i]
        proto_dev = self.deviations[data_order][proto_i]
        for i, obj in enumerate(closest_objects):
            d = 0
            for j, x in enumerate(obj):
                d += np.abs(obj[j] - proto_vec[j]) / proto_dev[j]
            if d < best_d:
                best_d = d
                best_i = i
        return closest_indices[best_i]

    def find_cluster(self, obj, data_order, update=True, nb_winners=1):
        n_closest_indices, max_dims = self.find_n_closest_prototypes(obj,
                                                           data_order,
                                                           n=self.nb_closest)
        if data_order >= 2:  # looking for the opposite contrast
            n_closest_indices2, max_dims2 = self.find_n_closest_prototypes(-obj,
                                                   data_order,
                                                   n=self.nb_closest)
            if max_dims2 > max_dims:
                n_closest_indices = n_closest_indices2
                max_dims = max_dims2
                obj = -obj
        winner_indices = self.cluster_battles(obj, n_closest_indices,
                                              data_order,
                                              nb_winners=nb_winners)
        if update:
            self.update_clusters(obj, data_order, winner_indices,
                                 method=self.update_method)
        return winner_indices

    def find_description(self, obj, data_order, nb_winners=4):
        winner_indices1 = self.find_cluster(obj, data_order, update=False, nb_winners=nb_winners)
        raw_contrasts, inversions  = self.extract_contrasts(obj, winner_indices1, data_order, fill_with_zeros=True)
        winner_indices2 = []
        for contrast in raw_contrasts:
            if not np.any(contrast):
                winner_indices2.append(-1)
            else:
                winner_array2 = self.find_cluster(contrast, data_order+1, update=False, nb_winners=1)  # only one contrast max
                winner_indices2.append(winner_array2[0])
        final_bit_weights = []
        weights_sorted1 = np.sort(self.weights[data_order])[::-1]
        weights_sorted2 = np.sort(self.weights[data_order+1])[::-1]
        for i, index1 in enumerate(winner_indices1):
            w1 = self.weights[data_order][index1]
            bit_weight1 = np.nonzero(weights_sorted1==w1)[0][0] + 1
            index2 = winner_indices2[i]
            bit_weight2 = 1
            if index2 != -1:
                w2 = self.weights[data_order+1][index2]
                bit_weight2 = np.nonzero(weights_sorted2==w2)[0][0] + 1
            final_bit_weights.append(bit_weight1 * bit_weight2)
        best = np.argmin(final_bit_weights)
        return(winner_indices1[best], winner_indices2[best], inversions[best])
    
    def find_n_closest_prototypes(self, obj, data_order, n=4):
        matching_dims = np.zeros(len(self.memories[data_order]))
        for i in range(len(self.memories[data_order])):
            for j in range(len(obj)):
                if self.is_in_cluster_dimension(obj, i, j, data_order):
                    matching_dims[i] += 1
#         unique_dims = np.unique(matching_dims)
#         closest_indices = np.concatenate(
#             [np.flatnonzero(matching_dims == i) for i in unique_dims])
        max_dims = np.max(matching_dims)
        # if data_order == 1:  # TEST
            # print("max_dims: {}".format(max_dims))  # TEST
        closest_indices = \
            np.flatnonzero(matching_dims == max_dims)
        # print("closest_indices: {}".format(closest_indices))
        if len(closest_indices) > n:
            # closest_indices = np.random.choice(closest_indices, 4,
            #                                    replace=False)
            closest_indices = closest_indices[-n:]
        # if data_order == 1:  # TEST
            # for i in closest_indices:  # TEST
                # print("index {}: {} matching dims".format(i, matching_dims[i]))  # TEST
        return closest_indices.tolist(), max_dims

    def find_optimal_description(self, shuffled_data, descriptions, canvas, data_order=1):
        "describes the first object in shuffled_data compared to the others"
        obj = shuffled_data[0]
        obj_desc = descriptions[0]
        base_str = "Object 1 is "
        same_proto = [desc[0] != obj_desc[0] for desc in descriptions[1:]]
        proto_translated = self.translate_prototype(obj_desc[0])
        if all(same_proto):
            canvas.create_text(cw/2, yshift + 350, fill="black", font="Droid", text=base_str + proto_translated)
            return
        else:
            contrast_indices = []
            objects = []
            for i, is_different in enumerate(same_proto):
                if not is_different:
                    contrast_indices.append(descriptions[i+1][0])
                    objects.append(shuffled_data[i+1])
            raw_contrasts, inversions  = self.extract_contrasts(obj, contrast_indices, data_order, fill_with_zeros=True, objects=objects, use_objects=True)
            winner_indices2 = []
            for contrast in raw_contrasts:
                if not np.any(contrast):
                    winner_indices2.append(-1)
                else:
                    winner_array2 = self.find_cluster(contrast, data_order+1, update=False, nb_winners=1)  # only one contrast max
                    winner_indices2.append(winner_array2[0])
            final_bit_weights = []
            weights_sorted1 = np.sort(self.weights[data_order])[::-1]
            weights_sorted2 = np.sort(self.weights[data_order+1])[::-1]
            pop_later = []
            for i, index1 in enumerate(contrast_indices):
                w1 = ca.weights[data_order][index1]
                bit_weight1 = np.nonzero(weights_sorted1==w1)[0][0] + 1
                index2 = winner_indices2[i]
                bit_weight2 = 1
                if index2 != -1:
                    w2 = ca.weights[data_order+1][index2]
                    bit_weight2 = np.nonzero(weights_sorted2==w2)[0][0] + 1
                    final_bit_weights.append(bit_weight1 * bit_weight2)
                else:
                    pop_later.append(i)
            for i in pop_later:
                winner_indices2.pop(i)
                inversions.pop(i)
            if len(final_bit_weights) > 0:
                best = np.argmin(final_bit_weights)
                final_desc = (winner_indices2[best], inversions[best])
#                 print(final_desc)
    #             adverbs = ["less", "more"]
    #             if final_desc[-1]:
    #                 adverbs = ["more", "less"]
    #             canvas.create_text(cw/2, yshift + 400, fill="black", font="Droid", text=base_str + proto_translated + " " + adverbs[final_desc[-1]] + " " + str(final_desc[0]))
                canvas.create_text(cw/2, yshift + 400, fill="black", font="Droid", text=base_str + proto_translated + " " + self.translate_contrast(final_desc[0], final_desc[1]))
            else:
                canvas.create_text(cw/2, yshift + 400, fill="black", font="Droid", text=base_str + proto_translated + ", unable to determine contrast")
    
    def forget_if_needed(self, data_order):
        while len(self.memories[data_order]) > self.memory_sizes[data_order]:
            i = np.argmin(self.weights[data_order])
            self.memories[data_order].pop(i)
            self.weights[data_order].pop(i)
            self.deviations[data_order].pop(i)
            # self.memories[data_order].pop(0)
            # self.weights[data_order].pop(0)
            # self.deviations[data_order].pop(0)

    def is_in_cluster_dimension(self, obj, i, j, data_order, tolerance=1):
        return np.abs(obj[j] - self.memories[data_order][i][j]) <= \
            self.deviations[data_order][i][j] * \
            self.adjusted_threshold(self.weights[data_order][i],
                                    self.maxi, self.mini) * \
            tolerance
    
    def learn_names(self, objects, prototype_labels, contrast_labels, data_order=1):
        self.names_prototypes = ["?"] * len(self.memories[data_order])
        self.names_contrasts_pos = ["?"] * len(self.memories[data_order])
        self.names_contrasts_neg = ["?"] * len(self.memories[data_order])
        for i, obj in enumerate(objects):
            desc = self.find_description(obj, data_order=data_order, nb_winners=4)
            self.names_prototypes[desc[0]] = prototype_labels[i]
            if desc[2]:
                self.names_contrasts_neg[desc[1]] = contrast_labels[i]
            else:
                self.names_contrasts_pos[desc[1]] = contrast_labels[i]
        return
             
    def translate_contrast(self, contrast_i, is_inverted, hardcoded=False):
        if hardcoded:
            adverbs = ["less", "more"]
            if is_inverted:
                adverbs = ["more", "less"]
            contrast_vec = ca.memories[2][contrast_i]
            positive_values = [x > 0 for x in contrast_vec]
            negative_values = [x < 0 for x in contrast_vec]
            adjectives = ADJECTIVES
            description = ""
            if any(positive_values):
                positive_desc = adverbs[not is_inverted] + " "
                comma = False
                for i, is_pos in enumerate(positive_values):
                    if is_pos:
                        if comma:
                            positive_desc += ", "
                        positive_desc += adjectives[i]
                        comma = True
                description += positive_desc
            if any(negative_values):
                negative_desc = adverbs[is_inverted] + " "
                comma = False
                for i, is_neg in enumerate(negative_values):
                    if is_neg:
                        if comma:
                            negative_desc += ", "
                        negative_desc += adjectives[i]
                        comma = True
                if description != "":
                    description += ", "
                description += negative_desc
            return description
        else:
            assert len(self.names_contrasts_pos) > 0, "positive contrast names not learned"
            assert len(self.names_contrasts_neg) > 0, "negative contrast names not learned"
            name_pos = self.names_contrasts_pos[contrast_i]
            name_neg = self.names_contrasts_neg[contrast_i]
            adverbs = ["less", "more"]
            if name_pos != "?":
                return adverbs[not is_inverted] + " " + name_pos
            elif name_neg != "?":
                return adverbs[is_inverted] + " " + name_neg
            else:
                return adverbs[not is_inverted] + " " + str(contrast_i)
    
    def translate_prototype(self, proto_i, hardcoded=False):
        if hardcoded:
            proto_vec = self.memories[1][proto_i]
            proto_type = "Spade"
            if proto_vec[0] == 0:
                proto_type = "Clubs"
            elif proto_vec[0] == 4:
                proto_type = "Diamond"
            elif proto_vec[1] == 2:
                proto_type = "Heart"
            return(proto_type)
        else:
            assert len(self.names_prototypes) > 0, "prototype names not learned"
            name = self.names_prototypes[proto_i]
            if name != "?":
                return name
            else:
                return str(proto_i)
    
    def update_clusters(self, obj, data_order,
                        winner_indices=None, method=2):
        """
            method 1: weighted averaged standard deviations
            method 2: weighted averaged distances
        """

        if winner_indices is not None:

            for cluster_index in winner_indices:
                # old_weight = self.weights[data_order][cluster_index]
                # old_deviations = self.deviations[data_order][cluster_index]
                # old_prototype = self.memories[data_order][cluster_index]
                old_weight = self.weights[data_order].pop(cluster_index)
                old_deviations = self.deviations[data_order].pop(cluster_index)
                old_prototype = self.memories[data_order].pop(cluster_index)

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

                # now we nullify components that are null in either: absorbing zeros
                if data_order >= 2:
                    for j, x in enumerate(obj):
                        if x == 0:
                            new_prototype[j] = 0
                        elif old_prototype[j] == 0:
                            new_prototype[j] = 0
                    if np.all(new_prototype <= 0):  # if everything negative, change to positive
                        new_prototype *= -1

                if np.any(new_prototype):
                    self.weights[data_order].append(new_weight)
                    self.deviations[data_order].append(new_deviations)
                    self.memories[data_order].append(new_prototype)

                # if data_order == 1:  # TEST
                    # print("old_prototype: {}".format(old_prototype))  # TEST
                    # print("new_prototype: {}".format(new_prototype))  # TEST

        self.weights[data_order].append(1)
        new_eps = self.eps
        # if data_order >= 2:
        #     new_eps = 1
        self.deviations[data_order].append(np.abs(new_eps * obj))
        self.memories[data_order].append(obj)

    # --- Plot functions ---

    def print_confusion_matrix(self, data, clusters_true,
                               cmap=plt.cm.Blues,
                               cmap2=plt.cm.Reds,
                               method="closest",
                               nb_winners=4,
                               tolerance=2.5):
        """
            method can be either "all" or "closest"
        """
        data_order = 1
        nb_clusters_true = np.max(clusters_true) + 1
        cluster_true_sizes = [np.count_nonzero(clusters_true == i) for i in
                              range(nb_clusters_true)]
        cm = np.zeros((len(self.memories[data_order]), nb_clusters_true))

        if method == "all":
            for obj_i, obj in enumerate(data):
                for i in range(len(self.memories[data_order])):
                    if all([self.is_in_cluster_dimension(
                                obj, i, j, data_order, tolerance=tolerance) \
                            for j in range(len(obj))]):
                        cm[i][clusters_true[obj_i]] += 1
        else:
            if method != "closest":
                print("print_confusion_matrix: unknown"
                      "is_in_cluster_method, using 'closest'")
            for obj_i, obj in enumerate(data):
                winner_indices = self.find_cluster(obj, data_order, update=False, nb_winners=nb_winners)
                for i in winner_indices:
                    cm[i][clusters_true[obj_i]] += 1

        cm /= cluster_true_sizes

        # print(cm)

        title = 'Normalized confusion matrix'
        fig, (ax1, ax2) = plt.subplots(
                              1, 2, sharey=True,
                              figsize=(4+nb_clusters_true, 4+len(self.memories[data_order])/5)
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

        im2 = ax2.imshow(np.reshape(self.weights[data_order], (-1, 1)),
                         interpolation='nearest',
                         cmap=cmap2)
        ax2.figure.colorbar(im2, ax=ax2)
        ax2.set(xticks=np.arange(0),
                yticks=np.arange(len(self.weights[data_order])),
                yticklabels=np.arange(len(self.weights[data_order])),
                title="Weights")

        # Rotate the tick labels and set their alignment.
        # plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
        #          rotation_mode="anchor")

        # Loop over data dimensions and create text annotations.
        fmt = '.2f'
        fmt2 = 'd'
        thresh = cm.max() / 2.
        thresh2 = np.max(self.weights[data_order]) / 2
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax1.text(j, i, format(cm[i, j], fmt),
                         ha="center", va="center",
                         color="white" if cm[i, j] > thresh else "black")
            ax2.text(0, i, format(self.weights[data_order][i], fmt2),
                     ha="center", va="center",
                     color="white" if self.weights[data_order][i] > thresh2 else "black")
        fig.tight_layout()
        plt.savefig(DATASET_PATH + DATASET_NAME + '_confusion_matrix.png')
        return fig

    def print_contrasts(self, nb_print=10, data_order=2, should_sort=True):
        if nb_print > self.memory_sizes[data_order]:
            nb_print = self.memory_sizes[data_order]
        order_to_print = np.arange(len(self.weights[data_order]))
        if should_sort:
            order_to_print = np.argsort(self.weights[data_order])[::-1]
        memory = np.array(ca.memories[data_order])[order_to_print][:nb_print]
        weights_to_print = np.array(self.weights[data_order])[order_to_print][:nb_print]
        n_dim = data.shape[1]+1
        fig, axes = plt.subplots(n_dim, 1, figsize=(nb_print, n_dim+5), sharex=True)
        x = np.arange(nb_print)
        mem = np.array(memory)
        for i, ax in enumerate(axes):
            if i == n_dim-1:
                sns.barplot(x=x, y=weights_to_print, ax=ax)
                ax.set(title="Weights")
            else:
                sns.barplot(x=x, y=mem[:,i], ax=ax)
        fig.tight_layout()
        plt.savefig(DATASET_PATH + DATASET_NAME + '_contrasts.png')


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


MANIPULATION_ROUNDS = 0

def manipulate_data(rounds):
    for i in range(rounds):
        manip = np.random.randint(2)
        j = np.random.randint(new_data.shape[1])
        x = (2 * np.random.random() - 1) * 10**np.random.randint(-8, 8)
        if manip == 0:
            data_add(j, x)
            print("Column {}: added {}".format(j, x))
        else:
            data_mult(j, x)
            print("Column {}: multiplied by {}".format(j, x))

manipulate_data(MANIPULATION_ROUNDS)

# if MANIPULATION_ROUNDS:
#     print(data)
#     print(new_data)

# %% [markdown]
# ## Clusterize

# %%
ca = ContrastAgent(cmemory_size=10,
                   divide_by_dev=True,
                   eps=0.01,
                   maxi=1,
                   memory_size=10,
                   mini=1,
                   nb_closest=4,
                   nb_winners=1,
                   update_method=2)

NB_REPETITIONS = 4
SHUFFLE_DATA_ENABLED = 1
for i in range(NB_REPETITIONS):
    if SHUFFLE_DATA_ENABLED:
        shuffled_data = shuffle(new_data)
        ca.feed_data_online(shuffled_data, 1)
    else:
        ca.feed_data_online(new_data, 1)

def print_metrics(d, ctrue, method="closest", nb_winners=4, tolerance=2.5, nb_print=20):
    """
    nb_winners only relevant for method "closest"
    tolerance only relevant for method "all"
    """
    ca.print_confusion_matrix(d, ctrue, method=method,
                              nb_winners=nb_winners, tolerance=tolerance)
    ca.print_contrasts(nb_print=nb_print, should_sort=False)


PRINT_METRICS_DEFINED = True
print_metrics(new_data, clusters_true2, method="closest", nb_winners=1, tolerance=1)

# %% [markdown]
# ## Description tests

# %%
# for obj in new_data:
#     winner = ca.find_cluster(obj, data_order=1, update=False, nb_winners=1)[0]
#     cluster, contrast = ca.find_description(obj, data_order=1, nb_winners=4)
#     if winner != cluster:
#         print("Candidates:", ca.find_cluster(obj, data_order=1, update=False, nb_winners=4))
#         print("Winner:", winner)
#         print("Description:", cluster, contrast)
#         print()

# %%
assert NAME == 'Cards2', "use dataset Cards2"
example_indices = [0, 4, 16, 21, 32, 33, 41, 44, 56, 69, 70, 77]
prototype_labels = ["Spade", "Spade", "Spade", "Clubs", "Clubs", "Clubs", "Heart", "Heart", "Heart", "Diamond", "Diamond", "Diamond"]
contrast_labels = ["small", "white", "huge", "big", "black", "white", "light red", "small", "dark red", "tiny", "big", "light blue"]
examples = [data[i] for i in example_indices]
ca.learn_names(examples, prototype_labels, contrast_labels, data_order=1)


# %% [markdown]
# ## GUI descriptions

# %%
def add_heart(canvas, tag, xshift=0, yshift=0, w=300, h=300, color="#fff", linewidth=2):
    circle1 = np.array([0, 0, w/2, 2*h/3])
    circle2 = np.array([w/2, 0, w, 2*h/3])
    triangle = np.array([0, h/3, w, h/3, w/2, h])
    line = np.array([0+1, h/3, w-1, h/3])
    circle1[::2] += xshift
    circle2[::2] += xshift
    triangle[::2] += xshift
    line[::2] += xshift
    circle1[1::2] += yshift
    circle2[1::2] += yshift
    triangle[1::2] += yshift
    line[1::2] += yshift
    canvas.create_arc(*zip(circle1), start=0, extent=180, outline="#000", fill=color, width=linewidth, tags=tag)
    canvas.create_arc(*zip(circle2), start=0, extent=180, outline="#000", fill=color, width=linewidth, tags=tag)
    canvas.create_polygon(*zip(triangle), outline='#000', fill=color, width=linewidth, tags=tag)
    canvas.create_line(*zip(line), fill=color, width=linewidth, tags=tag)
    
def add_diamond(canvas, tag, xshift=0, yshift=0, w=300, h=300, color="#fff", linewidth=2):
    diamond = np.array([0, h/2, w/2, 0, w, h/2, w/2, h])
    diamond[::2] += xshift
    diamond[1::2] += yshift
    canvas.create_polygon(*zip(diamond), outline='#000', fill=color, width=linewidth, tags=tag)

def add_spade(canvas, tag, xshift=0, yshift=0, w=300, h=300, color="#fff", linewidth=2):
    circle1 = np.array([0, h/3, w/2, h]) 
    circle2 = np.array([w/2, h/3, w, h])
    triangle = np.array([0, 2*h/3, w, 2*h/3, w/2, 0])
    triangle2 = np.array([w/2, 2*h/3, w/2-w/7, 9*h/8, w/2+w/7, 9*h/8])
    triangle3 = np.array([w/2, 2*h/3, w/2-w/30, 5*h/6, w/2+w/30, 5*h/6]) 
    line = np.array([0+1, 2*h/3, w-1, 2*h/3])
    circle1[::2] += xshift
    circle2[::2] += xshift
    triangle[::2] += xshift
    triangle2[::2] += xshift
    triangle3[::2] += xshift
    line[::2] += xshift
    circle1[1::2] += yshift
    circle2[1::2] += yshift
    triangle[1::2] += yshift
    triangle2[1::2] += yshift
    triangle3[1::2] += yshift
    line[1::2] += yshift
    canvas.create_arc(*zip(circle1), start=180, extent=180, outline="#000", fill=color, width=linewidth, tags=tag)
    canvas.create_arc(*zip(circle2), start=180, extent=180, outline="#000", fill=color, width=linewidth, tags=tag)
    canvas.create_polygon(*zip(triangle), outline='#000', fill=color, width=linewidth, tags=tag)
    canvas.create_polygon(*zip(triangle2), outline="#000", fill=color, width=linewidth, tags=tag)
    canvas.create_polygon(*zip(triangle3), outline=color, fill=color, width=linewidth, tags=tag)
    canvas.create_line(*zip(line), fill=color, width=linewidth, tags=tag)

def add_clubs(canvas, tag, xshift=0, yshift=0, w=300, h=300, color="#fff", linewidth=2):
    circle1 = np.array([0, h/2-h/5, w/2, h-h/5])
    circle2 = np.array([w/2, h/2-h/5, w, h-h/5])
    circle3 = np.array([w/4, 0, 3*w/4, h/2])
    triangle2 = np.array([w/2, 2*h/3-h/5, w/2-w/7, h, w/2+w/7, h]) 
    triangle3 = np.array([w/2, 2*h/3-h/5, w/2-w/10, 4*h/5-h/5, w/2+w/10, 4*h/5-h/5]) 
    circle1[::2] += xshift
    circle2[::2] += xshift
    circle3[::2] += xshift
    triangle2[::2] += xshift
    triangle3[::2] += xshift
    circle1[1::2] += yshift
    circle2[1::2] += yshift
    circle3[1::2] += yshift
    triangle2[1::2] += yshift
    triangle3[1::2] += yshift
    canvas.create_oval(*zip(circle1), outline="#000", fill=color, width=linewidth, tags=tag)
    canvas.create_oval(*zip(circle2), outline="#000", fill=color, width=linewidth, tags=tag)    
    canvas.create_oval(*zip(circle3), outline="#000", fill=color, width=linewidth, tags=tag)
    canvas.create_arc(*zip(circle3), start=195, extent=150, outline=color, fill=color, width=linewidth, tags=tag)
    canvas.create_polygon(*zip(triangle2), outline="#000", fill=color, width=linewidth, tags=tag)
    canvas.create_polygon(*zip(triangle3), outline=color, fill=color, width=linewidth, tags=tag)


# %%
nb_group = 5
root = tk.Tk()
cw = 1050
ch = 500
# w = tk.Label(root, text="Hello Tkinter!")
canvas = tk.Canvas(root, bg="#ffffff", width=cw, height=ch)

canvas.pack()
data_with_clusters = np.hstack((data, np.reshape([clusters_true], (data.shape[0], 1))))

shuffled_data = shuffle(data_with_clusters)[:nb_group]

add_functions = {}
add_functions["Spade"] = add_spade
add_functions["Diamond"] = add_diamond
add_functions["Heart"] = add_heart
add_functions["Clubs"] = add_clubs
# [1,2,1,1,1], [0,3,3,1,1], [1,2,0,-1,1], [4,0,0,0,0]

xshift = 120
yshift = 50

descriptions = []
canvas.create_text(xshift-70, yshift+30, fill="black", font="Droid", text="Objects")
canvas.create_text(xshift-70, yshift+180, fill="black", font="Droid", text="\"Prototype\"")
canvas.create_text(xshift-70, yshift+300, fill="black", font="Droid", text="Descriptions")
for obj in shuffled_data:
    add_functions[obj[-1]](canvas, xshift=xshift, yshift=yshift, tag='test', w=5*obj[-3], h=5*obj[-2], color=color_converter(obj[-6], obj[-5], obj[-4]))
    descriptions.append(ca.find_description(obj[:-1], data_order=1, nb_winners=4))
    proto_i = descriptions[-1][0]
#     display_vec = ca.memories[1][proto_i].tolist() # the prototype itself
    best_obj_i = ca.find_closest_object(proto_i, data, data_order=1)
    display_vec = data[best_obj_i].tolist() # closest object to prototype
    display_type = "Spade"
    if display_vec[0] == 0:
        display_type = "Clubs"
    elif display_vec[0] == 4:
        display_type = "Diamond"
    elif display_vec[1] == 2:
        display_type = "Heart"
    display_vec.append(display_type)
    
    add_functions[display_vec[-1]](canvas, xshift=xshift, yshift=yshift + 150, tag='test', w=5*display_vec[-3], h=5*display_vec[-2], color=color_converter(display_vec[-6], display_vec[-5], display_vec[-4]))
    canvas.create_text(xshift+50, yshift+300, fill="black", font="Droid", text=str(descriptions[-1]))
    xshift += 200

ca.find_optimal_description([obj[:-1] for obj in shuffled_data], descriptions, canvas)
    
root.update_idletasks()
root.update()

root.mainloop()


# %% [markdown]
# ## Comparison to other algorithms

# %%
def print_matrix(predictions):
    class_names = np.array([str(i) for i in range(1+np.max(
        np.concatenate([clusters_true2, predictions])))])
    plot_confusion_matrix(
        clusters_true2, predictions, classes=class_names,
        normalize=True, title='Normalized confusion matrix',
        path=DATASET_PATH + DATASET_NAME, should_save=True)


# %% [markdown]
# ### K-means

# %%
from sklearn.cluster import KMeans
kmeans_predictions = KMeans(n_clusters=nb_clusters_true).fit_predict(new_data)
print_matrix(kmeans_predictions)

# %%
from sklearn.mixture import GaussianMixture
gmm_predictions = GaussianMixture(n_components=nb_clusters_true).fit_predict(new_data)
print_matrix(gmm_predictions)
