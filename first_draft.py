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
# ## First draft of Contrast algorithm

# %% [markdown]
# ## To do
# - Compute standard_distances for each cluster
# - When updating st_dists of cluster, project along both infinite-stdev and 0-stdev dims
# - Update sensitivity -> how?
# - Inverse law when seeing duplicates
# - Implement context
# - Compare stdevs and distances
# - Clustering validity checking methods

# %% [markdown]
# ## Imports

# %%
import time
import csv
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns
import pandas as pd

np.random.seed(1)

# %% [markdown]
# ## Global variables

# %%
NB_FEATURES = 5
NB_GROUPS = 5
N = 500
SIZE = 200
DEV = 30

colors = ['r', 'g', 'b', 'y', 'c', 'm']

# stocks metadata as (DATASET_EXTENSION, DATASET_PATH, 
#     \ DATASET_CLUSTER_COLUMN_INDEX, DATASET_DATA_COLUMNS_INDICES)
metadata = {
    'Cards': ('Cards', '.csv', 'data/', 1, (2, None)),
    'Cards_truncated': ('Cards', '.csv', 'data/', 1, (2, 7))
}

SHOULD_LOAD_DATASET = 1 # 0 to generate, 1 to load
# used if 0
DUMMY_DATA_PATH = 'data/dummy_data.npy'
DUMMY_CLUSTERS_TRUE_PATH = 'data/dummy_clusters.npy'
# used if 1
NAME = 'Cards'
DATASET_NAME, DATASET_EXTENSION, DATASET_PATH, DATASET_CLUSTER_COLUMN_INDEX, \
    DATASET_DATA_COLUMNS_INDICES= metadata[NAME]
DATASET_PATH_FULL = DATASET_PATH + DATASET_NAME + DATASET_EXTENSION


# %% [markdown]
# ## Helper functions

# %%
def generate_dataset(nb_features = NB_FEATURES, nb_groups = NB_GROUPS, n = N, size=SIZE, dev=DEV):
    clusters_true = np.zeros(n)
    means = np.zeros((nb_groups, nb_features)) # holds the mean of each group
    st_devs = np.zeros((nb_groups, nb_features)) # holds the st_devs of each group
    for i in range(nb_groups):
        for j in range(nb_features):
            means[i][j] = np.random.randint(-size, size)
            st_devs[i][j] = dev*np.random.random()
    data = np.zeros((n, nb_features))
    for i in range(n):
        gi = np.random.randint(0,nb_groups)
        clusters_true[i] = gi
        for j in range(nb_features):
            data[i][j] = np.random.normal(loc = means[gi][j], scale = st_devs[gi][j])
    np.save(DUMMY_DATA_PATH, data)
    np.save(DUMMY_CLUSTERS_TRUE_PATH, clusters_true)
    
def distance(p, q, ord=2):
    return(np.linalg.norm(p-q, ord))


# %% [markdown]
# ## Generate or load dataset

# %%
if SHOULD_LOAD_DATASET:
    start, end = DATASET_DATA_COLUMNS_INDICES
    df1 = pd.read_csv(DATASET_PATH_FULL)
    df1_np = df1.to_numpy(copy=True)
    data = df1_np[:,start:end].astype('float')
    clusters_true = df1_np[:,DATASET_CLUSTER_COLUMN_INDEX]
else:
    generate_dataset()
    data = np.load(DUMMY_DATA_PATH)
    clusters_true = np.load(DUMMY_CLUSTERS_TRUE_PATH)
    
assert data is not None, 'data is None'
assert clusters_true is not None, 'clusters_true is None'

# %%
# print(data)
# print(clusters_true)

# %% [markdown]
# ## Preview dataset

# %%
columns = ["d" + str(i) for i in range(data.shape[1])] + ['true cluster']
df = pd.DataFrame(np.hstack((data,np.reshape([clusters_true],(data.shape[0],1)))), columns=columns)
true_data_plot = sns.pairplot(df, kind="scatter", hue='true cluster', vars=columns[:-1])
true_data_plot.savefig(DATASET_PATH + DATASET_NAME + '_true.png')


# %% [markdown]
# ## The agent

# %%
class ContrastAgent(object):
    def __init__(self, rigged_shuffle=False, verbose=False):
        self.clusters = np.array([]) # clusters[i] == j means that point i belongs to cluster j
        self.cluster_sizes = []
        self.data = np.array([[]])
        self.first_time = True # is True if nb
        self.nb_clusters = 0
        self.nb_seen = 0
        self.sensitiveness_find_cluster = 1.5 # if a point is alone with a radius of sensitiveness_find_cluster * stdist, create a new cluster
        self.sensitiveness_inf_dims = 5 # sensitiveness to determine if a dimension is too variable to be relevant for the cluster
        self.sensitiveness_zero_dims = 0.1 # sensitiveness to determine if a dimension is too concentrated to be relevant for the cluster
        self.stdist = 0
        self.stdists_per_cluster=[]
        self.verbose = verbose

    def clusterize_online(self):
        assert len(self.data), "empty data"
        if self.nb_clusters == 0:
            self.new_cluster(0)
            self.one_more_seen()
        for i, p in enumerate(self.data[self.nb_seen:], start=self.nb_seen):
            allZeros = self.find_cluster(i, p, i)
            if not allZeros:
                self.one_more_seen()

    def get_cluster_points(self, i):
        return data[np.argwhere(self.clusters == i)]

    def get_cluster_dimensions(self, i):
        """ return (relevant_dims, inf_dims, zero_dims) """
        stdist_i = self.stdists_per_cluster[i]
        cluster_points = self.get_cluster_points(i)
        relevant_dims, inf_dims, zero_dims = [], [], []
        for j in range(cluster_points.shape[1]):
            stdev = np.std(cluster_points[:,j])
            if stdev > self.sensitiveness_inf_dims * stdist_i:
                inf_dims.append(j)
            elif stdev < self.sensitiveness_zero_dims * stdist_i:
                zero_dims.append(j)
            else:
                relevant_dims.append(j)
        return relevant_dims, inf_dims, zero_dims
        
    def feed_data(self, d, shuffle=False):
        """Adds data to the agent's memory"""
        data = np.copy(d)
        if shuffle:
            np.random.shuffle(data)
        if self.nb_clusters == 0:
            self.data = np.copy(data)
            self.clusters = np.zeros(len(data))
            self.clusters.fill(-1)
        else:
            new_data = np.vstack((self.data, np.copy(data)))
            self.data = new_data
            new_clusters = np.zeros(len(data))
            new_clusters.fill(-1)
            new_clusters_all = np.hstack((self.clusters, new_clusters))
            self.clusters = new_clusters_all

    def find_cluster(self, i, p, until=None, recollection=False):
        distances = np.array([distance(p,q) for q in self.data[:until]])
        allZeros = np.all(distances == 0)
        dist_min = np.min(np.trim_zeros(distances)) if not allZeros else 0
        closest = np.argmin(distances)
        if dist_min > self.sensitiveness_find_cluster * self.stdist and not 0 in distances:
            if self.verbose:
                print("{} -> new cluster".format(i))
            if self.clusters[i] == -1 or not self.cluster_sizes[int(self.clusters[i])] == 1:
#                 if p not already seen or is not already alone
                self.new_cluster(i)
        else:
            if self.verbose:
                print("{} -> cluster of {}".format(i, closest))
            self.clusters[i] = self.clusters[closest]
            if not allZeros:
                self.update_stdists_per_cluster(int(self.clusters[i]), p)
            self.cluster_sizes[int(self.clusters[i])] += 1
        if not recollection and not allZeros:
            self.update_stdist(self.nb_seen, dist_min)
        return allZeros
            
    def new_cluster(self, p_index):
        self.clusters[p_index] = self.nb_clusters
        self.nb_clusters += 1
        self.cluster_sizes.append(1)
        self.stdists_per_cluster.append(0)

    def one_more_seen(self):
        self.nb_seen += 1

#     def print_clusters_old(self, only=-1):
#         self.colors = cm.rainbow(np.linspace(0, 1, self.nb_clusters))
#         for k in range(self.nb_clusters):
#             if only < 0 or k == only:
#                 points = np.array([self.data[i] for i in range(len(self.data)) if self.clusters[i] == k])
#                 plt.plot(points[:,0], points[:,1], 'o', color=self.colors[k])
                
    def print_clusters(self):
        columns = ["d" + str(i) for i in range(self.data.shape[1])] + ['affected cluster']
        df_1 = pd.DataFrame(np.hstack((self.data,np.reshape([self.clusters],(self.data.shape[0],1)))),  columns=columns)
        agent_plot = sns.pairplot(df_1, kind="scatter", hue="affected cluster", vars=columns[:-1])
        agent_plot.savefig(DATASET_PATH + DATASET_NAME + '_agent.png')    
            
    def shuffle(self, data_to_shuffle):
        np.random.shuffle(data_to_shuffle)

    def update_clusters(self, until_update=None, until_dist=None, recollection = True):
        for i, p in enumerate(self.data[:until_update]):
            self.find_cluster(i, p, until_dist, recollection = recollection)
        
    def update_stdist(self, nb_seen, distance):
        old_dist = self.stdist
        self.stdist = (max(1,nb_seen-1) * self.stdist + distance) / (nb_seen)
        if self.verbose:
            print('distance = {}, self.stdist = {} -> {}'.format(distance, old_dist, self.stdist))

    def update_stdists_per_cluster(self, i, p):
        old_dist = self.stdists_per_cluster[i]
        relevant_dims, inf_dims, zero_dims = self.get_cluster_dimensions(i)
        if len(relevant_dims):
            cluster_points_rd = self.get_cluster_points(i)[:,relevant_dims]
            p_rd = p[relevant_dims]
        elif len(inf_dims):
            cluster_points_rd = self.get_cluster_points(i)[:,inf_dims]
            p_rd = p[inf_dims]
        else:
            cluster_points_rd = self.get_cluster_points(i)[:,zero_dims]
            p_rd = p[zero_dims]
        array_dists = np.array([distance(p,q) for q in cluster_points_rd])
#         print(relevant_dims, inf_dims, zero_dims)
#         print(array_dists)
        dist_p = np.min(array_dists)
        csize = cluster_points_rd.shape[0]
        self.stdists_per_cluster[i] = (old_dist * csize + dist_p) / (csize + 1)

# %%
ca = ContrastAgent(verbose=False)
ca.feed_data(data, shuffle=True)
ca.clusterize_online()
print("Final stdist: {}".format(ca.stdist))
print("All points in a cluster? {}".format(-1 not in ca.clusters))
ca.print_clusters()

# %%
# ca.update_clusters()
# ca.print_clusters()

# %%
ca.clusters

# %%
