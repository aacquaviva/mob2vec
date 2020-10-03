import datetime
import timeit
import numpy as np
from scipy.spatial import distance as dist
import re
import matplotlib.pyplot as plt
import pandas as pd
from scipy import stats
from utils import load_config, load_labels


config = load_config("config.json")

mapping_loc_area_named_id_path = config["location_area-id_mapping_path"]
jensenshannon_distances = config["jensenshannon_distances_path"]
labels_data_path = config["labels"]
input_data_len = config["embedding_pairs_to_evaluate"]
distance = config["evaluation_distance"]
n_neighbors = config["umap_n_neighbors"]
min_dist = config["umap_min_dist"]

start_date_time = datetime.datetime.now()
start_time = timeit.default_timer()

labels = load_labels(labels_data_path)
n_users = len(labels)
n_embeddings = len(n_neighbors) * len(min_dist)
coords = np.empty((n_users, n_embeddings), dtype=tuple)

for neigh in n_neighbors:
    for j, min_d in enumerate(min_dist):
        umap_alt_embedding_path = umap_embedding_path = "data/umap_embeddings_{}_{}_{}.csv".format(neigh, dist, distance)
        print("Loading umap alt embedding... file={}".format(umap_alt_embedding_path))
        with open (umap_alt_embedding_path) as f_alt_umap:
            for user_index in range(input_data_len):
                coord_arr = [float(i) for i in re.sub("\s+", ' ',
                                                      f_alt_umap.readline().lstrip('[ ').rstrip(' ]\n')).split(' ')]
                coords[user_index][j] = tuple(coord_arr)

print("UMAP embeddings loaded")

couples_dict = {}
xDataJensenShannon = np.empty(int(((input_data_len*input_data_len) - input_data_len)/2) + 1, dtype=float)
yData = []
distances = np.empty((int(((input_data_len*input_data_len) - input_data_len)/2) + 1, n_embeddings), dtype=float)

print("\nLoading HUGE jensen-shannon distance csv")
df_jensenshannon_distances = pd.read_csv(jensenshannon_distances, memory_map=True, header=None)
df_jensenshannon_distances.columns = ['user_1', 'user_2', 'js']
df_jensenshannon_distances.drop_duplicates(keep="first", inplace=True)
df_jensenshannon_distances.set_index(['user_1','user_2'], inplace=True)
df_jensenshannon_distances.sort_index(inplace=True)
print(df_jensenshannon_distances.head())

for user_a in range(input_data_len):
    print("\ruser: {}/{} ".format(user_a + 1, input_data_len), end="", flush=True)
    for user_b in range(input_data_len):
        if user_a == user_b or labels[user_b] + "-" + labels[user_a] in couples_dict:
            continue

        for column in range(n_embeddings):
            distances[user_a][column] = dist.euclidean([coords[user_a][column], coords[user_a][column]],
                                                  [coords[user_b][column][0], coords[user_b][column][1]])

        if (int(labels[user_a][0]), int(labels[user_b][0])) in df_jensenshannon_distances.index:
            xDataJensenShannon[user_a] = df_jensenshannon_distances.loc[(int(labels[user_a]), int(labels[user_b]))].js
        else:
            xDataJensenShannon[user_a] = df_jensenshannon_distances.loc[(int(labels[user_b]), int(labels[user_a]))].js

        couples_dict.update({labels[user_a] + "-" + labels[user_b]: True})

print("Wait data plot...")

for j, neigh in enumerate(n_neighbors):
    for min_d in min_dist:
        print("Working on n_neigh={} minDist={}".format(neigh, min_d))
        p_corr, p_value = stats.pearsonr(xDataJensenShannon, distances[:, j])

        fig = plt.figure(figsize=(15, 10), facecolor='w')
        axes = plt.subplot(1, 1, 1)
        plt.title('r_pearson={} p-value={} N_users={} Neighbors={} MinDist={}\n'.format(p_corr, p_value, input_data_len,
                                                                                        neigh, min_d))
        plt.ylabel('Euclidean distance')
        plt.xlabel('Jensen Shannon distance')

        plt.scatter(xDataJensenShannon, distances[:, j], c="red", marker='o', s=3)

        plt.tight_layout()
        plt.savefig("plots/js-distance_n_users={}_neigh={}_minDist={}_p-corr={}_p-value={}.png".format(input_data_len,
                                                                                                       neigh, min_d,
                                                                                                       p_corr, p_value))
        plt.close(fig)

end_time = timeit.default_timer()
print("\nruntime: {}(s)".format(round(end_time - start_time, 2)))
