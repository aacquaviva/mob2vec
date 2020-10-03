import timeit
import datetime
import umap
import warnings
import csv
from utils import load_config, embedding_to_file
from numba.core.errors import NumbaPerformanceWarning
warnings.filterwarnings("ignore", category=NumbaPerformanceWarning)


def load_embeddings(file_name):
    with open(file_name) as f:
        csv_reader = csv.reader(f)
        embeddings = list(csv_reader)
    return embeddings


config = load_config("config.json")

embeddings_data_path = config["trained_embeddings_path"]
n_dim = config["umap_dimensions"]
distance = config["umap_distance"]
n_neighbors = config["umap_n_neighbors"]
min_dist = config["umap_min_dist"]

start_date_time = datetime.datetime.now()
start_time = timeit.default_timer()

data_vec = load_embeddings(embeddings_data_path)

print(data_vec[0])

for neigh in n_neighbors:
    for dist in min_dist:
        umap_embedding_path = "{}/umap_embedings_{}_{}_{}.csv".format(umap_embedding_folder, neigh, dist, distance)
        embedding_umap = umap.UMAP(n_neighbors=neigh, min_dist=dist, metric=distance, n_components=n_dim)
        data_embedded_umap = embedding_umap.fit_transform(data_vec)
        print("{} Writing UMAP embedding w\ n_neighbors={} min_dist={} distance={}".format(datetime.datetime.now(), neigh, dist, distance))
        embedding_to_file(data_embedded_umap, umap_embedding_path)
        del embedding_umap
        del data_embedded_umap

end_date_time = datetime.datetime.now()
end_time = timeit.default_timer()
print("start date time: {} and end date time: {}".format(start_date_time, end_date_time))
print("runtime: {}(s)".format(round(end_time-start_time, 2)))
