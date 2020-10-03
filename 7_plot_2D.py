import matplotlib.pyplot as plt
import re
from utils import load_config, load_labels


def import_embedding(file_name, limit=-1):
    count = 0
    x, y = [], []
    with open(file_name) as f:
        for line in f:
            count = count + 1
            if count == limit:
                return x, y
            clean_line = line.replace("[", '').replace("]",'').replace("\n",'')
            if "  " in line:
                xy = re.split(' ', clean_line)
            else:
                xy = re.split(' ', clean_line)
            b = 0

            for i in range(0,len(xy)):
                if xy[i-b] == '':
                    del xy[i-b]
                    b = b+1
            x.append(float(xy[0]))
            y.append(float(xy[1]))
    return x, y


config = load_config("config.json")

users_limit = 150000
print_user_id = False
users_to_print = []         # fill user labels to print next to 2d point

fig_size = (20, 20)
text_size = 10

labels_data_path = config["labels"]
umap_embedding_folder = config["umap_embedding_folder"]
distance = config["umap_distance"]
n_neighbors = config["umap_n_neighbors"]
min_dist = config["umap_min_dist"]


if print_user_id:
    labels = load_labels(labels_data_path)

for neigh in n_neighbors:
    for dist in min_dist:
        umap_embedding_path = "{}/embedding_umap_{}_{}_{}.csv".format(umap_embedding_folder, neigh, dist, distance)
        X_embedded_umap, Y_embedded_umap = import_embedding(umap_embedding_path, limit=users_limit)
        fig = plt.figure(figsize=fig_size, facecolor='w')
        plt.subplot(1, 1, 1)
        plt.scatter(X_embedded_umap, Y_embedded_umap, cmap = 'hsv')
        plt.title('2D embedding using UMAP Embedding n_neighbors={} min_dist={} distance={}\n'.format(neigh, dist, distance))
        plt.xlabel('Feature 1')
        plt.ylabel('Feature 2')
        if print_user_id:
            for i, lab in enumerate(labels):
                if lab in users_to_print:
                    print(lab)
                    plt.text(X_embedded_umap[i], Y_embedded_umap[i], lab, fontsize=text_size)
                    plt.scatter(X_embedded_umap[i], Y_embedded_umap[i], c='red', s=20)

        plt.tight_layout()
        path = "{}/embedding_umap_{}_{}_{}.png".format(umap_embedding_folder, neigh, dist, distance)
        plt.savefig(path)
        print("Saving figure in {}".format(path))
        plt.close(fig)
