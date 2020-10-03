import numpy as np
from scipy.spatial import distance as dist
import matplotlib.pyplot as plt

embedding2D_data_path = "experiments/all_embeddings.csv"    # it is necessary to merge old & new embeddings
label_data_path = "experiments/all_labels.csv"              # it is necessary to merge old & new labels

new_users_list_in = []                                      # fill with user labels to insert in this statistic
new_users_list = np.unique(new_users_list_in)
least_rank_amounts = [1, 2, 3, 4, 5]


def load_embeddings2D_csv(file_name):
    embeddings = []
    with open(file_name) as f:
        for line in f:
            if line != "":
                temp = line.rstrip(" ]\n").lstrip("[ ").split(" ")
                line_as_list = np.float64(temp[0]), np.float64(temp[-1])
                embeddings.append(line_as_list)
    return embeddings


def load_labels(file_name):
    labels = []
    with open(file_name) as f:
        for line in f:
            if line != "":
                labels.append(line.split('\t')[0])
    return labels


def get_coordinates(user, labels, embeddings):
    return embeddings[(labels.index(str(user)))]


embeddings = load_embeddings2D_csv(embedding2D_data_path)
labels = load_labels(label_data_path)

x_axis = []
y_axis = []
bp = [[], [], [], [], []]
counters = [0, 0, 0, 0, 0]

for user in new_users_list:
    if user < 9999999:
        for amount in least_rank_amounts:
           x_axis.append(amount)
           coords_user = get_coordinates(user, labels, embeddings)
           coords_user_without_least_rank = get_coordinates(str(user)+str(amount), labels, embeddings)
           y_axis.append(dist.euclidean(coords_user, coords_user_without_least_rank))
           bp[amount-1].append(dist.euclidean(coords_user, coords_user_without_least_rank))

print(x_axis)
print(y_axis)

fig = plt.figure(figsize=(15, 10), facecolor='w')
axes = plt.subplot(1, 1, 1)
plt.title(''.format())
plt.ylabel('Euclidean distance')
plt.xlabel('Number of removed symbols')
plt.xticks(least_rank_amounts)
plt.axhline(y=13, color='r', linestyle='-')

box = axes.boxplot(bp)
plt.text(5.2, 13.25, 'Max Distance',
         horizontalalignment='center',
         verticalalignment='top',
         multialignment='center')

plt.show()

