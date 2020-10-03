import numpy as np
from gensim.models import Doc2Vec
from utils import load_config, load_lab_seq_sp, embedding_to_file, labels_to_file

config = load_config("config.json")

dim = config["embedding_dimensions"]
model_sym_path = config["model_sym_path"]
model_SP_path = config["model_sp_path"]
training_data_file = config["training_data"]
embeddings_data_path = config["trained_embeddings_path"]
labels_data_path = config["labels"]

model_sym = Doc2Vec.load(model_sym_path)
model_SP = Doc2Vec.load(model_SP_path)

input_trajectories, input_sp, labels = load_lab_seq_sp(training_data_file)

labels_dict = dict.fromkeys(labels)
num_trajs = len(list(labels_dict))
print("Found {} unique user trajectories".format(num_trajs))

sum_vector = np.zeros(dim, dtype=np.float64)
index = 0
export_labels = []
total_labels = len(labels)

for label in labels:
    if index % 500 == 0:
        print("Evaluating traj {}/{} of user {}".format(index, total_labels, label))

    # makes random number (and embeddings) predictable
    model_sym.random.seed(0)
    model_SP.random.seed(0)

    # inference
    vector_sym = model_sym.infer_vector(input_trajectories[index], epochs=50)
    vector_SP = model_SP.infer_vector(input_sp[index], epochs=50)

    vector_symSP = np.array((vector_sym + vector_SP) / 2, dtype=np.float64)

    if index == 0:
        new_vectors = np.array([vector_symSP], dtype=np.float64)
    else:
        new_vectors = np.append(new_vectors, [vector_symSP], axis=0)

    index = index + 1
    export_labels.append(label)

last_label = labels[0]
export_embeddings = []
export_labels = []
counter = 0
week_counter = 0
sum_vector = [0] * dim

for vector in new_vectors:
    if labels[counter] != last_label:
        export_embeddings.append([i / week_counter for i in sum_vector])
        export_labels.append(last_label)
        week_counter = 1
        sum_vector = vector
    else:
        week_counter = week_counter + 1
        sum_vector = list(map(sum, zip(sum_vector, vector)))
    last_label = labels[counter]
    counter = counter + 1

export_embeddings.append([i / week_counter for i in sum_vector])
export_labels.append(last_label)

labels_to_file(export_labels, labels_data_path)
embedding_to_file((export_embeddings, embeddings_data_path))
