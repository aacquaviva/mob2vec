import numpy as np
import timeit
import datetime
from gensim.models import Doc2Vec
from gensim.models.doc2vec import TaggedDocument
from utils import load_config, load_lab_seq_sp, labels_to_file


def assign_sequence_id(sequences):
    sequences_with_ids = []
    for idx, val in enumerate(sequences):
        sequence_id = "s_{}".format(idx)
        sequences_with_ids.append(TaggedDocument(val, [sequence_id]))
    return sequences_with_ids


config = load_config("config.json")

dim = config["embedding_dimensions"]
mode = config["mode"]
epochs = config["epochs"]
training_data_path = config["training_data"]
output_embedding_path = config["trained_embeddings_path"]
output_labels_path = config["labels"]
output_model_sym_path = config["model_sym_path"]
output_model_SP_path = config["model_sp_path"]

start_date_time = datetime.datetime.now()
start_time = timeit.default_timer()

print(str(datetime.datetime.now())+" Loading sequences in the form of items")
data_i_X, data_p_X, data_y = load_lab_seq_sp(training_data_path)

n_users = len(np.unique(data_y))
print("Unique users: {}".format(n_users))

print(str(datetime.datetime.now())+" Assigning a sequence id to each sequence")
data_seq_i = assign_sequence_id(data_i_X)
data_seq_p = assign_sequence_id(data_p_X)

print("First row visual check: \nLabel: {}\nSym: {}\nSPs: {}".format(data_y[0], data_seq_i[0], data_seq_p[0]))

if mode == "dm":
    mode_bool = 1
elif mode == "dbow":
    mode_bool = 0

print(str(datetime.datetime.now())+" Learning sequence vectors using Doc2Vec from symbols sequences")
d2v_i = Doc2Vec(vector_size=dim, min_count=0, workers=6, dm=mode_bool, epochs=epochs)
d2v_i.build_vocab(data_seq_i)
d2v_i.train(data_seq_i, total_examples=d2v_i.corpus_count, epochs=d2v_i.iter)
data_i_vec = [d2v_i.docvecs[idx] for idx in range(len(data_seq_i))]

print(str(datetime.datetime.now())+" Learning sequence vectors using Doc2Vec from SPs sequences")
d2v_p = Doc2Vec(vector_size=dim, min_count=0, workers=6, dm=mode_bool, epochs=epochs)
d2v_p.build_vocab(data_seq_p)
d2v_p.train(data_seq_p, total_examples=d2v_p.corpus_count, epochs=d2v_p.iter)
data_p_vec = [d2v_p.docvecs[idx] for idx in range(len(data_seq_p))]

print(str(datetime.datetime.now())+" Taking average of sequence vectors")
data_i_vec = np.array(data_i_vec).reshape(len(data_i_vec), dim)
data_p_vec = np.array(data_p_vec).reshape(len(data_p_vec), dim)
data_vec = (data_i_vec + data_p_vec) / 2

# uncomment to save trained embeddings
# embedding_to_file(data_vec, output_embedding_file)

labels_to_file(data_y, output_labels_path)

print("Saving trained models to file ")
d2v_i.delete_temporary_training_data(keep_doctags_vectors=True, keep_inference=True)
d2v_p.delete_temporary_training_data(keep_doctags_vectors=True, keep_inference=True)
d2v_i.save(output_model_sym_path)
d2v_p.save(output_model_SP_path)

end_date_time = datetime.datetime.now()
end_time = timeit.default_timer()
print("start date time: {} and end date time: {}".format(start_date_time, end_date_time))
print("runtime: {}(s)".format(round(end_time-start_time, 2)))
