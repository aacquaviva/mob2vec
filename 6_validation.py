import random

import ed as ed
import pandas as pd
import numpy as np
import umap
from gensim.models import Doc2Vec
from pandas import Series
from scipy.spatial import distance as dist
import pickle
import warnings
from utils import load_config
from numba.core.errors import NumbaPerformanceWarning
warnings.filterwarnings("ignore", category=NumbaPerformanceWarning)

config = load_config("config.json")

raw_trajs_data_path = "data/weekly_rank_trajs.txt"
raw_100k_trajs_data_path = "data/100k_weekly_rank_trajs.txt"
raw_user_trajs_data_path = "data/full_rank_trajs.txt"
raw_user_100k_trajs_data_path = "data/100k_full_rank_trajs.txt"
mapping_sym_SPs_data_path = "data/mappingSymSp.txt"                 # simple txt with lines formatted "<ID>\t<SP>"
old_embeddings_data_path = config["trained_embeddings_path"]
old_labels_data_path = config["labels_path"]

# output
new_embeddings_data_path = "experiments/new_embeddings.csv"
new_labels_data_path = "experiments/new_labels.txt"
new_trajs_path = "experiments/new_weekly_rank_trajs.txt"

dim = config["embedding_dimensions"]
model_sym_file = config["model_sym_path"]
model_SP_file = config["model_sp_path"]

new_traj_source = "17k"                     # source dataset 17k | 100k
n_new_trajs = 1                             # number of new trajs
min_len = 13                                # min symbols per week (default=avg=13)
min_week_count = 9                          # min number of weeks (default=max=9)
sampling_rates = [0.9, 0.8, 0.7, 0.6, 0.5]
gap = 1                                     # gap-constraint (cit. sqn2vec)
traj_not_in_17k = False                     # request only trajs not in 17k | to use only in 100k mode
minimum_ranks_removal = True                # request to remove minimum ranks (least significant)
minimum_ranks = [1, 2, 3, 4, 5]             # amount of ranks to remove (from the least significant)

first_week_only = False
export_all_weeks = True
export_new_embeddings_to_csv = True
generate_new_trajs = False
save_reduced_embeddings = False
load_reduced_embeddings = False

# umap params
neighbors = 250
distance = 0.5

# analysis params
subset_size = 800
start_from_the_bottom = True
knn = 100

phase_1_create_traj_and_SP = True
phase_2_export_new_embeddings = True
phase_3_eval_downsampled_traj_rank = True


def check_if_present_in_df(id, dataframe):
    return id in dataframe.index


def get_traj_count(id, dataframe):
    return len(dataframe.loc[id])


def get_full_traj(id, dataframe):
    return "".join(list([dataframe.loc[id]][0].iloc[:, 0]))


def get_full_traj_list(id, dataframe):
    return ("".join(list([dataframe.loc[id]][0].iloc[:, 0]))).rstrip(" ").split(" ")


def get_specific_traj(id, dataframe, traj_n):
    return [dataframe.loc[id]][0].iloc[traj_n].traj


def get_specific_traj_len(id, dataframe, traj_n):
    return len([dataframe.loc[id]][0].iloc[traj_n].traj.rstrip(' \n').split(" "))


def check_traj_constraints(id, dataframe, traj_not_in_17k):
    if get_traj_count(id, dataframe) < min_week_count:
        return False
    for i in range(get_traj_count(id, dataframe)):
        if get_specific_traj_len(id, dataframe, i) < min_len:
            return False
    if traj_not_in_17k:
        return not check_if_present_in_df(id, df_raw_trajs)
    else:
        return True


def import_mapping(filename):
    mapping_dict = dict()
    with open(filename, mode='r', encoding="utf-8") as f:
        for line in f:
            item_id, item = line.rstrip(" \n").split("\t")
            if not ' ' in item:
                item = [item]
            else:
                item = item.split(" ")
            mapping_dict[int(item_id)] = item
    return mapping_dict


def downsample_traj(traj, percentage):
    export_traj = []
    for sym in traj.rstrip(" \n").split(" "):
        if random.random() < percentage:
            export_traj.append(sym)
            export_traj.append(" ")
    return "".join(export_traj)


def get_max_rank_values(user_id, amount):
    values = []
    if check_if_present_in_df(user_id, df_raw_user_trajs):
        correct_df = df_raw_user_trajs
    elif check_if_present_in_df(user_id, df_raw_100k_user_trajs):
        correct_df = df_raw_100k_user_trajs
    else:
        return 0
    row = correct_df.loc[user_id].traj
    if isinstance(row, Series):
        row = row.tolist()
        max_len = max([len(x) for x in row])
        for j, sequence in enumerate(row):
            if len(sequence) == max_len:
                t = list(map(lambda x: int(x), row[j].rstrip(' \n').split(' ')))
    else:
        t = list(map(lambda x: int(x), row.rstrip(' \n').split(' ')))
    for iter in range(amount):
        max_t = max(t)
        values.append(max_t)
        t = list(filter(lambda x: x != max_t, t))
    return values


def remove_least_significant_symbols(id, traj, amount):
    export_traj = []
    traj_as_list = traj.rstrip(" \n").split(" ")
    max_rank_values = get_max_rank_values(id, amount)
    for sym in traj_as_list:
        if int(sym) not in max_rank_values:
            export_traj.append(sym)
            export_traj.append(" ")
    return "".join(export_traj)


def find_SPs_withGap(t, SPs, gap):
    sym_present = True
    if SPs[0] in t:
        t = t[t.index(SPs[0]):]
    else:
        return False
    for SP in SPs:
        sym_present = sym_present and SP in t
    if sym_present and len(t) >= len(SPs):
        found = True
        right_index = 0
        for i in range(0, len(SPs) - 1):
            right_index = max(right_index, t.index(SPs[i + 1]))
            if t.index(SPs[i]) > t.index(SPs[i + 1]):
                found = False and found
            else:
                if t.index(SPs[i]) - t.index(SPs[i + 1]) == 0:
                    found = True and found
                elif (t.index(SPs[i + 1]) - t.index(SPs[i])) - 1 in range(0, gap + 1):
                    if (t.index(SPs[i + 1]) - t.index(SPs[i])) - 1 in [0, 1]:
                        gap = 0
                    else:
                        gap = gap - 1
                    found = True and found
                else:
                    found = False and found
        if found:
            return True
        else:
            return find_SPs_withGap(t[(right_index - len(SPs)):], SPs, gap)
    else:
        return False


def find_SPs(t):
    export_SPs = []
    for entry in range(1, len(mapping_dict)):
        sep = ' '
        sp_to_find = sep.join(list(map(lambda x: str(x), mapping_dict[entry])))
        sep = ' '
        traj_to_explore = sep.join(list(map(lambda x: str(x), t)))
        if " " + sp_to_find + " " in " " + traj_to_explore + " ":
            export_SPs.append(str(entry))
    return export_SPs


def embedding_to_file(embeddings, filename):
    with open(filename, 'w', encoding="utf-8") as f:
        for vector in embeddings:
            f.write("{}".format(vector[0]))
            for index in range(1, dim):
                f.write(",{}".format(vector[index]))
            f.write("\n")


def labels_to_file(labels, file_name):
    with open(file_name, 'w', encoding="utf-8") as f:
        for l in labels:
            f.write("{}\t1\n".format(l))


df_raw_trajs = pd.read_csv(raw_trajs_data_path, sep="\t", header=None, memory_map=True)
df_raw_100k_trajs = pd.read_csv(raw_100k_trajs_data_path, sep="\t", header=None, memory_map=True)
df_raw_trajs.columns, df_raw_100k_trajs.columns = ['id', 'traj'], ['id', 'traj']
df_raw_100k_trajs.astype({'id': 'int32'})
df_raw_trajs.set_index('id', inplace=True)
df_raw_100k_trajs.set_index('id', inplace=True)

df_raw_user_trajs = pd.read_csv(raw_user_trajs_data_path, sep="\t", header=None, memory_map=True)
df_raw_100k_user_trajs = pd.read_csv(raw_user_100k_trajs_data_path, sep="\t", header=None, memory_map=True)
df_raw_user_trajs.columns, df_raw_100k_user_trajs.columns = ['id', 'traj'], ['id', 'traj']
df_raw_user_trajs.set_index('id', inplace=True)
df_raw_100k_user_trajs.set_index('id', inplace=True)

new_trajs = []
new_trajs_labels = []

max_l = 0
sum_l = 0

new_trajs_to_export = []
constraints_compliant_labels = []

if phase_1_create_traj_and_SP:

    if generate_new_trajs:
        print("17K head\n{}\nshape\n{}".format(df_raw_trajs.head(), df_raw_trajs.shape))
        print("100K head\n{}\nshape\n{}".format(df_raw_100k_trajs.head(), df_raw_100k_trajs.shape))

        if new_traj_source == "17k":
            df = df_raw_trajs
        else:
            df = df_raw_100k_trajs

        traj_counter = 0
        df_new_trajs = pd.DataFrame(columns=['id', 'traj'])

        for index, id in enumerate(np.unique(df.index)):
            if traj_counter == n_new_trajs:
                break
            if check_traj_constraints(id, df, traj_not_in_17k):
                constraints_compliant_labels.append(id)
                traj_counter = traj_counter + 1
                if export_all_weeks:
                    if new_traj_source == "100k":
                        for week_i in range(get_traj_count(id, df)):
                            new_trajs_labels.append(id)
                            new_trajs.append(get_specific_traj(id, df, week_i))
                    if minimum_ranks_removal:
                        for i, amount in enumerate(minimum_ranks):
                            for week_i in range(get_traj_count(id, df)):
                                new_trajs_labels.append(int(str(id) + str(amount)))
                                new_trajs.append(
                                    remove_least_significant_symbols(id, get_specific_traj(id, df, week_i), amount))
                                df_new_trajs = df_new_trajs.append(
                                    pd.DataFrame({"id": [new_trajs_labels[-1]], "traj": [new_trajs[-1]]}))
                    else:
                        for i, perc in enumerate(sampling_rates):
                            for week_i in range(get_traj_count(id, df)):
                                downsampling_id = str(perc).split('.')[1]
                                new_trajs_labels.append(int(str(id) + downsampling_id))
                                new_trajs.append(downsample_traj(get_specific_traj(id, df, week_i), perc))
                                df_new_trajs = df_new_trajs.append(
                                    pd.DataFrame({"id": [new_trajs_labels[-1]], "traj": [new_trajs[-1]]}))

                else:
                    new_trajs_labels.append(id)
                    if first_week_only:
                        new_trajs.append(get_specific_traj(new_trajs_labels[len(new_trajs_labels) - 1], df, 0))
                    else:
                        new_trajs.append(get_full_traj(new_trajs_labels[len(new_trajs_labels) - 1], df))
                    if minimum_ranks_removal == True:
                        for i, amount in enumerate(minimum_ranks):
                            # print("Working with id={} i={} perc={}".format(id, i, perc))
                            new_trajs_labels.append(int(str(id) + str(amount)))
                            if first_week_only:
                                new_trajs.append(remove_least_significant_symbols(id, get_full_traj(
                                    new_trajs_labels[len(new_trajs_labels) - i - 2], df), amount))
                            else:
                                new_trajs.append(remove_least_significant_symbols(id, get_specific_traj(
                                    new_trajs_labels[len(new_trajs_labels) - i - 2], df, 0), amount))

                    else:
                        for i, perc in enumerate(sampling_rates):
                            downsampling_id = str(perc).split('.')[1]
                            new_trajs_labels.append(int(str(id) + downsampling_id))
                            if first_week_only:
                                new_trajs.append(
                                    downsample_traj(get_full_traj(new_trajs_labels[len(new_trajs_labels) - i - 2], df),
                                                    perc))
                            else:
                                new_trajs.append(downsample_traj(
                                    get_specific_traj(new_trajs_labels[len(new_trajs_labels) - i - 2], df, 0), perc))


        print("Found this trajs: {}".format(new_trajs_labels))
        print("Constraint compliant labels: {}".format(constraints_compliant_labels))

        new_trajs = list(map(lambda x: x.rstrip(" \n").split(" "), new_trajs))

        df_new_trajs.set_index('id', inplace=True)

        print("New trajs in df\n{}".format(df_new_trajs.tail()))

        for i, traj in enumerate(new_trajs_to_export):
            print("{} {}\n".format(new_trajs_labels[i], traj))

    else:
        df_new_trajs = pd.DataFrame(columns=['id', 'traj'])

        with open(new_trajs_path, 'r') as f_input:
            for line in f_input:
                lab, t = line.rstrip(" \n").split("\t")
                new_trajs_labels.append(int(str(lab) + "0"))
                new_trajs.append(t)
                df_new_trajs = df_new_trajs.append(
                    pd.DataFrame({"id": [new_trajs_labels[-1]], "traj": [new_trajs[-1]]}))

        new_trajs = list(map(lambda x: x.rstrip(" \n").split(" "), new_trajs))
        df_new_trajs.set_index('id', inplace=True)

    mapping_dict = import_mapping(mapping_sym_SPs_data_path)

    new_trajs_SPs = []
    for i, traj in enumerate(new_trajs):
        traj_sp = find_SPs(traj, gap)
        new_trajs_SPs.append(traj_sp)

    print("Example trajs")
    for i in range(100):
        print("Label: {}\n Sym: {}\n SP: {}".format(new_trajs_labels[i], new_trajs[i], new_trajs_SPs[i]))

    # new_traj_labels   -> original traj label + downsampled traj labels
    # new_traj          -> original traj + downsampled traj
    # new_trajs_SPs     -> original traj SP + downsampled traj SP

# -----------------------------------------------------------------------------------------------------------------

if phase_2_export_new_embeddings:
    new_embeddings = []
    model_sym = Doc2Vec.load(model_sym_file)
    model_SP = Doc2Vec.load(model_SP_file)

    for i, label in enumerate(new_trajs_labels):
        # print("Evaluating traj of user {}".format(label))
        model_sym.random.seed(0)
        vector_sym = model_sym.infer_vector(new_trajs[i], epochs=50)
        model_SP.random.seed(0)
        vector_SP = model_SP.infer_vector(new_trajs_SPs[i], epochs=50)

        vector_symSP = np.array((vector_sym + vector_SP) / 2, dtype=np.float64)

        if i == 0:
            new_vectors = np.array([vector_symSP], dtype=np.float64)
        else:
            new_vectors = np.append(new_vectors, [vector_symSP], axis=0)
        new_embeddings.append(vector_symSP)

    if export_new_embeddings_to_csv:
        embedding_to_file(new_embeddings, new_embeddings_data_path)
        labels_to_file(new_trajs_labels, new_labels_data_path)

    print("\nStats: \nnew_trajs={}\nnew_trajs_SPs={}\nnew_trajs_labels={}".format(len(new_trajs), len(new_trajs_SPs),
                                                                                  len(new_trajs_labels)))

if phase_3_eval_downsampled_traj_rank:

    last_label = new_trajs_labels[0]
    export_embeddings = []
    export_labels = []
    counter = 0
    week_counter = 0
    sum_vector = [0] * dim

    for vector in new_embeddings:
        if new_trajs_labels[counter] != last_label:
            export_embeddings.append([i / week_counter for i in sum_vector])
            export_labels.append(last_label)
            week_counter = 1
            sum_vector = vector
        else:
            week_counter = week_counter + 1
            sum_vector = list(map(sum, zip(sum_vector, vector)))
        last_label = new_trajs_labels[counter]
        counter = counter + 1
    export_embeddings.append([i / week_counter for i in sum_vector])
    export_labels.append(last_label)

    old_embeddings_df = pd.read_csv(old_embeddings_data_path, sep=",", header=None, memory_map=True, dtype=np.float64)
    old_labels_df = pd.read_csv(old_labels_data_path, sep="\t", header=None, memory_map=True, dtype=np.int32)

    print("Shape old\n{}\n{}".format(old_embeddings_df.shape, old_labels_df.shape))

    new_trajs_df = pd.DataFrame(export_embeddings)
    new_trajs_labels_df = pd.DataFrame(export_labels, dtype=np.int32)

    print("Shape\n{}\n{}".format(new_trajs_df.shape, new_trajs_labels_df.shape))

    merged_embeddings_df = pd.concat([old_embeddings_df, new_trajs_df])
    merged_labels_df = pd.concat([old_labels_df, new_trajs_labels_df])

    print("Shape merged embeddings{}\nShape merged labels{}".format(merged_embeddings_df.shape, merged_labels_df.shape))

    merged_embeddings_np = merged_embeddings_df.to_numpy(dtype=np.float64)

    if save_reduced_embeddings:
        print("Dim redux...")
        embedding_umap = umap.UMAP(n_neighbors=neighbors, min_dist=distance, metric='euclidean', n_components=2)
        data_embedded_umap = embedding_umap.fit_transform(merged_embeddings_df.to_numpy())
        with open('tmp_embeddings.txt', 'wb') as fp:
            pickle.dump(data_embedded_umap, fp)
    elif load_reduced_embeddings:
        print("Loading dim reduced embeddings...")
        with open('tmp_embeddings.txt', 'rb') as fp:
            data_embedded_umap = pickle.load(fp)

    print("Reduced \n{}".format(data_embedded_umap))
    print("Shape \n{}".format(data_embedded_umap.shape))

    embeddings_count = data_embedded_umap.shape[0]

    if start_from_the_bottom:
        traj_index_list = range(embeddings_count - subset_size, embeddings_count)
    else:
        traj_index_list = range(subset_size)

    umap_distance = []
    edit_distance = []
    labels = []

    for traj1 in traj_index_list:
        lab1 = int(merged_labels_df.iloc[traj1][0])
        if lab1 not in np.unique(new_trajs_labels) or lab1 > 9999999:
            continue
        print("Processing traj {} - label {}".format(traj1, lab1))
        labels.append(lab1)

        if lab1 in df_raw_trajs.index:
            full_traj1 = get_full_traj_list(lab1, df_raw_trajs)
        elif lab1 in df_raw_100k_trajs.index:
            full_traj1 = get_full_traj_list(lab1, df_raw_100k_trajs)
        else:
            full_traj1 = get_full_traj_list(lab1, df_new_trajs)

        umap_distance_temp = []
        edit_distance_temp = []
        for traj2 in traj_index_list:
            lab2 = int(merged_labels_df.iloc[traj2][0])
            umap_distance_temp.append((dist.euclidean([data_embedded_umap[traj1][0], data_embedded_umap[traj1][1]],
                                                      [data_embedded_umap[traj2][0], data_embedded_umap[traj2][1]]),
                                       lab2))
            if merged_labels_df.iloc[traj2][0] in df_raw_trajs.index:
                full_traj2 = get_full_traj_list(lab2, df_raw_trajs)
            elif merged_labels_df.iloc[traj2][0] in df_raw_100k_trajs.index:
                full_traj2 = get_full_traj_list(lab2, df_raw_100k_trajs)
            else:
                full_traj2 = get_full_traj_list(lab2, df_new_trajs)
            edit_distance_temp.append((ed.SequenceMatcher(a=full_traj1, b=full_traj2).distance(), lab2))

        edit_distance.append(sorted(edit_distance_temp))
        umap_distance.append(sorted(umap_distance_temp))

    for i, lab in enumerate(labels):
        rank_ed = 0
        rank_umap = 0
        for index, tup in enumerate(edit_distance[i][0:knn]):
            if tup[1] == int(str(lab) + "9"):
                break
            else:
                rank_ed = rank_ed + 1
        for index, tup in enumerate(umap_distance[i][0:knn]):
            if tup[1] == int(str(lab) + "9"):
                break
            else:
                rank_umap = rank_umap + 1
        print("User {} -> Rank EditDistance = {} | Rank Umap = {}".format(lab, rank_ed, rank_umap))
        print("{}\n{}\n--------".format(edit_distance[i], umap_distance[i]))
