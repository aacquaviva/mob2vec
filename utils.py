import json
import csv


def load_config(json_path):
    with open(json_path, 'r') as openfile:
        json_object = json.load(openfile)
        return json_object


def load_lab_seq_sp(file_name):
    labels, sequences , sp = [], [], []
    with open(file_name) as f:
        for line in f:
            label, symbols, patterns = line.split("\t")
            if symbols != "\n":
                labels.append(label)
                sequences.append(symbols.rstrip().split(" "))
                sp.append(patterns.rstrip().split(" "))
    return sequences, sp, labels


def embedding_to_file(vector_list, filename):
    with open(filename, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerows(vector_list)


def labels_to_file(labels, file_name):
    with open(file_name, 'w', encoding="utf-8") as f:
        for l in labels:
            f.write("{}\n".format(l))


def load_labels(file_name):
    labels_list = []
    with open(file_name) as f:
        for line in f:
            labels_list.append(int(line))
    return labels_list