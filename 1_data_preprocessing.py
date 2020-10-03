import subprocess
from utils import load_config

# call to sqn2vec sp_miner
# https://github.com/nphdang/Sqn2Vec
def mine_sp(file_seq, minSup, gap, file_seq_sp):
    subprocess.run("sp_miner.exe -dataset {} -minsup {} -gap {} -seqsymsp {}".
                   format(file_seq, minSup, gap, file_seq_sp))


config = load_config("config.json")
in_seq = config["weekly_rank_trajectories"]
minSup = config["minSup"]
gap = config["gap"]
out_seq_sym_sp = config["training_data"]

mine_sp(in_seq, minSup, gap, out_seq_sym_sp)
