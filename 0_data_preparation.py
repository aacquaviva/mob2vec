import pandas as pd
from utils import load_config


def get_loc_name_all(df_arg, id):
    return str(df_arg.iloc[int(id)-1].name)


def get_rank_from_sitename(df_arg, user_id, site_name):
    rank = 1
    try:
        while True:
            if str(df_arg.loc[int(user_id)][rank]).split(':')[0].rstrip(' ') == site_name:
                return rank
            else:
                rank = rank+1
    except KeyError:
        print("Cannot find rank for site_name {}".format(site_name))
        return -1


config = load_config("config.json")
csv = config['original_data_path']
csv_w_ranks = config['zone_ranks_path']
csv_w_mapping_name_id = config['location_area-id_mapping']
min_loc = config['minimum_location_repetition']
output_file = config['weekly_rank_trajectories']

df = pd.read_csv(csv, memory_map=True)
df_ranks = pd.read_csv(csv_w_ranks, memory_map=True, header=None, low_memory=False)
df_ranks.set_index(0, inplace=True)
df_loc_names_id = pd.read_csv(csv_w_mapping_name_id, memory_map=True, header=[0])
df_loc_names_id.set_index('label', inplace=True)

df_max_index = df.shape[0] - 1
limit = None
if limit is not None:
    df_max_index = max(df_max_index, limit)

loc = ""
week_n = df.iloc[0]['week_n']
last_user_id = df.iloc[0]['user_id']

with open(output_file, 'w', newline='\n') as f:
    n_loc = 0
    for i in range(0, int(df_max_index)):
        if i % 1000 == 0:
            print("Processing: {}%".format(round((i*100)/df_max_index, 2)))
        if week_n != df.iloc[i]['week_n'] or last_user_id != df.iloc[i]['user_id']:
            week_n = df.iloc[i]['week_n']

            if n_loc >= min_loc:
                f.write(str(last_user_id) + '\t' + loc + '\n')

            if last_user_id != df.iloc[i]['user_id']:
                last_user_id = df.iloc[i]['user_id']
            loc = ""
            n_loc = 0

        loc_name = get_loc_name_all(df_loc_names_id, df.iloc[i]['la_id'])
        rk = get_rank_from_sitename(df_ranks, last_user_id, loc_name)
        loc = loc + str(rk) + " "
        n_loc = n_loc + 1
