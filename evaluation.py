import pandas as pd
import numpy as np
import math
import pickle
from search_space import Topo_search_space
from plot import scale_systolic_dim_graph

def find_best_config(temp_result_):
    best_config = ""
    best_latency = float("inf")
    for config, result in temp_result_.items():
        if best_latency > result[2]:
            best_latency = result[2]
            best_config = config
    return best_config

def make_config_dict(cfg_name_):
    config_dict = {}
    card_core, core_config, topo_config = cfg_name_.split(" \t ")
    config_dict["card"], config_dict["core"] = [i.split("-")[1] for i in card_core.split("_")]
    config_dict["sh"], config_dict["sw"], config_dict["dataflow"] = core_config.split("_")
    config_dict["TP"], config_dict["PP"], config_dict["DP"], config_dict["mb"], config_dict["H"], config_dict["M"], config_dict["K"], config_dict["N"] = [i.split("-")[1] for i in topo_config.split("_")]

    return config_dict

# core config 별 best를 뽑아서 plot을 하고
def plot_core_config(systolic_dim_results_):
    print()

# <max scale = 64> best core config에 대해서 card parallelism에 대한 그래프를 그리고
def plot_card_parallelism(card_parallelism_list_):
    print()

# <max scale = 64>  best core config와 best card parallelism에 대해서 core parallelism에 대한 그래프를 그린다다
def plot_core_parallelism(core_parallelism_list_):
    print()

n_batch = 100

# n_layer, n_head, d_head, model_name
target_model = (12, 12, 64, "GPT2_S")
# target_model = (24, 16, 128, "MPT_1B_red_pajama")
# target_model = (32, 32, 128, "GPT3_7B")
base_raw_data_path = f"./scale-sim-v2/project_raw_data/{target_model[3]}"

with open(f'{base_raw_data_path}/results.pkl', 'rb') as file:
    total_result_dict = pickle.load(file)
    
card_parallelism_list, core_parallelism_list = [], []
Topo_search_space(8, 8, card_parallelism_list, core_parallelism_list)

total_result_df_dict = {}   # {scale: df}
scale_list = []

for card_core_config, result_dict in total_result_dict.items():     # scale 별 dictionary를 dataframe으로 변환
    best_config = find_best_config(result_dict)
    best_config_dict = make_config_dict(best_config)
    n_search_space = len(result_dict)
        
    n_card = card_core_config[0]
    n_core = card_core_config[1]
    scale = n_card * n_core
    df_row = []
    for cfg, result in result_dict.items():
        config_dict = make_config_dict(cfg)
        config_dict["cfg_name"] = cfg
        config_dict["latency"] = result[2]
        df_row.append(config_dict)
    
    total_result_df_dict[scale] =  pd.DataFrame(df_row)
    scale_list.append(scale)
    
scale_systolic_dim_results = {}
for scale in scale_list:
    df = total_result_df_dict[scale]
    df.loc[:, (df.columns != 'dataflow') & (df.columns != 'cfg_name') & (df.columns != 'latency')] = df.loc[:, (df.columns != 'dataflow') & (df.columns != 'cfg_name') & (df.columns != 'latency')].astype(int)
    
    systolic_dim_results = {
                                # (1, 16384) : {"ws" : [None, float("inf")], "os" : [None, float("inf")]},
                                # (2, 8192) : {"ws" : [None, float("inf")], "os" : [None, float("inf")]},
                                # (4, 4096) : {"ws" : [None, float("inf")], "os" : [None, float("inf")]},
                                # (8, 2048) : {"ws" : [None, float("inf")], "os" : [None, float("inf")]},
                                # (16, 1024) : {"ws" : [None, float("inf")], "os" : [None, float("inf")]},
                                (32, 512) : {"ws" : [None, float("inf")], "os" : [None, float("inf")]},
                                (64, 256) : {"ws" : [None, float("inf")], "os" : [None, float("inf")]},
                                (128, 128) : {"ws" : [None, float("inf")], "os" : [None, float("inf")]},
                                (256, 64) : {"ws" : [None, float("inf")], "os" : [None, float("inf")]},
                                (512, 32) : {"ws" : [None, float("inf")], "os" : [None, float("inf")]},
                                # (1024, 16) : {"ws" : [None, float("inf")], "os" : [None, float("inf")]},
                                # (2048, 8) : {"ws" : [None, float("inf")], "os" : [None, float("inf")]},
                                # (4096, 4) : {"ws" : [None, float("inf")], "os" : [None, float("inf")]},
                                # (8192, 2) : {"ws" : [None, float("inf")], "os" : [None, float("inf")]},
                                # (16384, 1) : {"ws" : [None, float("inf")], "os" : [None, float("inf")]},
                            }
    
    ws_rows = df[df['dataflow'] == 'ws']
    os_rows = df[df['dataflow'] == 'os']
    target_systolic_dim_list = list(systolic_dim_results.keys())
    
    for systolic_dim in target_systolic_dim_list:
        filtered_ws_rows = ws_rows[(ws_rows['sh'] == systolic_dim[0]) & (ws_rows['sw'] == systolic_dim[1])].copy()
        filtered_os_rows = os_rows[(os_rows['sh'] == systolic_dim[0]) & (os_rows['sw'] == systolic_dim[1])].copy()
        
        error_margin = 0.000000001 # ns
        filtered_ws_rows.loc[:, 'latency'] = (filtered_ws_rows['latency'] / error_margin).round() * error_margin
        filtered_os_rows.loc[:, 'latency'] = (filtered_os_rows['latency'] / error_margin).round() * error_margin
        
        filtered_ws_rows = filtered_ws_rows.sort_values(by=['latency', 'mb'], ascending=[True, False])
        filtered_os_rows = filtered_os_rows.sort_values(by=['latency', 'mb'], ascending=[True, False])
        min_latency_ws_row = filtered_ws_rows.head(1)
        min_latency_os_row = filtered_os_rows.head(1)
        
        systolic_dim_results[systolic_dim]['ws'][0] = min_latency_ws_row
        systolic_dim_results[systolic_dim]['ws'][1] = min_latency_ws_row['latency'].values[0]
        systolic_dim_results[systolic_dim]['os'][0] = min_latency_os_row
        systolic_dim_results[systolic_dim]['os'][1] = min_latency_os_row['latency'].values[0]
        
        # print(min_latency_ws_row)
        # print(min_latency_os_row)
        
    scale_systolic_dim_results[scale] = systolic_dim_results
    
scale_systolic_dim_graph(scale_systolic_dim_results)
breakpoint()

scale_core_parallelism_results = {}
# core parallelism graph
for scale in [8, 16, 32, 64]:
    df = total_result_df_dict[scale]
    df.loc[:, (df.columns != 'dataflow') & (df.columns != 'cfg_name') & (df.columns != 'latency')] = df.loc[:, (df.columns != 'dataflow') & (df.columns != 'cfg_name') & (df.columns != 'latency')].astype(int)
    
    core_parallelism_results = {}
    for core_parallelism in core_parallelism_list:
        core_parallelism_results[core_parallelism] = [None, float("inf")]
    
    
    
    fixed_systolic_dim_df = df[(df['sh'] == 128) & (df['sw'] == 128)].copy()
    fixed_systolic_dim_df = fixed_systolic_dim_df.sort_values(by=['latency', 'H', 'M', 'N'], ascending=[True, True, True, True])
    
    # [H, M, K, N]
    for core_parallelism in core_parallelism_list:
        filtered_rows = fixed_systolic_dim_df[(fixed_systolic_dim_df['H'] == core_parallelism[0]) & (fixed_systolic_dim_df['M'] == core_parallelism[1]) & (fixed_systolic_dim_df['N'] == core_parallelism[3])]
        filtered_rows = filtered_rows.sort_values(by=['latency', 'mb'], ascending=[True, False])
        
        min_latency_row = filtered_ws_rows.head(1)
        
        systolic_dim_results[core_parallelism][0] = min_latency_row
        systolic_dim_results[core_parallelism][1] = min_latency_row['latency'].values[0]
        
        print(min_latency_row)
        
    scale_core_parallelism_results[scale] = systolic_dim_results

breakpoint()
            