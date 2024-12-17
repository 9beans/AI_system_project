import os
import math
from search_space import check_exception, make_op_info, Topo_search_space, HW_search_space
from simulation import make_topology, simulation_mth
from latency_modeling import get_latency_util

def get_divisors(num):
    divisors = []
    for i in range(1, int(num**0.5) + 1):  # Only iterate up to the square root of num
        if num % i == 0:  # If i divides num without a remainder
            divisors.append(i)
            if i != num // i:  # Avoid duplicate divisors (e.g., for perfect squares)
                divisors.append(num // i)
    divisors.sort()
    return divisors

NPU_name = "TPUv4"
simulation_threads = 64

base_scalesim_code_path = "./scale-sim-v2/scalesim/scale.py"
base_config_path = "./scale-sim-v2/project_configs"
base_topology_path = "./scale-sim-v2/project_topologies"
base_raw_data_path = "./scale-sim-v2/project_raw_data"
total_raw_data_fname = "total_raw_data.csv"

os.makedirs(base_config_path, exist_ok=True)
os.makedirs(base_topology_path, exist_ok=True)
os.makedirs(base_raw_data_path, exist_ok=True)

frequency = 250 * (10 ** 6)             # unit : MHz (Peak performance = 42 GMACS)
scratchpad_bw = 10                      # word(1 byte)/cycle per scratchpad
base_PE_num = 128 * 128                 # the number of PE in a baseline core(TPUv4 core)
mem_bw = scratchpad_bw * 3 * frequency  # word(1 byte)/cycle per scratchpad
vector_FLOPS = 1

interface_bw = 64 * (10 ** 9)           # 64 GB/s - PCIe4

n_batch = 100
n_token = 86        # Average sequence length of wikitext-2
micro_batch = 1
assert n_batch % micro_batch == 0, "micro_batch가 batch의 약수이어야 합니다."


        


# n_layer, n_head, d_head
# target_model = (12, 12, 64)                         # gpt2_small
target_model = (24, 16, 128)                        # MPT-1B redpajama
# target_model = (32, 32, 128)                        # gpt3_6.7B
LLM_dimension = {}
n_layer        = target_model[0]                    # 12
n_head         = target_model[1]                    # 12
d_head         = target_model[2]                    # 64
d_model        = target_model[1] * target_model[2]  # 768
intersize      = d_model * 4                        # 3072
n_token        = n_token
micro_batch    = micro_batch

Topology_search_space = []

max_core_scale = 8
max_card_scale = 8

n_core = 1
while n_core <= max_core_scale:
    for n_card in range(1, max_card_scale + 1):
        if (n_card != 1) & (n_core != max_core_scale):      # card를 scaleup 하기 전에 core부터 최대한 scaleup 하도록 설정
            continue
        # [TP, PP, DP], [H, M, K, N], [h, w, dataflow, i, f, o]
        card_parallelism_list, core_parallelism_list, total_hw_search_space = [], [], []
        Topo_search_space(n_card, n_core, card_parallelism_list, core_parallelism_list)
        HW_search_space(base_PE_num, total_hw_search_space)
        
        cfg_name_list = []
        
        for core_config in total_hw_search_space:
            cfg_name_list.append(f"{NPU_name}_{core_config[0]}_{core_config[1]}_{core_config[2]}_{core_config[3]}_{core_config[4]}_{core_config[5]}")
        
        for card_parallelism in card_parallelism_list:
            TP = card_parallelism[0]
            PP = card_parallelism[1]
            DP = card_parallelism[2]
            
            if (n_head % TP != 0):
                continue
            
            micro_batch_list = get_divisors(math.ceil(n_batch / DP))
            
            for micro_batch in micro_batch_list:
                topo_name_dict = {}
                card_workload_dim_dict = {}
                
                card_workload_dim_dict["n_micro_batch"] = micro_batch
                card_workload_dim_dict["n_token"]       = n_token
                card_workload_dim_dict["n_head"]        = int(n_head / TP)
                card_workload_dim_dict["d_head"]        = d_head
                card_workload_dim_dict["d_model"]       = d_model
                card_workload_dim_dict["intersize"]     = math.ceil(intersize / TP)
            
                for core_parallelism in core_parallelism_list:
                    op_info = make_op_info(core_parallelism, card_workload_dim_dict)
                    if op_info == None:
                        continue
                    if core_parallelism[2] != 1:        # accumulation dimension은 나누지 않는다 (EWADD 구현 복잡)
                        continue
                    
                    topo_name = f"TP-{TP}_M-{core_parallelism[1]}_K-{core_parallelism[2]}_N-{core_parallelism[3]}"
                    make_topology(topo_name, op_info)
                    
                    for core_config in total_hw_search_space:
                        cfg_name = f"{NPU_name}_{core_config[0]}_{core_config[1]}_{core_config[2]}_{core_config[3]}_{core_config[4]}_{core_config[5]}"
                        check = check_exception(op_info, core_config)
                        if check:
                            if cfg_name not in topo_name_dict:
                                topo_name_dict[cfg_name] = []
                            topo_name_dict[cfg_name].append(topo_name)
                        else:
                            continue
                        
                simulation_mth(cfg_name_list, topo_name_dict)
                print(f"{n_card}-card {n_core}-core {micro_batch}-micro_batch {TP}-{PP}-{DP}-card_parallelism simulation end")
    n_core *= 2