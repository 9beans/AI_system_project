import os
import math
from search_space import check_exception, make_op_info, Topo_search_space, HW_search_space
from simulation import make_topology, simulation_mth

def get_divisors(num):
    divisors = []
    for i in range(1, int(num**0.5) + 1):  # Only iterate up to the square root of num
        if num % i == 0:  # If i divides num without a remainder
            divisors.append(i)
            if i != num // i:  # Avoid duplicate divisors (e.g., for perfect squares)
                divisors.append(num // i)
    divisors.sort()
    return divisors


simulation_threads = 64

base_scalesim_code_path = "./scale-sim-v2/scalesim/scale.py"
base_config_path = "./scale-sim-v2/project_configs"
base_topology_path = "./scale-sim-v2/project_topologies"
base_raw_data_path = "./scale-sim-v2/project_raw_data"
total_raw_data_fname = "total_raw_data.csv"

os.makedirs(base_config_path, exist_ok=True)
os.makedirs(base_topology_path, exist_ok=True)
os.makedirs(base_raw_data_path, exist_ok=True)

def get_latency(op_info_, core_FLOPS_, BW_, n_card_):
    op_type = op_info_[0]
    H = op_info_[1]
    M = op_info_[2]
    K = op_info_[3]
    N = op_info_[4]
    
    if op_type in ["FC", "Attn"]:
        1
    elif op_type == "Softmax":
        SM_FLOPs = 1
        return (SM_FLOPs / core_FLOPS_, 0)
        
    elif op_type == "AllReduce":
        AR_FLOPs = 1
        
        T_comm = 1
        T_ewadd = LN_FLOPs / core_FLOPS_
        return (T_comm + T_ewadd, 0)
    elif op_type == "Residual":
        RD_FLOPs = 1
        return (RD_FLOPs / core_FLOPS_, 0)
    else:   # Layernorm
        LN_FLOPs = 1
        return (LN_FLOPs / core_FLOPS_, 0)
        
        

frequency = 250 * (10 ** 6)             # unit : MHz (Peak performance = 42 GMACS)
scratchpad_bw = 10                      # word(1 byte)/cycle per scratchpad
base_PE_num = 168                       # the number of PE in a baseline core(eyeriss core)
mem_bw = scratchpad_bw * 3 * frequency  # word(1 byte)/cycle per scratchpad

interface_bw = 64 * (10 ** 9)           # 64 GB/s - PCIe4

n_batch = 100
n_token = 86        # Average sequence length of wikitext-2
micro_batch = 1
assert n_batch % micro_batch == 0, "micro_batch가 batch의 약수이어야 합니다."

# n_layer, n_head, d_head
target_model = (12, 12, 64)                         # gpt2_small
LLM_dimension = {}
n_layer        = target_model[0]                    # 12
n_head         = target_model[1]                    # 12
d_head         = target_model[2]                    # 64
d_model        = target_model[1] * target_model[2]  # 768
intersize      = d_model * 4                        # 3072
n_token        = n_token
micro_batch    = micro_batch

transformer_operation = {               # (Op type  , H                        , M         , K         , N)
                            "QKV"       : ("FC"     , micro_batch              , n_token   , d_model   , d_model * 3),
                            "QKT"       : ("Attn"   , n_head * micro_batch     , n_token   , d_head    , n_token),
                            "Softmax"   : ("else"   , n_head * micro_batch     , n_token   , 1         , n_token),
                            "SV"        : ("Attn"   , n_head * micro_batch     , n_token   , n_token   , d_head),
                            "Proj"      : ("FC"     , micro_batch              , n_token   , d_model   , d_model),
                            "AllReduce" : ("else"   , n_token * micro_batch    , d_model   , 1         , 1),       
                            "Residual"  : ("else"   , n_token * micro_batch    , d_model   , 1         , 1),       
                            "Layernorm" : ("else"   , n_token * micro_batch    , d_model   , 1         , 1),       
                            "FFN1"      : ("FC"     , micro_batch              , n_token   , d_model   , intersize),
                            "FFN2"      : ("FC"     , micro_batch              , n_token   , intersize , d_model),
                            "Residual"  : ("else"   , n_token * micro_batch    , d_model   , 1         , 1),
                            "Layernorm" : ("else"   , n_token * micro_batch    , d_model   , 1         , 1),
                            "AllReduce" : ("else"   , n_token * micro_batch    , d_model   , 1         , 1)
                        }

Topology_search_space = []

max_core_scale = 128
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
            cfg_name_list.append(f"eyeriss_{core_config[0]}_{core_config[1]}_{core_config[2]}_{core_config[3]}_{core_config[4]}_{core_config[5]}")
        
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
                        cfg_name = f"eyeriss_{core_config[0]}_{core_config[1]}_{core_config[2]}_{core_config[3]}_{core_config[4]}_{core_config[5]}"
                        check = check_exception(op_info, core_config)
                        if check:
                            if cfg_name not in topo_name_dict:
                                topo_name_dict[cfg_name] = []
                            topo_name_dict[cfg_name].append(topo_name)
                        else:
                            continue
                        
                simulation_mth(cfg_name_list, topo_name_dict)
    n_core *= 2


