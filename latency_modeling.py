import os
import pandas as pd
import math


def latency_breakdown(OP_latency_util_):
    latency_util_per_operation =    {
                                        "FC" : [0, 0],
                                        "Attn" : [0, 0],
                                        "comm" : [0, 0],
                                        "etc" : [0, 0]
                                    }
    Total_FC_latency, Total_Attn_latency = 0, 0
    for Op_name, result in OP_latency_util_.items():
        if Op_name in ["QKV", "PROJ", "FFN1", "FFN2"]:
            Total_FC_latency += result[0]
        elif Op_name in ["QKT", "SV", "Softmax"]:
            Total_Attn_latency += result[0]
        elif Op_name in ["Residual", "Layernorm", "ACT"]:
            latency_util_per_operation["etc"][0] += result[0]
        elif Op_name in ["AllReduce", "Send_Recv"]:
            latency_util_per_operation["comm"][0] += result[0]
        else:
            assert False, "지원하지 않는 operation 입니다."
    
    latency_util_per_operation["FC"][0] = Total_FC_latency
    latency_util_per_operation["Attn"][0] = Total_Attn_latency
    
    for Op_name, result in OP_latency_util_.items():
        if Op_name in ["QKV", "PROJ", "FFN1", "FFN2"]:
            latency_util_per_operation["FC"][1] += (result[0] / Total_FC_latency) * result[1]
        elif Op_name in ["QKT", "SV", "Softmax"]:
            latency_util_per_operation["Attn"][1] += (result[0] / Total_Attn_latency) * result[1]
            
    return latency_util_per_operation

def get_total_latency(layer_result_, n_batch_, micro_batch_, card_parallelism_):    
    TP = card_parallelism_[0]
    PP = card_parallelism_[1]
    DP = card_parallelism_[2]
    
    PP_iteration = math.ceil(math.ceil(n_batch_ / DP) / micro_batch_) + PP - 1
    
    OP_latency_util =   {
                        "QKV"       : [0, 0],
                        "QKT"       : [0, 0],
                        "SV"        : [0, 0],
                        "PROJ"      : [0, 0],
                        "FFN1"      : [0, 0],
                        "FFN2"      : [0, 0], 
                        "Softmax"   : [0, 0],
                        "Layernorm" : [0, 0],
                        "Residual"  : [0, 0],
                        "ACT"       : [0, 0],
                        "AllReduce" : [0, 0],
                        "Send_Recv" : [0, 0]
                        }
    
    for Op_name, result in layer_result_.items():
        # breakpoint()
        OP_latency_util[Op_name][0] = result[0] * PP_iteration
        OP_latency_util[Op_name][1] = result[1]
        
    latency_util_per_operation = latency_breakdown(OP_latency_util)
    
    
    total_latency = 0
    for Op_name, result in OP_latency_util.items():
        total_latency += result[0]
    
    total_latency_ = 0
    for Op_name, result in OP_latency_util.items():
        total_latency_ += result[0]
    
    if total_latency != total_latency_:
        breakpoint()
    return OP_latency_util, latency_util_per_operation, total_latency
            
def D2D_latency(H_, d_size_):
    from project import interface_bw
    byte = 2
    total_d_size_ = byte * (H_ * d_size_)
    Send_Recv_latency = total_d_size_ / interface_bw
    
    return Send_Recv_latency
    

def AR_latency(H_, d_size_, TP_, n_core_):
    from project import vector_FLOPS, interface_bw
    byte = 2
    
    total_d_size_ = byte * ((TP_ - 1) * (H_ * d_size_))
    All_Comm_latency = total_d_size_ / (TP_ * interface_bw)
    
    EWADD_FLOPs = (TP_ - 1) * (H_ * d_size_)
    EWADD_latency = EWADD_FLOPs / (vector_FLOPS * n_core_)
        
    return 2 * All_Comm_latency + EWADD_latency

def get_FLOPs(Op_name_, dim_info_):
    Op_type = dim_info_[0]
    H       = dim_info_[1]
    M       = dim_info_[2]
    K       = dim_info_[3]
    N       = dim_info_[4]
    
    if "Softmax" in Op_name_:
        return 5 * H * M * N
    
    elif "ACT" in Op_name_:
        return 8 * H * M * N        # GeLU
        # return 1 * H * M * N        # ReLU
    
    elif "Residual" in Op_name_:
        return H * M * N
    
    elif "Layernorm" in Op_name_:
        return 5 * H * M * N
    
    else:
        assert False, "지원하지 않는 Op_type 입니다."

def access_raw_data(card_parallelism_, core_parallelism_, Op_name_, core_config_):
    from simulation import cfg_name_maker
    from project import base_raw_data_path, total_raw_data_fname
    
    Op_name_layernum = {
                        "QKV"   : 0,
                        "QKT"   : 1,
                        "SV"    : 2,
                        "PROJ"  : 3,
                        "FFN1"  : 4,
                        "FFN2"  : 5
                        }
    
    TP = card_parallelism_[0]
    PP = card_parallelism_[1]
    DP = card_parallelism_[2]
    
    Ht = core_parallelism_[0]
    Mt = core_parallelism_[1]
    Kt = core_parallelism_[2]
    Nt = core_parallelism_[3]
    
    existing_df = pd.read_csv(f"{base_raw_data_path}/{total_raw_data_fname}")
        
    cfg_name = cfg_name_maker(core_config_)
    result_name = f"{cfg_name}_TP-{TP}_M-{Mt}_K-{Kt}_N-{Nt}_{Op_name_layernum[Op_name_]}"
    row_data = existing_df.loc[existing_df['LayerID'] == result_name]
    op_cycle, op_compute_util = (row_data[" Total Cycles"].values)[0], (row_data[" Compute Util %"].values)[0]
    return op_cycle, op_compute_util

def get_latency_util(n_batch_, micro_batch_, op_info_, n_core_, card_parallelism_, core_parallelism_, core_config_, n_layer_):
    from project import vector_FLOPS, frequency
        
    TP = card_parallelism_[0]
    PP = card_parallelism_[1]
    DP = card_parallelism_[2]
    
    Ht = core_parallelism_[0]
    Mt = core_parallelism_[1]
    Kt = core_parallelism_[2]
    Nt = core_parallelism_[3]
    
    n_layer_of_each_stage = int(n_layer_ / PP)
    
    layer_result = {}
    for Op_name, dim_info in op_info_.items():
        dim_info = op_info_[Op_name]
        Op_type = dim_info[0]
        H       = dim_info[1]
        M       = dim_info[2]
        K       = dim_info[3]
        N       = dim_info[4]
    
        
        
        if Op_type in ["FC", "Attn"]:
            op_cycle, op_compute_util = access_raw_data(card_parallelism_, core_parallelism_, Op_name, core_config_)
            layer_result[Op_name] = ((op_cycle / frequency) * H * n_layer_of_each_stage, op_compute_util)
            
        elif "Softmax" in Op_name:
            layer_result[Op_name] = ((get_FLOPs(Op_name, dim_info) * n_layer_of_each_stage) / (vector_FLOPS * n_core_), 0)
        
        elif "Layernorm" in Op_name:
            layer_result[Op_name] = ((get_FLOPs(Op_name, dim_info) * n_layer_of_each_stage) / (vector_FLOPS * n_core_), 0)
        
        elif "Residual" in Op_name:
            layer_result[Op_name] = ((get_FLOPs(Op_name, dim_info) * n_layer_of_each_stage) / (vector_FLOPS * n_core_), 0)
            
        elif "AllReduce" in Op_name:
            layer_result[Op_name] = ((AR_latency(H, M, TP, n_core_) * n_layer_of_each_stage) / (vector_FLOPS * n_core_), 0)
            
        elif "Send_Recv" in Op_name:
            layer_result[Op_name] = (D2D_latency(H, M) / (vector_FLOPS * n_core_), 0)
            
        elif "ACT" in Op_name:
            layer_result[Op_name] = ((get_FLOPs(Op_name, dim_info) * n_layer_of_each_stage) / (vector_FLOPS * n_core_), 0)
        
        else:
            assert False, "지원하지 않는 Operation 입니다."
        
    # breakpoint()
    return get_total_latency(layer_result, n_batch_, micro_batch_, card_parallelism_)