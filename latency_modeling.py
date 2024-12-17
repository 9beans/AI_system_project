import os
import pandas as pd

def AR_latency(H_, d_size_, BW_, n_dev_):
    from project import vector_FLOPS
    byte = 2
    
    total_d_size_ = byte * ((n_dev_ - 1) * (H_ * d_size_))
    All_Comm_latency = total_d_size_ / (n_dev_ * BW_)
    
    EWADD_FLOPs = (n_dev_ - 1) * (H_ * d_size_)
    EWADD_latency = EWADD_FLOPs / vector_FLOPS
        
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
        return 1 * H * M * N        # ReLU
    
    elif "Residual" in Op_name_:
        return H * M * N
    
    elif "Layernorm" in Op_name_:
        return 5 * H * M * N
    
    else:
        assert False, "지원하지 않는 Op_type 입니다."



def get_latency_util(op_info_, BW_, n_card_, n_core, core_config_):
    from project import vector_FLOPS
    from project import base_raw_data_path, total_raw_data_fname
    
    # for Op_name, dim_info in op_info_.items():
    #     dim_info = op_info_[Op_name]
    #     Op_type = dim_info[0]
    #     H       = dim_info[1]
    #     M       = dim_info[2]
    #     K       = dim_info[3]
    #     N       = dim_info[4]
    
    #     existing_df = None
    #     if os.path.isfile(f"{base_raw_data_path}/{total_raw_data_fname}"):
    #         existing_df = pd.read_csv(f"{base_raw_data_path}/{total_raw_data_fname}")
            
        
    #     if Op_type == "FC":
    #         existing_df
    #     elif Op_type == "Attn":
            
    #     elif Op_name in "Softmax":
    #         return (get_FLOPs(Op_name, dim_info) / (vector_FLOPS * n_core), 0)
            
    #     elif Op_name in "AllReduce":
    #         return (get_FLOPs(Op_name, dim_info) / (vector_FLOPS * n_core), 0)
        
    #     elif Op_name in "Layernorm":
    #         return (get_FLOPs(Op_name, dim_info) / (vector_FLOPS * n_core), 0)
        
    #     elif Op_name in "Residual":
    #         return (get_FLOPs(Op_name, dim_info) / (vector_FLOPS * n_core), 0)
        
    #     else:
    #         assert False, "지원하지 않는 Operation 입니다."