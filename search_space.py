import math

def check_exception(op_info_, core_config_):
    target_op_list = ['QKV', 'QKT', 'SV', 'PROJ', 'FFN1', 'FFN2']
    
    for op in target_op_list:
        dim_info = op_info_[op]
        # Op_type = dim_info[0]
        H = dim_info[1]
        M = dim_info[2]
        K = dim_info[3]
        N = dim_info[4]
        
        Systolic_H = core_config_[0]
        Systolic_W = core_config_[1]
        Systolic_Dataflow = core_config_[2]
        
        # if (Systolic_Dataflow == "os") & ((M < Systolic_H) & (N < Systolic_W)):
        #     return False
        # elif (Systolic_Dataflow == "ws") & ((K < Systolic_H) & (N < Systolic_W)):
        #     return False
        # else:
        #     return True
        return True
    
    
# Card_parallelism_ : (TP, PP, DP), Core_parallelism_ : (H, M, K, N)
def make_op_info(Core_parallelism_, workload_dim_dict_):
    n_micro_batch = workload_dim_dict_["n_micro_batch"]
    n_token = workload_dim_dict_["n_token"]
    n_head = workload_dim_dict_["n_head"]
    d_head = workload_dim_dict_["d_head"]
    d_model = workload_dim_dict_["d_model"]
    intersize = workload_dim_dict_["intersize"]
    
    H_tiles = Core_parallelism_[0]
    M_tiles = Core_parallelism_[1]
    K_tiles = Core_parallelism_[2]
    N_tiles = Core_parallelism_[3]
        
    transformer_operation = {               # (Op type  , H                                             , M                                 , K                                     , N)
                            "QKV"           : ("FC"     , math.ceil(n_micro_batch / H_tiles)            , math.ceil(n_token / M_tiles)      , math.ceil(d_model / K_tiles)          , math.ceil(n_head * d_head * 3 / N_tiles)),
                            "QKT"           : ("Attn"   , math.ceil(n_head * n_micro_batch / H_tiles)   , math.ceil(n_token / M_tiles)      , math.ceil(d_head / K_tiles)           , math.ceil(n_token / N_tiles)),
                            "Softmax"       : ("else"   , n_head * n_micro_batch                        , n_token                           , 1                                     , n_token),
                            "SV"            : ("Attn"   , math.ceil(n_head * n_micro_batch / H_tiles)   , math.ceil(n_token / M_tiles)      , math.ceil(n_token / K_tiles)          , math.ceil(d_head / N_tiles)),
                            "PROJ"          : ("FC"     , math.ceil(n_micro_batch / H_tiles)            , math.ceil(n_token / M_tiles)      , math.ceil(n_head * d_head / K_tiles)  , math.ceil(d_model / N_tiles)),
                            "AllReduce"     : ("comm"   , n_token * n_micro_batch * 2                   , d_model                           , 1                                     , 1),       
                            "Residual"      : ("else"   , n_token * n_micro_batch * 2                   , d_model                           , 1                                     , 1),       
                            "Layernorm"     : ("else"   , n_token * n_micro_batch * 2                   , d_model                           , 1                                     , 1),       
                            "FFN1"          : ("FC"     , math.ceil(n_micro_batch / H_tiles)            , math.ceil(n_token / M_tiles)      , math.ceil(d_model / K_tiles)          , math.ceil(intersize / N_tiles)),       
                            "ACT"           : ("else"   , n_token * n_micro_batch                       , intersize                         , 1                                     , 1),
                            "FFN2"          : ("FC"     , math.ceil(n_micro_batch / H_tiles)            , math.ceil(n_token / M_tiles)      , math.ceil(intersize / K_tiles)        , math.ceil(d_model / N_tiles)),
                            "Send_Recv"     : ("else"   , n_token * n_micro_batch                       , d_model                           , 1                                     , 1),
                            }
    
    target_op_list = ['QKV', 'QKT', 'SV', 'PROJ', 'FFN1', 'FFN2']
    for op in target_op_list:
        dim_info = transformer_operation[op]
        # Op_type = dim_info[0]
        # H = dim_info[1]
        M = dim_info[2]
        K = dim_info[3]
        N = dim_info[4]
        if (M == 1) | (K == 1) | (N == 1):
            return None
        
    return transformer_operation

def partition_product(num, N, partial_partition=None, output_list=None):
    if partial_partition is None:
        partial_partition = []
    if output_list is None:
        output_list = []

    # Base case: If only one factor is left, check if it divides `num` exactly
    if N == 1:
        if num >= 1:
            partial_partition.append(num)
            output_list.append(partial_partition.copy())
            partial_partition.pop()
        return

    # Recursively generate all partitions
    for i in range(1, num + 1):
        if num % i == 0:  # i must divide num
            partial_partition.append(i)
            partition_product(num // i, N - 1, partial_partition, output_list)
            partial_partition.pop()

    
# Card : (TP, PP, DP), Core : (H, M, K, N)
def Topo_search_space(card_num_, core_num_, card_parallelism_list_, core_parallelism_list_):    
    partition_product(card_num_, 3, output_list=card_parallelism_list_)
    partition_product(core_num_, 4, output_list=core_parallelism_list_)

def HW_search_space(PE_num_, total_hw_search_space_):
    systolic_dim_list = [
        # (1, 16384),
        # (2, 8192),
        # (4, 4096),
        # (8, 2048),
        # (16, 1024),
        (32, 512),
        (64, 256),
        (128, 128),
        (256, 64),
        (512, 32),
        # (1024, 16),
        # (2048, 8),
        # (4096, 4),
        # (8192, 2),
        # (16384, 1),
    ]
    IRAM = 6    # MiB
    WRAM = 6    # MiB
    ORAM = 4    # MiB
    
    for systolic_dim in systolic_dim_list:
        if (systolic_dim[0] % 2 == 0) & (systolic_dim[1] % 2 == 0):     # 홀수 PE dimension은 건너뛰기
            assert PE_num_ == (systolic_dim[0] * systolic_dim[1]), "설정한 전체 PE 개수와 systolic array에 배치된 PE 개수가 다릅니다."
            total_hw_search_space_.append((systolic_dim[0], systolic_dim[1], "os", IRAM, WRAM, ORAM))
            total_hw_search_space_.append((systolic_dim[0], systolic_dim[1], "ws", IRAM, WRAM, ORAM))