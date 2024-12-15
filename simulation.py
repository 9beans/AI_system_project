import pandas as pd
import os
import math
import multiprocessing
import subprocess

def make_topology(topo_name_, op_info_):
    from project import base_topology_path
    target_op_list = ['QKV', 'QKT', 'SV', 'PROJ', 'FFN1', 'FFN2']
    
    data = {
        'Layer name': target_op_list,
        'IFMAP Height': [],                         # M
        'IFMAP Width': [],                          # K
        'Filter Height': [1, 1, 1, 1, 1, 1],           # 1
        'Filter Width': [],                         # K
        'Channels': [1, 1, 1, 1, 1, 1],                # 1
        'Num Filter': [],                          # N
        'Strides': [1, 1, 1, 1, 1, 1],                 # 1
        'batch size': [1, 1, 1, 1, 1, 1]               # 1
    }

    for op in target_op_list:
        dim_info = op_info_[op]
        # Op_type = dim_info[0]
        # H = dim_info[1]
        M = dim_info[2]
        K = dim_info[3]
        N = dim_info[4]
        data['IFMAP Height'].append(M)
        data['IFMAP Width'].append(K)
        data['Filter Width'].append(K)
        data['Num Filter'].append(N)
    
    df = pd.DataFrame(data)
    df.to_csv(f"{base_topology_path}/{topo_name_}.csv", index=False)

def simulation_sth(cfg_name_, topo_name_, thread_num_):
    from project import base_scalesim_code_path, base_config_path, base_topology_path, base_raw_data_path
    os.makedirs(f"{base_raw_data_path}/{thread_num_}", exist_ok=True)
    
    cmd = [
        "python3", 
        base_scalesim_code_path, 
        "-c", f"{base_config_path}/{cfg_name_}.cfg", 
        "-t", f"{base_topology_path}/{topo_name_}.csv", 
        "-p", f"{base_raw_data_path}/{thread_num_}", 
        "-i", "conv"
    ]
    
    with open(os.devnull, 'w') as devnull:
        subprocess.run(cmd, stdout=devnull, stderr=devnull)
    
    # Load the CSV file
    file_path_compute_report = f'{base_raw_data_path}/{thread_num_}/COMPUTE_REPORT.csv'
    compute_report_df = pd.read_csv(file_path_compute_report)
    compute_report_df['LayerID'] = compute_report_df['LayerID'].apply(lambda x: f'{cfg_name_}_{topo_name_}_{x}')
    # # Extract the relevant columns
    # extracted_data = compute_report_df[['Total Cycles', 'Overall Util %', 'Mapping Efficiency %', 'Compute Util %']]

    return compute_report_df

def simulation_mth(cfg_name_list_, topo_name_dict_):
    from project import base_raw_data_path, simulation_threads, total_raw_data_fname
    search_space_topo = 0
    search_space_cfg = 0
    
    for cfg_name, topo_name_list in topo_name_dict_.items():
        search_space_cfg += 1
        search_space_topo += len(topo_name_list)
        
    print(f"# cfg : {search_space_cfg}")
    print(f"# total : {search_space_topo}")
    
    existing_df = None
    if os.path.isfile(f"{base_raw_data_path}/{total_raw_data_fname}"):
        existing_df = pd.read_csv(f"{base_raw_data_path}/{total_raw_data_fname}")
        
    cfg_topo_list = []
    case_num = 0
    for cfg_name in cfg_name_list_:
        topo_name_list = topo_name_dict_[cfg_name]
        for topo_name in topo_name_list:
            if (existing_df != None):
                if f"{cfg_name}_{topo_name}_0" in existing_df['LayerID']:
                    continue
            cfg_topo_list.append((cfg_name, topo_name, case_num % simulation_threads))
            case_num += 1
    
    print(f"# total simulation : {case_num}")
    assert len(cfg_topo_list) == search_space_topo, "전체 search space 크기가 이론상 크기와 맞지 않습니다."
    # breakpoint()
    
    max_iter = math.ceil(len(cfg_topo_list)/simulation_threads)
    for iter in range(max_iter):
        total_thread_num = 0
        if iter == (max_iter - 1):
            total_thread_num = len(cfg_topo_list) % simulation_threads
            assert cfg_topo_list[-1][2] == (total_thread_num - 1)
        else:
            total_thread_num = simulation_threads
            
        pool = multiprocessing.Pool(processes=total_thread_num)
        assert (cfg_topo_list[total_thread_num * iter][2] == 0) & (cfg_topo_list[total_thread_num * iter + (total_thread_num - 1)][2] == (total_thread_num - 1)), "Task가 다른 Thread에 할당되었습니다."
        output_list = pool.starmap(simulation_sth, cfg_topo_list[total_thread_num * iter : total_thread_num * iter + total_thread_num])
        pool.close()
        pool.join()
        
        result_df_list = []
        for thread in range(len(total_thread_num)):
            result_df_list.append(output_list[thread])
        combined_df = pd.concat(result_df_list, ignore_index=True)
        
        # Check if the file exists
        if os.path.isfile(f"{base_raw_data_path}/{total_raw_data_fname}"):
            existing_df = pd.read_csv(f"{base_raw_data_path}/{total_raw_data_fname}")
            combined_df = pd.concat([existing_df, combined_df], ignore_index=True)
        
        combined_df.to_csv({base_raw_data_path}/{total_raw_data_fname}, index=False)
        
