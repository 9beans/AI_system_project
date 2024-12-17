import pandas as pd
import os
import math
import multiprocessing
import subprocess

def cfg_name_maker(core_config_):
    from project import NPU_name
    return f"{NPU_name}_{core_config_[0]}_{core_config_[1]}_{core_config_[2]}_{core_config_[3]}_{core_config_[4]}_{core_config_[5]}"

def make_topology(topo_name_, op_info_):
    from project import base_topology_path
    target_op_list = ['QKV', 'QKT', 'SV', 'PROJ', 'FFN1', 'FFN2']
    
    os_data = {
        'Layer name': target_op_list,
        'IFMAP Height': [],                         # M
        'IFMAP Width': [],                          # K
        'Filter Height': [1, 1, 1, 1, 1, 1],        # 1
        'Filter Width': [],                         # K
        'Channels': [1, 1, 1, 1, 1, 1],             # 1
        'Num Filter': [],                           # N
        'Strides': [1, 1, 1, 1, 1, 1],              # 1
        'batch size': [1, 1, 1, 1, 1, 1]            # 1
    }
    
    ws_data = {
        'Layer name': target_op_list,
        'IFMAP Height': [],                         # M
        'IFMAP Width': [],                          # K
        'Filter Height': [1, 1, 1, 1, 1, 1],        # 1
        'Filter Width': [],                         # K
        'Channels': [1, 1, 1, 1, 1, 1],             # 1
        'Num Filter': [],                           # N
        'Strides': [1, 1, 1, 1, 1, 1],              # 1
        'batch size': [1, 1, 1, 1, 1, 1]            # 1
    }

    for op in target_op_list:
        dim_info = op_info_[op]
        # Op_type = dim_info[0]
        # H = dim_info[1]
        M = dim_info[2]
        K = dim_info[3]
        N = dim_info[4]
        
        os_data['IFMAP Height'].append(M)
        os_data['IFMAP Width'].append(K)
        os_data['Filter Width'].append(K)
        os_data['Num Filter'].append(N)
        
        ws_data['IFMAP Height'].append(K)
        ws_data['IFMAP Width'].append(M)
        ws_data['Filter Width'].append(M)
        ws_data['Num Filter'].append(N)
    
    os_df = pd.DataFrame(os_data)
    os_df.to_csv(f"{base_topology_path}/{topo_name_}_os.csv", index=False)
    
    ws_df = pd.DataFrame(ws_data)
    ws_df.to_csv(f"{base_topology_path}/{topo_name_}_ws.csv", index=False)


def simulation_sth(core_config_, topo_name_, thread_num_):
    cfg_name = cfg_name_maker(core_config_)
    from project import base_scalesim_code_path, base_config_path, base_topology_path, base_raw_data_path
    os.makedirs(f"{base_raw_data_path}/{thread_num_}", exist_ok=True)
    
    cmd = f"python3 {base_scalesim_code_path} -c {base_config_path}/{cfg_name}.cfg -t {base_topology_path}/{topo_name_}_{core_config_[2]}.csv -p {base_raw_data_path}/{thread_num_} -i conv"
    # cmd = f"python3 {base_scalesim_code_path} -c {base_config_path}/{cfg_name}.cfg -t {base_topology_path}/{topo_name_}.csv -p {base_raw_data_path}/{thread_num_} -i conv > /dev/null 2>&1"
    # print(cmd)
    # Use os.popen and read to ensure the process waits for completion
    os.popen(cmd).read()  # read() waits for the command to complete
    
    # Load the CSV file
    file_path_compute_report = f'{base_raw_data_path}/{thread_num_}/{cfg_name}/COMPUTE_REPORT.csv'
    compute_report_df = pd.read_csv(file_path_compute_report)
    compute_report_df['LayerID'] = compute_report_df['LayerID'].apply(lambda x: f'{cfg_name}_{topo_name_}_{x}')
    # # Extract the relevant columns
    # extracted_data = compute_report_df[['Total Cycles', 'Overall Util %', 'Mapping Efficiency %', 'Compute Util %']]

    return compute_report_df

def simulation_mth(total_hw_search_space_, topo_name_dict_):
    from project import base_raw_data_path, simulation_threads, total_raw_data_fname
    search_space_topo = 0
    search_space_cfg = 0
    
    for cfg_name, topo_name_list in topo_name_dict_.items():
        search_space_cfg += 1
        search_space_topo += len(topo_name_list)
        
    print(f"# cfg : {search_space_cfg}")
    print(f"# total : {search_space_topo}")
    
    existing_df = None
    exist = False
    if os.path.isfile(f"{base_raw_data_path}/{total_raw_data_fname}"):
        existing_df = pd.read_csv(f"{base_raw_data_path}/{total_raw_data_fname}")
        exist = True
        
    cfg_topo_list = []
    case_num = 0
    for core_config in total_hw_search_space_:
        cfg_name = cfg_name_maker(core_config)
        topo_name_list = topo_name_dict_[cfg_name]
        for topo_name in topo_name_list:
            if exist:
                if f"{cfg_name}_{topo_name}_0" in existing_df['LayerID'].values:
                    continue
            cfg_topo_list.append((core_config, topo_name, case_num % simulation_threads))
            case_num += 1
    
    print(f"# total simulation : {case_num}")
    # assert len(cfg_topo_list) == search_space_topo, "전체 search space 크기가 이론상 크기와 맞지 않습니다."
    # breakpoint()
    
    max_iter = math.ceil(len(cfg_topo_list)/simulation_threads)
    for iter in range(max_iter):
        print(f"simulation...{iter + 1}/{max_iter}")
        total_thread_num = 0
        if iter == (max_iter - 1):
            total_thread_num = len(cfg_topo_list) % simulation_threads
            if total_thread_num == 0:
                total_thread_num = simulation_threads
        else:
            total_thread_num = simulation_threads
        # breakpoint()
        pool = multiprocessing.Pool(processes=total_thread_num)
        assert (cfg_topo_list[simulation_threads * iter][2] == 0) & (cfg_topo_list[simulation_threads * iter + (total_thread_num - 1)][2] == (total_thread_num - 1)), "Task가 다른 Thread에 할당되었습니다."
        output_list = pool.starmap(simulation_sth, cfg_topo_list[simulation_threads * iter : simulation_threads * iter + total_thread_num])
        pool.close()
        pool.join()
        
        result_df_list = []
        for thread in range(total_thread_num):
            result_df_list.append(output_list[thread])
        combined_df = pd.concat(result_df_list, ignore_index=True)
        
        # Check if the file exists
        if exist:
            existing_df = pd.read_csv(f"{base_raw_data_path}/{total_raw_data_fname}")
            combined_df = pd.concat([existing_df, combined_df], ignore_index=True)
        
        combined_df.to_csv(f"{base_raw_data_path}/{total_raw_data_fname}", index=False)
        
        
