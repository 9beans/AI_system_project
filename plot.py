import matplotlib.pyplot as plt
from project import target_model

def scale_systolic_dim_latency_graph(scale_systolic_dim_latency_results_, subplot_col_):
    main_fontsize = 15
    small_fontsize = 12
    # Number of subplots (rows and columns) based on the number of scales
    num_scales = len(scale_systolic_dim_latency_results_)
    num_cols = subplot_col_
    num_rows = (num_scales + num_cols - 1) // num_cols  # Calculate required rows for the number of subplots

    # Create subplots
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(13, 6))
    axes = axes.flatten()

    # Plot for each scale
    for idx, (scale, systolic_dim_results) in enumerate(scale_systolic_dim_latency_results_.items()):
        x_values = list(systolic_dim_results.keys())
        x_labels = [f"({x[0]}, {x[1]})" for x in x_values]  # Format keys as labels
        y_ws = [systolic_dim_results[dim]['ws'][1] for dim in x_values]  # Get 'ws' values
        y_os = [systolic_dim_results[dim]['os'][1] for dim in x_values]  # Get 'os' values
        
        ax = axes[idx]
        ax.plot(x_labels, y_ws, marker='o', label='ws', color='blue')
        ax.plot(x_labels, y_os, marker='x', label='os', color='red')
        
        card = int(scale / 8)
        if card == 0:
            card = 1
        core = scale % 8
        if core == 0:
            core = 8
        ax.set_title(f"Scale (card, core): ({card}, {core})", fontsize = small_fontsize)
        ax.set_xlabel("Systolic Dimension", fontsize = small_fontsize)
        if (idx == 0) | (idx == 4):
            ax.set_ylabel("Latency (sec)", fontsize = small_fontsize)
        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.6)
        ax.tick_params(axis='x', rotation=45, labelsize = small_fontsize)
        ax.tick_params(axis='y', labelsize = small_fontsize)

    # Remove any unused subplots
    for idx in range(len(scale_systolic_dim_latency_results_), len(axes)):
        fig.delaxes(axes[idx])

    # Adjust layout
    plt.tight_layout()
    plt.savefig(f"{target_model[3]}_Explore_Core_Config_with_System_Scaling.png", format="png", dpi=300, fontsize = small_fontsize)

def scale_systolic_dim_throughput_graph(scale_systolic_dim_throughput_results_):
    import matplotlib.cm as cm
    import numpy as np
    
    main_fontsize = 15
    small_fontsize = 12
    

    # Extract unique systolic_dim keys from all scales
    all_systolic_dims = set()
    for systolic_dim_results in scale_systolic_dim_throughput_results_.values():
        all_systolic_dims.update(systolic_dim_results.keys())

    # Create a color map based on the number of unique systolic dimensions
    colors = cm.tab10(np.linspace(0, 1, len(all_systolic_dims)))
    colors = ['C0', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9']
    color_map = {systolic_dim: colors[i] for i, systolic_dim in enumerate(sorted(all_systolic_dims))}

    # Create subplots
    fig, ax = plt.subplots(figsize=(10, 5))

    # x-axis: scale values
    x_values = list(scale_systolic_dim_throughput_results_.keys())

    # Map x-values to indices to ensure equal distance between scales
    x_labels = []
    x_positions = list(range(len(x_values)))

    for scale in x_values:
        card = int(scale / 8)
        if card == 0:
            card = 1
        core = scale % 8
        if core == 0:
            core = 8
        x_labels.append((card, core))

    # Plot for each (sh, sw) combination for ws and os
    for systolic_dim in sorted(all_systolic_dims):
        y_ws = [scale_systolic_dim_throughput_results_[scale][systolic_dim]['ws'][1] if systolic_dim in scale_systolic_dim_throughput_results_[scale] else None for scale in x_values]
        y_os = [scale_systolic_dim_throughput_results_[scale][systolic_dim]['os'][1] if systolic_dim in scale_systolic_dim_throughput_results_[scale] else None for scale in x_values]

        ax.plot(x_positions, y_ws, marker='o', label=f"ws {systolic_dim}", linestyle='-', color=color_map[systolic_dim])
        ax.plot(x_positions, y_os, marker='x', label=f"os {systolic_dim}", linestyle='--', color=color_map[systolic_dim])
    # breakpoint()
    # Set x-axis tick labels to be the scale values and ensure equal distance
    ax.set_xticks(x_positions)
    ax.set_xticklabels([str(scale) for scale in x_labels], fontsize = main_fontsize)
    plt.yticks(fontsize = main_fontsize)

    ax.set_title("Throughput Scalability for Different Systolic Dimensions (log-scale)", fontsize = main_fontsize)
    ax.set_xlabel("Scale (card, core)", fontsize = main_fontsize)
    ax.set_ylabel("Throughput (token/sec)", fontsize = main_fontsize)
    ax.legend(bbox_to_anchor=(1, 1), loc='upper left', fontsize = small_fontsize)
    ax.grid(True, linestyle='--', alpha=0.6)
    
    plt.yscale('log', base=2)
    
    # Adjust layout
    plt.tight_layout()
    plt.savefig(f"{target_model[3]}_Scalability_for_Throughput.png", format="png", dpi=300)

    
def scale_core_parallelism_graph(scale_core_parallelism_results_, subplot_col_):
    main_fontsize = 15
    small_fontsize = 12
    
    # Number of subplots (rows and columns) based on the number of scales
    num_scales = len(scale_core_parallelism_results_)
    num_cols = subplot_col_
    num_rows = (num_scales + num_cols - 1) // num_cols  # Calculate required rows for the number of subplots

    # Create subplots
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(8, 10))
    axes = axes.flatten()

    # Plot for each scale
    for idx, (scale, systolic_dim_results) in enumerate(scale_core_parallelism_results_.items()):
        x_values = list(systolic_dim_results.keys())
        x_labels = [f"({x[0]}, {x[1]}, {x[2]})" for x in x_values]  # Format keys as labels
        y_ws = [systolic_dim_results[dim][1] for dim in x_values]  # Get 'ws' values
        
        ax = axes[idx]
        ax.plot(x_labels, y_ws, marker='o', label='128x128 core', color='blue')
        
        card = int(scale / 8)
        if card == 0:
            card = 1
        core = scale % 8
        if core == 0:
            core = 8
            
        ax.set_title(f"Scale (card, core): ({card}, {core})", fontsize = main_fontsize)
        if idx == num_rows - 1:
            ax.set_xlabel("GEMM Partitioning at Core-level (H, M, N)", fontsize = main_fontsize)
        ax.set_ylabel("Latency (sec)", fontsize = main_fontsize)
        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.6)
        ax.tick_params(axis='x', labelsize=main_fontsize, rotation=45)  # X-axis ticks
        ax.tick_params(axis='y', labelsize=main_fontsize)  # Y-axis ticks
        

    # Remove any unused subplots
    for idx in range(len(scale_core_parallelism_results_), len(axes)):
        fig.delaxes(axes[idx])
    
    # Adjust layout
    plt.tight_layout()
    plt.savefig(f"{target_model[3]}_Explore_Core_Parallelism_with_System_Scaling.png", format="png", dpi=300)
    
def scale_card_parallelism_graph(card_parallelism_results_, H_M_K_N_):
    main_fontsize = 15
    small_fontsize = 12

    # Create subplots
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot for each systolic dimension result
    x_values = list(card_parallelism_results_.keys())
    x_labels = [f"({x[0]}, {x[1]}, {x[2]})" for x in x_values]  # Format keys as labels
    y_ws = [card_parallelism_results_[dim][1] for dim in x_values]  # Get 'ws' values
    
    ax.plot(x_labels, y_ws, marker='o', label='128x128 core', color='blue')
    
    # Calculate card and core values (assuming scale is not provided directly)
    # Assuming "scale" can be derived from the number of entries or a specific logic if available
    
    scale = 64
    card = int(scale / 8)
    if card == 0:
        card = 1
    core = scale % 8
    if core == 0:
        core = 8
    
    ax.set_title(f"Scale (card, core): ({card}, {core}) , Core parallelism: {H_M_K_N_}", fontsize = main_fontsize)
    ax.set_xlabel("Model Parallelism & Data Parallelism at Card-level (TP, PP, DP)", fontsize = main_fontsize)
    ax.set_ylabel("Latency (sec)", fontsize = main_fontsize)
    ax.legend(fontsize = small_fontsize)
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.tick_params(axis='x', labelsize=main_fontsize, rotation=45)  # X-axis ticks
    ax.tick_params(axis='y', labelsize=main_fontsize)  # Y-axis ticks

    # Adjust layout and save the plot
    plt.tight_layout()
    plt.savefig(f"{target_model[3]}_Explore_Card_Parallelism_with_{H_M_K_N_}.png", format="png", dpi=300)