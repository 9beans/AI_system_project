import matplotlib.pyplot as plt
from project import target_model

def scale_systolic_dim_latency_graph(scale_systolic_dim_latency_results_, subplot_col_):
    # Number of subplots (rows and columns) based on the number of scales
    num_scales = len(scale_systolic_dim_latency_results_)
    num_cols = subplot_col_
    num_rows = (num_scales + num_cols - 1) // num_cols  # Calculate required rows for the number of subplots

    # Create subplots
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(15, 10))
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
        ax.set_title(f"Scale (card, core): ({card}, {core})")
        ax.set_xlabel("Systolic Dimension")
        ax.set_ylabel("Latency (sec)")
        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.6)
        ax.tick_params(axis='x', rotation=45)

    # Remove any unused subplots
    for idx in range(len(scale_systolic_dim_latency_results_), len(axes)):
        fig.delaxes(axes[idx])

    # Adjust layout
    plt.tight_layout()
    plt.savefig(f"{target_model[3]}_Explore_Core_Config_with_System_Scaling.png", format="png", dpi=300)

def scale_systolic_dim_throughput_graph(scale_systolic_dim_throughput_results_):

    # Extract unique systolic_dim keys from all scales
    all_systolic_dims = set()
    for systolic_dim_results in scale_systolic_dim_throughput_results_.values():
        all_systolic_dims.update(systolic_dim_results.keys())

    # Create subplots
    fig, ax = plt.subplots(figsize=(12, 8))

    # x-axis: scale values
    x_values = list(scale_systolic_dim_throughput_results_.keys())

    # Map x-values to indices to ensure equal distance between scales
    x_positions = list(range(len(x_values)))

    # Plot for each (sh, sw) combination for ws and os
    for systolic_dim in all_systolic_dims:
        y_ws = [scale_systolic_dim_throughput_results_[scale][systolic_dim]['ws'][1] if systolic_dim in scale_systolic_dim_throughput_results_[scale] else None for scale in x_values]
        y_os = [scale_systolic_dim_throughput_results_[scale][systolic_dim]['os'][1] if systolic_dim in scale_systolic_dim_throughput_results_[scale] else None for scale in x_values]
        
        ax.plot(x_positions, y_ws, marker='o', label=f"ws {systolic_dim}", linestyle='-')
        ax.plot(x_positions, y_os, marker='x', label=f"os {systolic_dim}", linestyle='--')

    # Set x-axis tick labels to be the scale values and ensure equal distance
    ax.set_xticks(x_positions)
    ax.set_xticklabels([str(scale) for scale in x_values])

    ax.set_title("Latency vs Scale for Different Systolic Dimensions")
    ax.set_xlabel("Scale")
    ax.set_ylabel("Latency (ns)")
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True, linestyle='--', alpha=0.6)

    # Adjust layout
    plt.tight_layout()
    plt.savefig(f"{target_model[3]}_Scalability_for_Throughput.png", format="png", dpi=300)

    
def scale_core_parallelism_graph(scale_core_parallelism_results_, subplot_col_):

    # Number of subplots (rows and columns) based on the number of scales
    num_scales = len(scale_core_parallelism_results_)
    num_cols = subplot_col_
    num_rows = (num_scales + num_cols - 1) // num_cols  # Calculate required rows for the number of subplots

    # Create subplots
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(15, 10))
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
            
        ax.set_title(f"Scale (card, core): ({card}, {core})")
        ax.set_xlabel("GEMM Partitioning at Core-level (H, M, N)")
        ax.set_ylabel("Throughput (Token/sec)")
        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.6)
        ax.tick_params(axis='x', rotation=45)

    # Remove any unused subplots
    for idx in range(len(scale_core_parallelism_results_), len(axes)):
        fig.delaxes(axes[idx])

    # Adjust layout
    plt.tight_layout()
    plt.savefig(f"{target_model[3]}_Explore_Core_Parallelism_with_System_Scaling.png", format="png", dpi=300)
    
def scale_card_parallelism_graph(card_parallelism_results_, H_M_K_N_):

    # Create subplots
    fig, ax = plt.subplots(figsize=(15, 10))
    
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
    
    ax.set_title(f"Scale (card, core): ({card}, {core}) , Core parallelism: {H_M_K_N_}")
    ax.set_xlabel("Model Parallelism & Data Parallelism at Card-level (TP, PP, DP)")
    ax.set_ylabel("Latency (sec)")
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.tick_params(axis='x', rotation=45)

    # Adjust layout and save the plot
    plt.tight_layout()
    plt.savefig(f"{target_model[3]}_Explore_Card_Parallelism_with_{H_M_K_N_}.png", format="png", dpi=300)