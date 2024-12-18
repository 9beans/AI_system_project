import matplotlib.pyplot as plt

def scale_systolic_dim_graph(scale_systolic_dim_results_):
    # Number of subplots (rows and columns) based on the number of scales
    num_scales = len(scale_systolic_dim_results_)
    num_cols = 4
    num_rows = (num_scales + num_cols - 1) // num_cols  # Calculate required rows for the number of subplots

    # Create subplots
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(15, 10))
    axes = axes.flatten()

    # Plot for each scale
    for idx, (scale, systolic_dim_results) in enumerate(scale_systolic_dim_results_.items()):
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
    for idx in range(len(scale_systolic_dim_results_), len(axes)):
        fig.delaxes(axes[idx])

    # Adjust layout
    plt.tight_layout()
    plt.savefig("test.png", format="png", dpi=300)
