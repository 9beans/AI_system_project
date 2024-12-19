import argparse
import os

def str_to_bool(value):
    """
    Convert a string to a boolean value.
    """
    if value.lower() in {'True', 'true', '1', 'yes'}:
        return True
    elif value.lower() in {'False', 'false', '0', 'no'}:
        return False
    else:
        raise argparse.ArgumentTypeError(f"Invalid boolean value: '{value}'")

def str_to_list(value):
    """
    Convert a comma-separated string into a list of integers.
    Accepts a string like '10,20,30' and converts it to [10, 20, 30].
    Raises an error if the input string cannot be parsed into integers.
    
    Parameters:
        value (str): The input string to be converted to a list.

    Returns:
        list: A list of integers extracted from the input string.
    """
    try:
        return [int(item.strip()) for item in value.split(',')]
    except ValueError:
        raise argparse.ArgumentTypeError(f"Invalid list format: '{value}'. Expected format: 'val1,val2,val3 ...'")

def generate_sets(pe_size, pe_threshold):
    """
    Generate all pairs of factors for the given total size,
    filtering out pairs where either value is less than the threshold.
    
    Parameters:
        pe_size (int): The total size for which factor pairs are generated.
        pe_threshold (int): The minimum value for both factors to be included in the result.

    Returns:
        list: A list of tuples (ArrayHeight, ArrayWidth) where both values meet the threshold.
    """
    sets = []
    for i in range(1, pe_size + 1):
        if pe_size % i == 0:
            # Calculate the pair (i, pe_size // i)
            pair = (i, pe_size // i)
            # Include the pair only if both values are greater than or equal to the threshold
            if pair[0] >= pe_threshold and pair[1] >= pe_threshold:
                sets.append(pair)
    return sets


def modify_config(file_path, array_height, array_width, dataflow, ifmap_size, filter_size, ofmap_size, bandwidth, mode):
    """
    Modify the configuration file with given parameters and create a new file in the specified path.
    """
    # Extract config_name and its directory
    config_name = os.path.splitext(os.path.basename(file_path))[0]
    config_dir = os.path.dirname(file_path)

    # Create a directory for temporary config files in the same directory as the original config
    temp_dir = os.path.join(config_dir, f"{config_name}_gen")
    os.makedirs(temp_dir, exist_ok=True)

    # Generate the new file path in the temporary directory
    new_file_name = f"{config_name}_{array_height}_{array_width}_{dataflow}_{ifmap_size//1000}_{filter_size//1000}_{ofmap_size//1000}.cfg"
    new_file_path = os.path.join(temp_dir, new_file_name)

    with open(file_path, "r") as f:
        lines = f.readlines()

    # Modify the config file content and save it to the new file path
    with open(new_file_path, "w") as f:
        for line in lines:
            if line.startswith("run_name ="):
                # Update run_name
                f.write(f"run_name = {config_name}_{array_height}_{array_width}_{dataflow}_{ifmap_size//1000}_{filter_size//1000}_{ofmap_size//1000}\n")
            elif line.startswith("ArrayHeight"):
                f.write(f"ArrayHeight:    {array_height}\n")
            elif line.startswith("ArrayWidth"):
                f.write(f"ArrayWidth:     {array_width}\n")
            elif line.startswith("IfmapSramSzkB"):
                f.write(f"IfmapSramSzkB:    {ifmap_size}\n")
            elif line.startswith("FilterSramSzkB"):
                f.write(f"FilterSramSzkB:   {ifmap_size}\n")
            elif line.startswith("OfmapSramSzkB"):
                f.write(f"OfmapSramSzkB:    {ifmap_size}\n")
            elif line.startswith("Dataflow"):
                f.write(f"Dataflow: {dataflow}\n")
            elif line.startswith("Bandwidth"):
                f.write(f"Bandwidth: {bandwidth}\n")
            elif line.startswith("InterfaceBandwidth"):
                f.write(f"InterfaceBandwidth: {mode}\n")
            else:
                f.write(line)

    return new_file_path


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', metavar='Config file', type=str,
                        default="../configs/scale.cfg",
                        help="Path to the baseline config file"
                        )
    parser.add_argument('-s', metavar='PE size', type=int,
                        required=True, 
                        help="Total size for ArrayHeight x ArrayWidth")
    parser.add_argument('-t', metavar='PE min dimension', type=int,
                        required=True, 
                        help="Minimum dimension of PE")
    parser.add_argument('-m', metavar='Mem size', type=str_to_list,
                        required=True,
                        help="[Ifmap size (KB), Filter size (KB), Ofmap size (KB)]"
                        )
    parser.add_argument('-b', metavar='Bandwidth', type=int,
                        required=True,
                        help="DRAM bandwidth (words/cycle)"
                        )
    parser.add_argument('-e', metavar='mode', type=str,
                        default="USER",
                        help="InterfaceBandwidth mode")

    args = parser.parse_args()
    print(args)
    config = args.c
    pe_size = args.s
    pe_threshold = args.t
    mem_size = args.m
    bandwidth = args.b
    mode = args.e
    
    pe_sets = generate_sets(pe_size, pe_threshold)
    print(f"PE sets: {pe_sets} | PE min dimension: {pe_threshold}")
    
    for array_height, array_width in pe_sets:
        for dataflow in ["ws", "os"]:
            modify_config(
                            file_path=config,
                            array_height=array_height,
                            array_width=array_width,
                            dataflow=dataflow,
                            ifmap_size=mem_size[0],
                            filter_size=mem_size[1],
                            ofmap_size=mem_size[2],
                            bandwidth=bandwidth,
                            mode=mode
                          )