# main.py

import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from signals import (
    damped_sinusoid,
    damped_chirp,
    damped_exponential,
    gaussian_pulse,
    damped_square_wave,
    lorentzian_pulse,
    half_sine_pulse,
    damped_sawtooth
)
from monte_carlo import run_custom_simulation_batch

# Time duration to simulate
T_total = 0.1
sample_rate = 2000

# "Base" configurations for your 8 signals
base_configs = {
    'sinusoid': {
        'func': damped_sinusoid,
        'params': {'A': 1.0, 'f': 50.0, 'lambda_': 60.0, 'phase': 0.0},
        'fs': 10.0,
        'T_total': T_total,
        'sample_rate': sample_rate
    },
    'chirp': {
        'func': damped_chirp,
        # e.g., from 40 Hz to 60 Hz within 0.1s, high damping
        'params': {'A': 1.0, 'f0': 30.0, 'f1': 120.0, 'lambda_': 0.01},
        'fs': 10.0,
        'T_total': T_total,
        'sample_rate': sample_rate
    },
    'exponential': {
        'func': damped_exponential,
        'params': {'A': 1.0, 'lambda_': 60.0},
        'fs': 10.0,
        'T_total': T_total,
        'sample_rate': sample_rate
    },
    'gaussian': {
        'func': gaussian_pulse,
        'params': {'A': 1.0, 't0': 0.05, 'sigma': 0.01},  # quick
        'fs': 10.0,
        'T_total': T_total,
        'sample_rate': sample_rate
    },
    'square': {
        'func': damped_square_wave,
        'params': {'A': 1.0, 'duty': 0.5, 'lambda_': 60.0},
        'fs': 10.0,
        'T_total': T_total,
        'sample_rate': sample_rate
    },
    'lorentzian': {
        'func': lorentzian_pulse,
        'params': {'A': 1.0, 't0': 0.05, 'gamma': 0.001}, 
        'fs': 10.0,
        'T_total': T_total,
        'sample_rate': sample_rate
    },
    'half_sine': {
        'func': half_sine_pulse,
        'params': {'A': 1.0, 'width': 0.3, 't0': 0.0, 'lambda_': 60.0},
        'fs': 10.0,
        'T_total': T_total,
        'sample_rate': sample_rate
    },
    'sawtooth': {
        'func': damped_sawtooth,
        'params': {'A': 1.0, 'lambda_': 1.0},
        'fs': 10.0,
        'T_total': T_total,
        'sample_rate': sample_rate
    }
}

def run_multiple_N():
    """
    Loop over N in [3, 5, 8, 12], run Monte Carlo for 8 signal types,
    and produce simulation result folders.
    """
    # We'll analyze first 10 or 15 harmonics. Adjust as needed.
    selected_harmonics = list(range(1, 21))
    num_runs = 1000  # Monte Carlo runs
    
    # We'll store a list of all result directories so we can compile them later.
    all_result_dirs = []
    
    for N in [3, 5, 8, 12]:
        # Build a config dictionary that includes the correct 'N' for each signal:
        configs_for_N = {}
        for sig_name, cfg in base_configs.items():
            # Make a shallow copy
            temp_cfg = cfg.copy()
            # Insert N
            temp_cfg['N'] = N
            configs_for_N[sig_name] = temp_cfg
        
        print(f"\n=== Running batch for N={N} ===\n")
        # We run the batch in parallel or single-threaded as needed.
        results = run_custom_simulation_batch(
            signal_configs=configs_for_N,
            num_runs=num_runs,
            selected_harmonics=selected_harmonics,
            output_dir='simulation_results',
            n_processes=2  # or any number of processes you prefer
        )
        all_result_dirs.extend(results)
    
    return all_result_dirs

def generate_example_signals(output_dir="example_signals",plot=True):
    """
    Generate a single example of each signal type and save to files.
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate time array
    t = np.linspace(0, T_total, int(T_total * sample_rate), endpoint=False)
    
    # For each signal type, generate an example and save
    for signal_name, config in base_configs.items():
        print(f"Generating example for {signal_name}...")
        
        # Extract parameters
        signal_func = config['func']
        params = config['params']
        fs = config['fs']
        
        # Generate signal
        signal = signal_func(t, fs=fs, **params)
        
        if plot:
            # Plot signal
            plt.figure(figsize=(10, 6))
            plt.plot(t, signal)
            plt.title(f"{signal_name.capitalize()} Signal")
            plt.xlabel("Time (s)")
            plt.ylabel("Amplitude")
            plt.grid(True)
            
            # Save plot
            plot_file = os.path.join(output_dir, f"{signal_name}_example.png")
            plt.savefig(plot_file)
            plt.close()
        
        # Save signal data as CSV
        data = pd.DataFrame({"time": t, "amplitude": signal})
        csv_file = os.path.join(output_dir, f"{signal_name}_example.csv")
        data.to_csv(csv_file, index=False)
    
    print(f"Example signals saved to {output_dir}")

def compile_monte_carlo_results(simulation_dirs, output_csv="compiled_results.csv"):
    """
    Reads each Monte Carlo result directory (those from run_custom_simulation_batch)
    and compiles them into a single Pandas DataFrame, then saves to CSV.
    
    The function looks for 'all_z_metrics_by_harmonic.csv' inside each directory.
    It also infers the signal name and N from the folder naming convention:
       e.g. "sinusoid_N5_YYYYMMDD_HHMMSS"
    Adjust the regex or parsing logic based on your actual folder names.
    """
    import re
    
    rows = []
    pattern = re.compile(r'^(?P<signal>.+)_N(?P<N>\d+)_(?P<date>\d{8})_(?P<time>\d{6})$')
    
    for d in simulation_dirs:
        # d might look like: ".../simulation_results/sinusoid_N5_20250226_134500"
        folder_name = os.path.basename(d)  # e.g. 'sinusoid_N5_20250226_134500'
        
        m = pattern.match(folder_name)
        if not m:
            print(f"Warning: folder name '{folder_name}' did not match expected pattern. Skipping.")
            continue
        
        signal_type = m.group('signal')
        N_value = int(m.group('N'))
        
        # read the file "all_z_metrics_by_harmonic.csv" if it exists
        csv_path = os.path.join(d, 'all_z_metrics_by_harmonic.csv')
        if not os.path.isfile(csv_path):
            print(f"No 'all_z_metrics_by_harmonic.csv' found in {d}. Skipping.")
            continue
        
        df = pd.read_csv(csv_path)  # Each row corresponds to a run, columns are harmonic indices if you used that format
        # If your CSV has columns named differently, adjust accordingly.
        
        # Flatten or melt the DataFrame if needed. For simplicity, let's keep as is.
        # Add columns for context:
        df['signal_type'] = signal_type
        df['N'] = N_value
        
        rows.append(df)
    
    if not rows:
        print("No data found to compile. Exiting.")
        return
    
    # Concatenate all data
    big_df = pd.concat(rows, ignore_index=True)
    
    # Save to disk
    big_df.to_csv(output_csv, index=False)
    print(f"Compiled results saved to {output_csv}")

def main():
    # 1) Run the simulations for N=3,5,8,12
    simulation_dirs = run_multiple_N()
    
    # 2) Compile all Monte Carlo results into a single CSV
    compile_monte_carlo_results(
        simulation_dirs,
        output_csv="compiled_ecdf_data.csv"
    )

if __name__ == "__main__":
    main()
    # Generate example signals as a second step
    generate_example_signals()