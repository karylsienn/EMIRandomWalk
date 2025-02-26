# monte_carlo.py
import os
import time
import glob
import pandas as pd
import numpy as np
from datetime import datetime
import multiprocessing
from functools import partial

# You might need to import your signals or simulation modules
from simulation import run_simulation
from signals import DEFAULT_SIGNAL_CONFIGS

def monte_carlo_simulation(signal_func, signal_params, 
                           num_runs=10, N=5, fs=10.0, 
                           T_total=0.5, sample_rate=1000,
                           output_dir='simulation_results',
                           selected_harmonics=None):
    """
    Perform Monte Carlo simulations with random time shifts.
    Saves results to CSV files for further analysis.
    """
    if selected_harmonics is None:
        selected_harmonics = list(range(1, 101))
    max_harmonic = max(selected_harmonics)
    
    # Make directory
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    
    # Create a timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    signal_name = str(signal_func.__name__)
    batch_dir = os.path.join(output_dir, f"{signal_name}_N{N}_{timestamp}")
    os.makedirs(batch_dir, exist_ok=True)
    
    # Save simulation params
    params_dict = {
        'num_runs': num_runs,
        'N': N,
        'fs': fs,
        'T_total': T_total,
        'sample_rate': sample_rate,
        'selected_harmonics': selected_harmonics
    }
    params_df = pd.DataFrame([params_dict])
    params_df.to_csv(os.path.join(batch_dir, 'simulation_params.csv'), index=False)
    
    # For collecting Z-metrics
    all_z_metrics = {h: [] for h in selected_harmonics}
    all_taus = np.zeros((num_runs, N))
    
    for run in range(num_runs):
        if run % 10 == 0:
            print(f"Running simulation {run+1}/{num_runs}...")
        
        ts = 1/fs
        tau = np.zeros(N)
        tau[1:] = np.random.uniform(0, ts, N-1)
        all_taus[run] = tau
        
        # Run single simulation
        results = run_simulation(signal_func, signal_params, N=N, fs=fs, 
                                 T_total=T_total, sample_rate=sample_rate,
                                 tau=tau, plot=False, max_harmonic=max_harmonic)
        
        # Store z metrics
        for h in selected_harmonics:
            val = results['z_metrics'].get(h, 0.0)
            all_z_metrics[h].append(val)
        
        # Save per-run harmonic magnitudes if desired
        run_data = []
        for h in selected_harmonics:
            n_harm = results['n_harmonics'].get(h, {})
            single_harm = results['single_harmonics'].get(h, {})
            run_data.append({
                'harmonic': h,
                'I_h': n_harm.get('magnitude', 0.0),
                'I1_h': single_harm.get('magnitude', 0.0),
                'Z_h': results['z_metrics'].get(h, 0.0)
            })
        
        if run_data:
            df_run = pd.DataFrame(run_data)
            df_run.to_csv(os.path.join(batch_dir, f'run_{run+1}_harmonics.csv'), index=False)
    
    # Save aggregated Z metrics
    z_metrics_df = pd.DataFrame(all_z_metrics)
    z_metrics_df.to_csv(os.path.join(batch_dir, 'all_z_metrics_by_harmonic.csv'), index=False)
    
    tau_df = pd.DataFrame(all_taus, columns=[f'tau_{i+1}' for i in range(N)])
    tau_df.to_csv(os.path.join(batch_dir, 'all_taus.csv'), index=False)
    
    with open(os.path.join(batch_dir, 'selected_harmonics.txt'), 'w') as f:
        f.write(','.join(map(str, selected_harmonics)))
    
    print(f"Monte Carlo simulation completed. Results in: {batch_dir}")
    return batch_dir



def _worker(k, signal_configs, num_runs, selected_harmonics, output_dir):
    """
    A top-level worker function. Multiprocessing can pickle/unpickle this
    because it's not nested inside another function.
    """
    cfg = signal_configs[k]
    func = cfg['func']
    params = cfg['params']
    N = cfg.get('N', 5)
    fs = cfg.get('fs', 10.0)
    T_total = cfg.get('T_total', 0.5)
    sample_rate = cfg.get('sample_rate', 1000)

    result_dir = monte_carlo_simulation(
        func, params,
        num_runs=num_runs,
        N=N, fs=fs, T_total=T_total,
        sample_rate=sample_rate,
        output_dir=output_dir,
        selected_harmonics=selected_harmonics
    )
    return result_dir


def run_custom_simulation_batch(signal_configs, num_runs=1000,
                                selected_harmonics=None,
                                output_dir='simulation_results',
                                n_processes=None):
    """
    Runs multiple configurations in parallel (or single-threaded).
    """
    if selected_harmonics is None:
        selected_harmonics = [1, 3, 5, 7, 101]

    if n_processes is None:
        n_processes = min(len(signal_configs), multiprocessing.cpu_count())

    keys = list(signal_configs.keys())

    if n_processes > 1:
        with multiprocessing.Pool(n_processes) as pool:
            # Use functools.partial to avoid a lambda
            worker_func = partial(_worker,
                                  signal_configs=signal_configs,
                                  num_runs=num_runs,
                                  selected_harmonics=selected_harmonics,
                                  output_dir=output_dir)
            results = pool.map(worker_func, keys)
    else:
        results = []
        for k in keys:
            results.append(_worker(k, signal_configs, num_runs, selected_harmonics, output_dir))

    return results

# def run_custom_simulation_batch(signal_configs, num_runs=1000,
#                                 selected_harmonics=None, 
#                                 output_dir='simulation_results',
#                                 n_processes=None):

#     if selected_harmonics is None:
#         selected_harmonics = [1, 3, 5, 7, 101]
    
#     if n_processes is None:
#         n_processes = min(len(signal_configs), multiprocessing.cpu_count())
    
#     def _worker(cfg_key):
#         cfg = signal_configs[cfg_key]
#         func = cfg['func']
#         params = cfg['params']
#         N = cfg.get('N', 5)
#         fs = cfg.get('fs', 10.0)
#         T_total = cfg.get('T_total', 0.5)
#         sample_rate = cfg.get('sample_rate', 1000)
#         print(f"Running {cfg_key} with N={N}, fs={fs} ...")
#         return monte_carlo_simulation(func, params, num_runs=num_runs, 
#                                       N=N, fs=fs, T_total=T_total,
#                                       sample_rate=sample_rate,
#                                       output_dir=output_dir,
#                                       selected_harmonics=selected_harmonics)
    
#     keys = list(signal_configs.keys())
#     if n_processes > 1:
#         with multiprocessing.Pool(n_processes) as pool:
#             results = pool.map(_worker, keys)
#     else:
#         results = []
#         for k in keys:
#             results.append(_worker(k))
    
#     return results