import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
from scipy.special import jv  # Bessel functions
from scipy.integrate import quad, simpson
import warnings
from statsmodels.distributions.empirical_distribution import ECDF

#------------------------------------------------------------------------------
# Analytical ECDF Functions
#------------------------------------------------------------------------------

def ecdf_integrand(t, r, N):
    """
    Integrand function for the analytical ECDF calculation.
    
    Parameters:
    -----------
    t : float
        Integration variable
    r : float
        Value at which to evaluate the ECDF
    N : int
        Number of signals
        
    Returns:
    --------
    float: Value of the integrand at t
    """
    return jv(0, t)**(N-1) * jv(1, r*t)

def analytical_ecdf_value(r, N, integration_limit=100, tol=1e-8):
    """
    Calculate a single value of the analytical ECDF for given r and N.
    
    Parameters:
    -----------
    r : float
        Value at which to evaluate the ECDF
    N : int
        Number of signals
    integration_limit : float
        Upper limit for the numerical integration (approximating infinity)
    tol : float
        Tolerance for the numerical integration
        
    Returns:
    --------
    float: ECDF value at r
    """
    # For values greater than or equal to N-1, ECDF = 1
    if r >= N-1:
        return 1.0
    
    # For very small r, ECDF approaches 0
    if r < 1e-10:
        return 0.0
    
    # Suppress integration warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        
        # Compute the integral using SciPy's quad function
        integral, _ = quad(
            lambda t: ecdf_integrand(t, r, N), 
            0, 
            integration_limit,
            epsabs=tol,
            epsrel=tol,
            limit=1000
        )
    
    # Multiply by r as in the formula
    return r * integral

def compute_analytical_ecdf(r_values, N, cache_dir=None):
    """
    Compute the analytical ECDF for a range of r values.
    
    Parameters:
    -----------
    r_values : array_like
        Values at which to evaluate the ECDF
    N : int
        Number of signals
    cache_dir : str, optional
        Directory to cache results. If None, results are not cached.
        
    Returns:
    --------
    numpy.ndarray: ECDF values corresponding to r_values
    """
    # Check if cached file exists
    if cache_dir is not None:
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir)
        cache_file = os.path.join(cache_dir, f'analytical_ecdf_N{N}.csv')
        
        if os.path.exists(cache_file):
            # Load from cache
            print(f"Loading analytical ECDF for N={N} from cache")
            df = pd.read_csv(cache_file)
            # Interpolate to match requested r_values
            cached_r = df['r'].values
            cached_ecdf = df['ecdf'].values
            
            # Simple linear interpolation
            result = np.interp(r_values, cached_r, cached_ecdf)
            return result
    
    # Compute ECDF values
    print(f"Computing analytical ECDF for N={N}...")
    ecdf_values = np.zeros_like(r_values, dtype=float)
    
    for i, r in enumerate(r_values):
        ecdf_values[i] = analytical_ecdf_value(r, N)
        if i % 10 == 0:  # Print progress every 10 points
            print(f"Computing point {i+1}/{len(r_values)}: r={r:.4f}, ECDF={ecdf_values[i]:.4f}")
    
    # Save to cache if requested
    if cache_dir is not None:
        df = pd.DataFrame({'r': r_values, 'ecdf': ecdf_values})
        df.to_csv(cache_file, index=False)
        print(f"Cached analytical ECDF for N={N} to {cache_file}")
    
    return ecdf_values

def precompute_analytical_ecdfs(r_max=10, num_points=1000, N_values=None, cache_dir='analytical_ecdfs'):
    """
    Precompute analytical ECDFs for different N values and save to files.
    
    Parameters:
    -----------
    r_max : float
        Maximum r value to compute
    num_points : int
        Number of points to compute
    N_values : list, optional
        List of N values to compute. If None, uses [3, 5, 8, 12]
    cache_dir : str
        Directory to save results
        
    Returns:
    --------
    dict: Dictionary mapping N values to DataFrames with ECDF values
    """
    if N_values is None:
        N_values = [3, 5, 8, 12]
    
    r_values = np.linspace(0, r_max, num_points)
    results = {}
    
    for N in N_values:
        print(f"Precomputing analytical ECDF for N={N}")
        ecdf_values = compute_analytical_ecdf(r_values, N, cache_dir=cache_dir)
        results[N] = pd.DataFrame({'r': r_values, 'ecdf': ecdf_values})
    
    return results


#------------------------------------------------------------------------------
# ECDF Analysis Functions
#------------------------------------------------------------------------------

def compute_and_plot_ecdf(simulation_dirs, output_dir=None, harmonics_to_plot=None, figsize=(15, 10),
                       show_analytical=True, analytical_cache_dir='analytical_ecdfs'):
    """
    Compute and plot ECDFs for selected harmonics across different signal types.
    
    Parameters:
    -----------
    simulation_dirs : list
        List of directories containing simulation results
    output_dir : str, optional
        Directory to save the ECDF plots. If None, plots are displayed but not saved.
    harmonics_to_plot : list, optional
        List of harmonics to plot. If None, harmonics are read from the simulation files.
    figsize : tuple
        Figure size for the plots
    show_analytical : bool
        Whether to include analytical ECDF curves
    analytical_cache_dir : str
        Directory containing cached analytical ECDF values
    """
    # Group simulation directories by signal type
    signal_type_dirs = {}
    for directory in simulation_dirs:
        signal_type = os.path.basename(directory).split('_')[0]
        if signal_type not in signal_type_dirs:
            signal_type_dirs[signal_type] = []
        signal_type_dirs[signal_type].append(directory)
    
    # If no harmonics specified, read from first directory
    if harmonics_to_plot is None:
        # Try to read from selected_harmonics.txt
        for first_dir in simulation_dirs:
            harmony_file = os.path.join(first_dir, 'selected_harmonics.txt')
            if os.path.exists(harmony_file):
                with open(harmony_file, 'r') as f:
                    harmonics_to_plot = list(map(int, f.read().split(',')))
                break
        
        # If still None, try to infer from Z metrics files
        if harmonics_to_plot is None:
            for first_dir in simulation_dirs:
                z_metrics_file = os.path.join(first_dir, 'all_z_metrics_by_harmonic.csv')
                if os.path.exists(z_metrics_file):
                    z_df = pd.read_csv(z_metrics_file)
                    harmonics_to_plot = sorted([int(h) for h in z_df.columns])
                    break
    
    if harmonics_to_plot is None:
        print("No harmonics specified and couldn't infer from files.")
        return
    
    # Define a color map for different signal types
    color_map = {
        'sinusoid': 'blue',
        'exponential': 'red',
        'gaussian': 'green',
        'chirp': 'purple',
        'square': 'orange'
    }
    
    # Define linestyles for analytical curves
    analytical_linestyles = {
        3: '-',
        5: '--',
        8: '-.',
        12: ':'
    }
    
    # Calculate number of rows and columns for subplots
    n_harmonics = len(harmonics_to_plot)
    n_cols = min(3, n_harmonics)
    n_rows = (n_harmonics + n_cols - 1) // n_cols
    
    # Create figure and subplots
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    if n_harmonics == 1:
        axes = np.array([axes])
    axes = axes.flatten()
    
    # Process each signal type
    for signal_type, directories in signal_type_dirs.items():
        # Aggregate Z metrics for each harmonic across all runs
        z_values = {h: [] for h in harmonics_to_plot}
        
        # Track N values encountered for each simulation
        n_values = set()
        
        for directory in directories:
            z_metrics_file = os.path.join(directory, 'all_z_metrics_by_harmonic.csv')
            params_file = os.path.join(directory, 'simulation_params.csv')
            
            # Extract N value from params file if it exists
            if os.path.exists(params_file):
                try:
                    params_df = pd.read_csv(params_file)
                    if 'N' in params_df.columns:
                        n_values.add(params_df['N'].iloc[0])
                except Exception as e:
                    print(f"Error reading params file {params_file}: {e}")
            
            # Extract Z metrics
            if os.path.exists(z_metrics_file):
                z_df = pd.read_csv(z_metrics_file)
                
                # Extract relevant harmonics
                for h in harmonics_to_plot:
                    if str(h) in z_df.columns:
                        z_values[h].extend(z_df[str(h)].tolist())
        
        # Plot ECDF for each harmonic
        for i, h in enumerate(harmonics_to_plot):
            if i < len(axes) and len(z_values[h]) > 0:
                # Compute ECDF
                ecdf = ECDF(z_values[h])
                
                # Plot ECDF
                axes[i].step(ecdf.x, ecdf.y, where='post', 
                            label=f"{signal_type}", 
                            color=color_map.get(signal_type, 'black'),
                            linewidth=2)
                
                axes[i].set_title(f'Harmonic {h}')
                axes[i].set_xlabel('Z[h]')
                axes[i].set_ylabel('ECDF')
                axes[i].grid(True, alpha=0.3)
    
    # Add analytical ECDFs if requested
    if show_analytical:
        standard_n_values = [3, 5, 8, 12]
        n_values_to_plot = list(n_values) if n_values else standard_n_values
        
        for i, h in enumerate(harmonics_to_plot):
            if i < len(axes):
                # Get current x limits to set appropriate range for analytical curves
                xlim = axes[i].get_xlim()
                r_values = np.linspace(0, min(xlim[1], 12), 1000)  # Cap at 12 for performance
                
                for N in n_values_to_plot:
                    # Only include analytical curves for N values present in simulations
                    if N in standard_n_values:
                        try:
                            ecdf_values = compute_analytical_ecdf(r_values, N, cache_dir=analytical_cache_dir)
                            
                            # Plot analytical ECDF
                            axes[i].plot(r_values, ecdf_values, 
                                        label=f"Analytical N={N}",
                                        color='black',
                                        linestyle=analytical_linestyles.get(N, '-'),
                                        linewidth=1.5,
                                        alpha=0.7)
                        except Exception as e:
                            print(f"Error plotting analytical ECDF for N={N}: {e}")
                
                # Add legend
                axes[i].legend(loc='best')
    else:
        # Add legend without analytical curves
        for i in range(min(n_harmonics, len(axes))):
            axes[i].legend(loc='best')
    
    # Hide unused subplots
    for i in range(n_harmonics, len(axes)):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    
    # Save figure if output directory is specified
    if output_dir is not None:
        if not os.path.exists(output_dir):
            os.makedirs(output_dir,exist_ok=True)
        plt.savefig(os.path.join(output_dir, 'ecdf_plots.png'), dpi=300)
        plt.savefig(os.path.join(output_dir, 'ecdf_plots.pdf'))
        print(f"ECDF plots saved to {output_dir}")
    
    plt.show()

if __name__=="__main__":
    # Precompute analytical ECDFs (only needs to be done once)
    results = precompute_analytical_ecdfs(r_max=12, num_points=1000, N_values=[3, 5, 8, 12])
    print("Precomputation complete")