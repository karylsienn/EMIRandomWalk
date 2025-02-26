# simulation.py
import numpy as np
import matplotlib.pyplot as plt

from analysis import analyze_harmonics, calculate_z_metric

def simulate_signals(signal_func, signal_params, N, fs, T_total, sample_rate, tau=None):
    """
    Simulate N shifted damped signals and their sum.
    Returns time array, individual signals, composite, and time shifts.
    """
    ts = 1 / fs
    t = np.arange(0, T_total, 1/sample_rate)
    if tau is None:
        tau = np.zeros(N)
        tau[1:] = np.random.uniform(0, ts, N-1)
    signals = np.zeros((N, len(t)))
    for i in range(N):
        signals[i] = signal_func(t - tau[i], fs=fs, **signal_params)
    composite = np.sum(signals, axis=0)
    return t, signals, composite, tau

def run_simulation(signal_func, signal_params, N=5, fs=10.0,
                   T_total=1.0, sample_rate=1000, tau=None,
                   plot=True, min_harmonic=1, max_harmonic=100):
    """
    Wrapper that simulates signals, analyzes harmonics, and computes the Z metric.
    """
    t, signals, composite, tau = simulate_signals(
        signal_func, signal_params, N, fs, T_total, sample_rate, tau=tau
    )
    
    # Analyze the composite
    freq, fft_mag, n_harmonics = analyze_harmonics(
        t, composite, fs, sample_rate, plot=plot,
        min_harmonic=min_harmonic, max_harmonic=max_harmonic
    )
    
    # Analyze just the first signal
    _, _, single_harmonics = analyze_harmonics(
        t, signals[0], fs, sample_rate, plot=False,
        min_harmonic=min_harmonic, max_harmonic=max_harmonic
    )
    
    # Compute Z metric
    z_metrics = calculate_z_metric(single_harmonics, n_harmonics)
    
    # Optionally plot all signals + composite
    if plot:
        plt.figure(figsize=(12, 8))
        for i in range(N):
            plt.plot(t, signals[i], alpha=0.5, label=f'Signal {i+1} (τ={tau[i]:.4f})')
        plt.plot(t, composite, 'k', linewidth=2, label='Composite')
        plt.xlabel('Time (s)')
        plt.ylabel('Amplitude')
        plt.title(f'Simulation of {N} Shifted Signals (fs={fs} Hz)')
        plt.legend()
        plt.grid(True)
        plt.show()
        
        # Plot Z metric
        plt.figure(figsize=(12, 6))
        harmonics_sorted = sorted(z_metrics.keys())
        z_values = [z_metrics[h] for h in harmonics_sorted]
        plt.bar(harmonics_sorted, z_values)
        plt.xlabel('Harmonic Number')
        plt.ylabel('Z[h]')
        plt.title('Z Metric')
        plt.grid(True, alpha=0.3)
        plt.show()
    
    return {
        'time': t,
        'signals': signals,
        'composite': composite,
        'tau': tau,
        'frequency': freq,
        'fft_magnitude': fft_mag,
        'n_harmonics': n_harmonics,
        'single_harmonics': single_harmonics,
        'z_metrics': z_metrics
    }