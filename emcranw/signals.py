# signals.py
import numpy as np

def damped_sinusoid(t, A=1.0, f=1.0, lambda_=1.0, phase=0.0, fs=1.0):
    """
    A damped sinusoidal signal that repeats with period ts = 1/fs.
    """
    ts = 1 / fs
    t_mod = t % ts
    return A * np.sin(2*np.pi*f*t_mod + phase) * np.exp(-lambda_ * t_mod)

def damped_exponential(t, A=1.0, lambda_=1.0, fs=1.0):
    """
    A damped exponential pulse signal that repeats with period ts = 1/fs.
    """
    ts = 1 / fs
    t_mod = t % ts
    return A * np.exp(-lambda_ * t_mod)

def gaussian_pulse(t, A=1.0, t0=0.0, sigma=0.1, fs=1.0):
    """
    A Gaussian pulse signal that repeats with period ts = 1/fs.
    """
    ts = 1 / fs
    t_mod = t % ts
    return A * np.exp(-((t_mod - t0)**2) / (2 * sigma**2))

def damped_chirp(t, A=1.0, f0=1.0, f1=5.0, lambda_=1.0, fs=1.0):
    """
    A damped chirp signal that repeats with period ts = 1/fs.
    """
    ts = 1 / fs
    t_mod = t % ts
    f_t = f0 + (f1 - f0) * (t_mod / ts)
    phase = 2 * np.pi * (f0 * t_mod + 0.5 * (f1 - f0) * t_mod**2 / ts)
    return A * np.sin(phase) * np.exp(-lambda_ * t_mod)

def damped_square_wave(t, A=1.0, duty=0.5, lambda_=1.0, fs=1.0):
    """
    A damped square wave signal that repeats with period ts = 1/fs.
    """
    ts = 1 / fs
    t_mod = t % ts
    square = np.zeros_like(t_mod)
    square[t_mod < duty * ts] = 1.0
    return A * square * np.exp(-lambda_ * t_mod)

def damped_sawtooth(t, A=1.0, lambda_=1.0, fs=1.0):
    """
    A damped sawtooth wave signal that repeats with period ts = 1/fs.
    
    Parameters:
    -----------
    t : array_like
        Time array
    A : float
        Amplitude
    lambda_ : float
        Damping factor
    fs : float
        Fundamental frequency (1/ts)
    
    Returns:
    --------
    array_like
        Damped sawtooth wave signal
    """
    ts = 1/fs
    t_mod = t % ts
    
    # Create sawtooth wave (linear ramp from 0 to 1)
    sawtooth = 2 * (t_mod / ts - np.floor(0.5 + t_mod / ts))
    
    return A * sawtooth * np.exp(-lambda_*t_mod)

def damped_triangle(t, A=1.0, lambda_=1.0, fs=1.0):
    """
    A damped triangle wave signal that repeats with period ts = 1/fs.
    
    Parameters:
    -----------
    t : array_like
        Time array
    A : float
        Amplitude
    lambda_ : float
        Damping factor
    fs : float
        Fundamental frequency (1/ts)
    
    Returns:
    --------
    array_like
        Damped triangle wave signal
    """
    ts = 1/fs
    t_mod = t % ts
    
    # Create triangle wave
    triangle = 2 * np.abs(2 * (t_mod / ts - np.floor(t_mod / ts + 0.5))) - 1
    
    return A * triangle * np.exp(-lambda_*t_mod)

def lorentzian_pulse(t, A=1.0, t0=0.0, gamma=0.1, fs=1.0):
    """
    A Lorentzian pulse signal that repeats with period ts = 1/fs.
    Used to model resonant phenomena.
    
    Parameters:
    -----------
    t : array_like
        Time array
    A : float
        Amplitude
    t0 : float
        Center time of pulse
    gamma : float
        Width parameter (half-width at half-maximum)
    fs : float
        Fundamental frequency (1/ts)
    
    Returns:
    --------
    array_like
        Lorentzian pulse signal
    """
    ts = 1/fs
    t_mod = t % ts
    
    return A / (1 + ((t_mod - t0) / gamma)**2)

def damped_multi_tone(t, A=1.0, freqs=[1.0, 2.0, 3.0], amps=[1.0, 0.5, 0.3], lambda_=1.0, fs=1.0):
    """
    A superposition of multiple damped sinusoids with different frequencies.
    
    Parameters:
    -----------
    t : array_like
        Time array
    A : float
        Overall amplitude scaling
    freqs : list
        List of frequencies for each tone
    amps : list
        List of relative amplitudes for each tone
    lambda_ : float
        Damping factor
    fs : float
        Fundamental frequency (1/ts)
    
    Returns:
    --------
    array_like
        Damped multi-tone signal
    """
    ts = 1/fs
    t_mod = t % ts
    
    # Normalize amplitudes
    amps = np.array(amps) / np.sum(amps)
    
    # Create multi-tone signal
    signal = np.zeros_like(t_mod)
    for f, amp in zip(freqs, amps):
        signal += amp * np.sin(2*np.pi*f*t_mod)
    
    return A * signal * np.exp(-lambda_*t_mod)

def ricker_wavelet(t, A=1.0, t0=0.0, sigma=0.1, fs=1.0):
    """
    A Ricker wavelet (Mexican hat wavelet) that repeats with period ts = 1/fs.
    This is the second derivative of a Gaussian.
    
    Parameters:
    -----------
    t : array_like
        Time array
    A : float
        Amplitude
    t0 : float
        Center time of wavelet
    sigma : float
        Width parameter
    fs : float
        Fundamental frequency (1/ts)
    
    Returns:
    --------
    array_like
        Ricker wavelet signal
    """
    ts = 1/fs
    t_mod = t % ts
    
    # Compute normalized time difference
    tau = (t_mod - t0) / sigma
    
    # Ricker wavelet formula
    return A * (1 - tau**2) * np.exp(-tau**2 / 2)

def half_sine_pulse(t, A=1.0, width=0.2, t0=0.0, lambda_=1.0, fs=1.0):
    """
    A half-sine pulse with exponential decay that repeats with period ts = 1/fs.
    
    Parameters:
    -----------
    t : array_like
        Time array
    A : float
        Amplitude
    width : float
        Width of the pulse as a fraction of ts
    t0 : float
        Start time of pulse
    lambda_ : float
        Damping factor
    fs : float
        Fundamental frequency (1/ts)
    
    Returns:
    --------
    array_like
        Half-sine pulse signal
    """
    ts = 1/fs
    t_mod = t % ts
    
    # Adjust t0 to be within [0, ts)
    t0 = t0 % ts
    
    # Calculate pulse duration
    pulse_duration = width * ts
    
    # Create half-sine pulse
    pulse = np.zeros_like(t_mod)
    mask = (t_mod >= t0) & (t_mod <= t0 + pulse_duration)
    pulse[mask] = np.sin(np.pi * (t_mod[mask] - t0) / pulse_duration)
    
    return A * pulse * np.exp(-lambda_*t_mod)

def log_sweep_chirp(t, A=1.0, f0=1.0, f1=10.0, lambda_=1.0, fs=1.0):
    """
    A logarithmic sweep chirp (exponential chirp) with exponential decay
    that repeats with period ts = 1/fs.
    
    Parameters:
    -----------
    t : array_like
        Time array
    A : float
        Amplitude
    f0 : float
        Start frequency
    f1 : float
        End frequency
    lambda_ : float
        Damping factor
    fs : float
        Fundamental frequency (1/ts)
    
    Returns:
    --------
    array_like
        Logarithmic sweep chirp signal
    """
    ts = 1/fs
    t_mod = t % ts
    
    # Avoid division by zero
    epsilon = 1e-10
    t_ratio = t_mod / (ts + epsilon)
    
    # Logarithmic frequency sweep
    if f0 <= 0 or f1 <= 0:
        raise ValueError("Frequencies must be positive for log sweep")
    
    # Logarithmically increasing instantaneous frequency
    freq_ratio = f1 / f0
    
    # Phase is the integral of frequency
    phase = 2 * np.pi * f0 * ts * (np.power(freq_ratio, t_ratio) - 1) / np.log(freq_ratio)
    
    return A * np.sin(phase) * np.exp(-lambda_*t_mod)

def damped_rectangular_pulses(t, A=1.0, num_pulses=3, pulse_width=0.05, lambda_=1.0, fs=1.0):
    """
    A series of rectangular pulses with exponential decay
    that repeats with period ts = 1/fs.
    
    Parameters:
    -----------
    t : array_like
        Time array
    A : float
        Amplitude
    num_pulses : int
        Number of pulses within one period
    pulse_width : float
        Width of each pulse as a fraction of ts/num_pulses
    lambda_ : float
        Damping factor
    fs : float
        Fundamental frequency (1/ts)
    
    Returns:
    --------
    array_like
        Rectangular pulse train signal
    """
    ts = 1/fs
    t_mod = t % ts
    
    # Calculate spacing between pulses
    pulse_spacing = ts / num_pulses
    pulse_duration = pulse_spacing * pulse_width
    
    # Create pulse train
    pulse_train = np.zeros_like(t_mod)
    
    for i in range(num_pulses):
        pulse_start = i * pulse_spacing
        pulse_end = pulse_start + pulse_duration
        mask = (t_mod >= pulse_start) & (t_mod < pulse_end)
        pulse_train[mask] = 1.0
    
    return A * pulse_train * np.exp(-lambda_*t_mod)


# Default signal configurations
DEFAULT_SIGNAL_CONFIGS = {
    'sinusoid': {
        'params': {'A': 1.0, 'f': 50.0, 'lambda_': 60.0, 'phase': 0.0},
        'N': 5,
        'fs': 10.0
    },
    'exponential': {
        'params': {'A': 1.0, 'lambda_': 60.0},
        'N': 5,
        'fs': 10.0
    },
    'gaussian': {
        'params': {'A': 1.0, 't0': 0.05, 'sigma': 0.02},
        'N': 5,
        'fs': 10.0
    },
    'chirp': {
        'params': {'A': 1.0, 'f0': 70.0, 'f1': 100.0, 'lambda_': 60.0},
        'N': 5,
        'fs': 10.0
    },
    'square': {
        'params': {'A': 1.0, 'duty': 0.5, 'lambda_': 60.0},
        'N': 5,
        'fs': 10.0
    },
    'sawtooth': {
        'params': {'A': 1.0, 'lambda_': 60.0},
        'N': 5,
        'fs': 10.0
    },
    'triangle': {
        'params': {'A': 1.0, 'lambda_': 60.0},
        'N': 5,
        'fs': 10.0
    },
    'lorentzian': {
        'params': {'A': 1.0, 't0': 0.05, 'gamma': 0.0002},
        'N': 5,
        'fs': 10.0
    },
    'multi_tone': {
        'params': {'A': 1.0, 'freqs': [3.0, 7.0, 11.0], 'amps': [1.0, 0.6, 0.3], 'lambda_': 3.0},
        'N': 5,
        'fs': 10.0
    },
    'ricker': {
        'params': {'A': 1.0, 't0': 0.05, 'sigma': 0.02},
        'N': 5,
        'fs': 10.0
    },
    'half_sine': {
        'params': {'A': 1.0, 'width': 0.0003, 't0': 0.0, 'lambda_': 60.0},
        'N': 5,
        'fs': 10.0
    },
    'log_sweep': {
        'params': {'A': 1.0, 'f0': 1.0, 'f1': 20.0, 'lambda_': 3.0},
        'N': 5,
        'fs': 10.0
    },
    'rect_pulses': {
        'params': {'A': 1.0, 'num_pulses': 3, 'pulse_width': 0.3, 'lambda_': 3.0},
        'N': 5,
        'fs': 10.0
    }
}

# ... existing code ...

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import numpy as np
    import random
    from matplotlib.gridspec import GridSpec
    
    # Signal function mapping
    signal_function_map = {
        'sinusoid': damped_sinusoid,
        'exponential': damped_exponential,
        'gaussian': gaussian_pulse,
        'chirp': damped_chirp,
        'square': damped_square_wave,
        'sawtooth': damped_sawtooth,
        'triangle': damped_triangle,
        'lorentzian': lorentzian_pulse,
        'multi_tone': damped_multi_tone,
        'ricker': ricker_wavelet,
        'half_sine': half_sine_pulse,
        'log_sweep': log_sweep_chirp,
        'rect_pulses': damped_rectangular_pulses
    }
    
    # Parameters for visualization
    T_total = 0.5  # Total time duration in seconds
    sample_rate = 1000  # Sampling rate in Hz
    time = np.linspace(0, T_total, int(T_total * sample_rate), endpoint=False)
    
    # Loop through each signal type
    for signal_name, config in DEFAULT_SIGNAL_CONFIGS.items():
        print(f"Plotting {signal_name}...")
        
        # Get the signal function and parameters
        signal_func = signal_function_map.get(signal_name)
        if signal_func is None:
            print(f"Function for signal type '{signal_name}' not found, skipping.")
            continue
        
        params = config['params']
        N = config['N']
        fs = config['fs']
        
        # Create figure
        fig = plt.figure(figsize=(10, 8))
        fig.suptitle(f"{signal_name.capitalize()} with Random Phase Shifts", fontsize=16)
        gs = GridSpec(2, 1, height_ratios=[2, 1])
        
        # Generate N signals with random time shifts
        signals = np.zeros((N, len(time)))
        tau = np.array([random.uniform(0, 1/fs) for _ in range(N)])
        
        # Plot individual signals
        ax1 = fig.add_subplot(gs[0])
        for i in range(N):
            # Shift time by random tau
            shifted_time = time - tau[i]
            signals[i] = signal_func(shifted_time, fs=fs, **params)
            ax1.plot(time, signals[i], alpha=0.7, label=f"Signal {i+1}, τ={tau[i]:.3f}")
        
        ax1.set_title(f"{N} Individual Signals")
        ax1.set_xlabel("Time (s)")
        ax1.set_ylabel("Amplitude")
        ax1.legend(loc="upper right")
        ax1.grid(True)
        
        # Plot composite signal
        ax2 = fig.add_subplot(gs[1])
        composite = np.sum(signals, axis=0)
        ax2.plot(time, composite, 'r-', linewidth=2)
        ax2.set_title("Composite Signal (Sum)")
        ax2.set_xlabel("Time (s)")
        ax2.set_ylabel("Amplitude")
        ax2.grid(True)
        
        # Adjust layout
        plt.tight_layout()
    
    # Show all plots
    plt.show()