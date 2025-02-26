# analysis.py
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy.fft import fft, fftfreq

def analyze_harmonics(t, signal, fs, sample_rate, plot=True, 
                      min_harmonic=1, max_harmonic=100):
    """
    Analyze the frequency components of a signal using FFT.
    Returns frequency array, FFT magnitudes, and a dict of harmonic info.
    """
    fft_result = fft(signal)
    freq = fftfreq(len(t), 1/sample_rate)
    pos_mask = freq > 0
    freq_pos = freq[pos_mask]
    fft_pos = fft_result[pos_mask]
    fft_mag = np.abs(fft_pos) / len(t)

    peak_indices, _ = find_peaks(fft_mag, height=np.max(fft_mag)/20)
    harmonics = {}

    for n in range(min_harmonic, max_harmonic + 1):
        expected_freq = n * fs
        closest_idx = np.argmin(np.abs(freq_pos - expected_freq))
        if (closest_idx in peak_indices
            or np.abs(freq_pos[closest_idx] - expected_freq) / fs < 0.1):
            harmonics[n] = {
                'frequency': freq_pos[closest_idx],
                'magnitude': fft_mag[closest_idx],
                'complex': fft_pos[closest_idx],
                'harmonic_ratio': freq_pos[closest_idx] / (n * fs)
            }
        else:
            harmonics[n] = {
                'frequency': expected_freq,
                'magnitude': 0.0,
                'complex': 0.0,
                'harmonic_ratio': 1.0
            }

    if plot:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        # Time domain
        ax1.plot(t, signal)
        ax1.set_xlabel('Time (s)')
        ax1.set_ylabel('Amplitude')
        ax1.set_title('Time Domain Signal')
        ax1.grid(True)
        # Frequency domain
        ax2.plot(freq_pos, fft_mag)
        ax2.set_xlabel('Frequency (Hz)')
        ax2.set_ylabel('Magnitude')
        ax2.set_title('Frequency Domain')
        ax2.grid(True)
        # Mark the detected harmonics
        for n, harm in harmonics.items():
            ax2.axvline(x=harm['frequency'], color='r', linestyle='--', alpha=0.5)
            ax2.text(harm['frequency'], harm['magnitude'], f'H{n}', 
                     verticalalignment='bottom', horizontalalignment='center')
        plt.tight_layout()
        plt.show()

    return freq_pos, fft_mag, harmonics

def calculate_z_metric(single_harmonics, n_harmonics):
    """
    Calculate Z[h] = |I[h]/I1[h] - 1| for each harmonic.
    """
    z_metrics = {}
    common_harmonics = set(single_harmonics.keys()).intersection(n_harmonics.keys())
    for h in common_harmonics:
        I1_h = single_harmonics[h]['complex']
        I_h = n_harmonics[h]['complex']
        if I1_h != 0:
            z_metrics[h] = np.abs(I_h / I1_h - 1)
        else:
            z_metrics[h] = 0.0
    return z_metrics