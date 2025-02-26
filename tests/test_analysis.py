import pytest
import numpy as np
from emcranw.analysis import analyze_harmonics, calculate_z_metric
from emcranw.signals import damped_sinusoid

def test_analyze_harmonics_pure_sin():
    """Generate a pure sinusoid and check if the fundamental harmonic is identified properly."""
    sample_rate = 1000
    fs = 5.0  # fundamental frequency
    t = np.linspace(0, 1, sample_rate, endpoint=False)
    # Perfect sinusoid at 5 Hz
    signal = np.sin(2*np.pi*fs*t)

    freq_pos, fft_mag, harmonics = analyze_harmonics(
        t, signal, fs, sample_rate, plot=False, min_harmonic=1, max_harmonic=5
    )
    # Check that harmonic #1 is non-zero and is the largest
    assert 1 in harmonics, "Fundamental harmonic (1) not found"
    fundamental_mag = harmonics[1]['magnitude']
    assert fundamental_mag > 0.1, "Fundamental magnitude is unexpectedly small"
    # Check that other harmonics are negligible
    for h in [2, 3, 4, 5]:
        if h in harmonics:
            assert harmonics[h]['magnitude'] < fundamental_mag * 0.3, \
                f"Harmonic {h} unexpectedly large"

def test_calculate_z_metric():
    """Check that the Z metric is correct for known ratios."""
    single_harmonics = {
        1: {'complex': 1.0},
        2: {'complex': 2.0},
        3: {'complex': 0.0},  # zero in reference
    }
    n_harmonics = {
        1: {'complex': 1.0},
        2: {'complex': 4.0},
        3: {'complex': 1.0},
    }
    z_metrics = calculate_z_metric(single_harmonics, n_harmonics)
    # For harmonic 1: (1/1 - 1) = 0
    assert np.isclose(z_metrics[1], 0.0)
    # For harmonic 2: (4.0 / 2.0 - 1) = 1.0
    assert np.isclose(z_metrics[2], 1.0)
    # For harmonic 3: reference=0 => we said if I1_h=0, Z=0.0 or special handling?
    # Depending on your code, check the correct behavior (some do 0, some do undefined)
    assert np.isclose(z_metrics[3], 0.0), "Expected 0 if reference amplitude=0"