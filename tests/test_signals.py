import pytest
import numpy as np
from emcranw.signals import (
    damped_sinusoid,
    damped_exponential,
    gaussian_pulse,
    damped_chirp,
    damped_square_wave
)

@pytest.mark.parametrize("func,kwargs", [
    (damped_sinusoid, {"A": 1.0, "f": 5.0, "lambda_": 3.0, "phase": 0.0}),
    (damped_exponential, {"A": 1.0, "lambda_": 3.0}),
    (gaussian_pulse, {"A": 1.0, "t0": 0.05, "sigma": 0.01}),
    (damped_chirp, {"A": 1.0, "f0": 2.0, "f1": 10.0, "lambda_": 3.0}),
    (damped_square_wave, {"A": 1.0, "duty": 0.5, "lambda_": 3.0}),
])
def test_signal_shape(func, kwargs):
    """Test that each signal function returns correct shape for a given time array."""
    t = np.linspace(0, 1, 100)
    signal_out = func(t, fs=10.0, **kwargs)
    assert len(signal_out) == len(t), f"Signal shape mismatch for {func.__name__}"

def test_damped_sinusoid_value():
    """Test known values of damped_sinusoid at specific times."""
    t = np.array([0.0, 0.1])
    signal_out = damped_sinusoid(t, A=1.0, f=5.0, lambda_=3.0, phase=0.0, fs=10.0)
    # At t=0, sin(0 + 0)=0 => output=0
    assert np.isclose(signal_out[0], 0.0, atol=1e-9), "damped_sinusoid not zero at t=0"
    # Check sign or approximate magnitude at t=0.1
    # This is just an example; you can do more specific checks if you want
    assert -1.0 <= signal_out[1] <= 1.0