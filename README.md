## EMI Random Walk

Simulations for random-walk modelling of common-mode (CM) current, extended to include a suite of damped signal types. The repository verifies that for certain classes of signals, their aggregate behaviour across $N$ converters can be modelled using a Random Walk approach — including an analytical ECDF derived from Bessel functions.

### Related Publication

The code in this repository was used to generate simulations presented in the following paper:

> Karol Niewiadomski, Erjon Ballukja, Mattia Simonazzi, Leonardo Sandrolini, and Jacek Bojarski, "Extension of Random Walk Modelling of CM Current Harmonics in Multi-converter Setups," in *2025 International Symposium on Electromagnetic Compatibility – EMC Europe*, 2025, pp. 916–921. [doi:10.1109/EMCEurope61644.2025.11176374](https://doi.org/10.1109/EMCEurope61644.2025.11176374).

```bibtex
@INPROCEEDINGS{11176374,
  author={Niewiadomski, Karol and Ballukja, Erjon and Simonazzi, Mattia and Sandrolini, Leonardo and Bojarski, Jacek},
  booktitle={2025 International Symposium on Electromagnetic Compatibility – EMC Europe},
  title={Extension of Random Walk Modelling of CM Current Harmonics in Multi-converter Setups},
  year={2025},
  pages={916-921},
  doi={10.1109/EMCEurope61644.2025.11176374}
}
```

If you use this code in your research, please cite the above publication.

### Package Structure

```
emcranw/
├── __init__.py
├── main.py          # Entry point: run_multiple_N(), compile_monte_carlo_results()
├── signals.py       # 13 damped signal types + DEFAULT_SIGNAL_CONFIGS
├── simulation.py    # simulate_signals(), run_simulation()
├── analysis.py      # analyze_harmonics(), calculate_z_metric()
├── monte_carlo.py   # monte_carlo_simulation(), run_custom_simulation_batch()
└── ecdf.py          # analytical_ecdf_value(), compute_and_plot_ecdf()
plots/               # R/Quarto scripts for paper figures
tests/               # pytest suite
```

### Usage

This is research code — no package installation is needed. Clone and run directly from the repository root.

**Run simulations** (for N = 3, 5, 8, 12 across all signal types):

```bash
python -m emcranw.main
```

This runs Monte Carlo simulations for each signal type at each N value, saves per-run results to `simulation_results/`, and compiles them into `compiled_ecdf_data.csv`.

**Generate example signal plots:**

```python
from emcranw.main import generate_example_signals
generate_example_signals()   # saves to example_signals/
```

**Run a single simulation:**

```python
from emcranw.simulation import run_simulation
from emcranw.signals import damped_sinusoid

result = run_simulation(
    signal_func=damped_sinusoid,
    signal_params={'A': 1.0, 'f': 50.0, 'lambda_': 60.0, 'phase': 0.0},
    N=5, fs=10.0, T_total=0.1, sample_rate=2000,
    plot=True
)
```

**Compute and plot ECDFs:**

```python
from emcranw.ecdf import compute_and_plot_ecdf, compute_analytical_ecdf

# Compute ECDF from simulation result directories
compute_and_plot_ecdf(
    simulation_dirs=['simulation_results/sinusoid_N5_*'],
    output_dir='ecdf_plots',
    harmonics_to_plot=[1, 3, 5, 7, 11, 19]
)

# Compute analytical ECDF values
from emcranw.ecdf import analytical_ecdf_value
ecdf_val = analytical_ecdf_value(r=2.5, N=5)
```

### Implemented Signal Types

The following 13 damped signal types are available in `emcranw.signals`:

| Signal | Function | Key Parameters |
|--------|----------|----------------|
| Damped sinusoid | `damped_sinusoid` | A, f, lambda_, phase |
| Damped exponential | `damped_exponential` | A, lambda_ |
| Gaussian pulse | `gaussian_pulse` | A, t0, sigma |
| Damped chirp | `damped_chirp` | A, f0, f1, lambda_ |
| Damped square wave | `damped_square_wave` | A, duty, lambda_ |
| Damped sawtooth | `damped_sawtooth` | A, lambda_ |
| Damped triangle | `damped_triangle` | A, lambda_ |
| Lorentzian pulse | `lorentzian_pulse` | A, t0, gamma |
| Damped multi-tone | `damped_multi_tone` | A, freqs, amps, lambda_ |
| Ricker wavelet | `ricker_wavelet` | A, t0, sigma |
| Half-sine pulse | `half_sine_pulse` | A, width, t0, lambda_ |
| Log-sweep chirp | `log_sweep_chirp` | A, f0, f1, lambda_ |
| Rectangular pulse train | `damped_rectangular_pulses` | A, num_pulses, pulse_width, lambda_ |

Each signal repeats with period `ts = 1/fs` and applies exponential decay within each period. Default configurations are available in `DEFAULT_SIGNAL_CONFIGS`.

### Tests

```bash
pip install pytest
pytest tests/
```

### R Plotting Scripts

The `plots/` directory contains R/Quarto scripts used to produce the paper figures:

- `plots_for_emc_europe.qmd` — ECDF plots and signal examples (requires `tidyverse`, `patchwork`, `tikzDevice`)
- `combine_analytical_cdf.R` — helper for combining analytical ECDF data
- `combine_ecdf.R` — helper for combining ECDF data

### Dependencies

Runtime dependencies (Python):
- numpy
- scipy
- pandas
- matplotlib
- statsmodels
- multiprocess (stdlib)

R dependencies (for `plots/`):
- tidyverse
- patchwork
- tikzDevice

### Output Files

| File / Directory | Description |
|------------------|-------------|
| `simulation_results/` | Per-signal-type Monte Carlo results (CSV) |
| `compiled_ecdf_data.csv` | All Monte Carlo results compiled into one file |
| `example_signals/` | Time-domain signal examples (CSV + PNG) |
| `analytical_ecdfs/` | Cached analytical ECDF values |
