import pytest
import os
import shutil
import numpy as np
from emcranw.ecdf import compute_and_plot_ecdf, compute_analytical_ecdf

@pytest.fixture
def temp_plot_dir(tmp_path):
    """Provide a temporary directory for ECDF plot outputs."""
    yield tmp_path
    shutil.rmtree(str(tmp_path), ignore_errors=True)

def test_compute_and_plot_ecdf(temp_plot_dir):
    """Basic test: supply a fake simulation directory list and see if it generates a plot."""
    # Suppose compute_and_plot_ecdf checks CSV files inside these dirs.
    # We'll create a small fake directory structure with minimal CSVs:
    sim_dir = temp_plot_dir / "sinusoid_N5_20230101_120000"
    sim_dir.mkdir(parents=True)
    # Create a minimal CSV file for the Z metrics
    (sim_dir / "all_z_metrics_by_harmonic.csv").write_text(
        "Z1,Z2\n0.1,0.2\n0.05,0.15\n"  # example data
    )

    # Now call compute_and_plot_ecdf
    output_dir = temp_plot_dir / "ecdf_plots"
    output_dir.mkdir()
    compute_and_plot_ecdf(
        [str(sim_dir)],
        output_dir=str(output_dir),
        harmonics_to_plot=[1, 2],
        figsize=(4, 3),
        show_analytical=False
    )
    # Check that at least one plot file was created
    files_in_output = list(output_dir.glob("*.png"))
    assert len(files_in_output) > 0, "No ECDF plot output created"

def test_compute_analytical_ecdf():
    """If you have a known closed-form, test for correctness at certain points."""
    r_values = np.linspace(0, 1, 10)
    # Hypothetical usage
    ecdf_vals = compute_analytical_ecdf(r_values, N=5, cache_dir='.')  
    # Just check shape or known boundary conditions
    assert len(ecdf_vals) == len(r_values)
    assert ecdf_vals[0] >= 0.0, "ECDF must be >= 0 at smallest r"
    assert ecdf_vals[-1] <= 1.0, "ECDF must be <= 1 at largest r"