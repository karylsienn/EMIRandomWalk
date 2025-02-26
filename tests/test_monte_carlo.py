import pytest
import shutil
import os
from emcranw.monte_carlo import run_custom_simulation_batch
from emcranw.signals import damped_sinusoid

@pytest.fixture
def temp_output_dir(tmp_path):
    """Fixture to provide a temporary directory for test output."""
    yield tmp_path
    # Cleanup after test
    shutil.rmtree(str(tmp_path), ignore_errors=True)

def test_run_custom_simulation_batch(temp_output_dir):
    """Check that a batch run creates the expected folder structure."""
    configs = {
        'sinusoid_test': {
            'func': damped_sinusoid,
            'params': {'A': 1.0, 'f': 5.0, 'lambda_': 3.0},
            'N': 2,
            'fs': 5.0,
            'T_total': 0.1,
            'sample_rate': 100
        }
    }
    results = run_custom_simulation_batch(
        configs,
        num_runs=2,
        selected_harmonics=[1, 2],
        output_dir=str(temp_output_dir),
        n_processes=1  # single process to keep it simple
    )

    # 'results' is a list of directories created for each config
    assert len(results) == len(configs)
    # Check that the directory actually got created
    for folder in results:
        assert os.path.exists(folder), f"Simulation output folder was not created: {folder}"
        # Check for expected CSV files, e.g., 'simulation_params.csv'
        params_csv = os.path.join(folder, 'simulation_params.csv')
        assert os.path.isfile(params_csv), f"simulation_params.csv not found in {folder}"
        z_metric_csv = os.path.join(folder, 'all_z_metrics_by_harmonic.csv')
        assert os.path.isfile(z_metric_csv), f"all_z_metrics_by_harmonic.csv not found in {folder}"