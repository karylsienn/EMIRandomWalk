## EMI Random Walk

Simulations for Random walk modelling of CM current, extended to include a suite of signals.
The purpose of the repository is to verify that for certain classes of signals, we can model their behaviour for $N$ signals using Random Walk approach.

### Usage

To use this simulation tool, follow these steps:

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/emi-random-walk.git
   cd emi-random-walk
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run a simulation:
   ```bash
   python simulate.py --signal sine --amplitude 1.0 --frequency 1e6
   ```

4. Analyze the results:
   ```bash
   python analyze.py --results-file output.csv
   ```

### Implemented Signals

The following signal types are currently implemented:

1. **Sine Wave** - A basic sinusoidal signal with configurable amplitude and frequency
2. **Square Wave** - Square wave with adjustable duty cycle
3. **Triangle Wave** - Triangular waveform with configurable symmetry
4. **Pulse** - Single or multiple pulses with adjustable width and spacing
5. **Gaussian Pulse** - Gaussian-shaped pulse with configurable width
6. **Random Noise** - White or colored noise with adjustable spectral properties
7. **AM Signal** - Amplitude modulated signal with configurable carrier and modulation
8. **FM Signal** - Frequency modulated signal with configurable parameters
9. **PWM Signal** - Pulse width modulated signal with adjustable duty cycle

Each signal type can be configured through command-line arguments or configuration files.
