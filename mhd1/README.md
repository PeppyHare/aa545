# Linearized Ideal MHD Stability Analysis

This module implements an axisymmetric finite difference solver for the linearized ideal MHD equations. The solver is applied to a force-free spheromak configuration to determine the stability criterion (as a function of L/R).

## Running the Script

To run the investigation, execute the `run_spheromak_equilibrium.py` script. Requires a Python install (NumPy 1.20.2 supports either Python 3.7 or Python 3.8) with either `pip` or `poetry` for depdency installation. To install dependencies, follow the appropriate section:

# Ideal MHD 3D MacCormack Finite Difference Integrator

This module implements a Lax-Wendroff type finite difference method to solve the nonlinear, time-dependent ideal MHD equations. Two test problems are used to validate the model:

- The Brio-Wu shock tube problem. A 1.5D Riemann problem is set up which generates five different wave solutions. This tests the model's ability to capture waves and handle discontinuities.
- A screw pinch plasma with a parabolic axial current density and constant axial magnetic field. As an equilibrium configuration, the solution should remain stable for certain values of the safety factor q. Initializing a perturbation in velocity helps to incite unstable kink modes.

## Running the Script

To run the investigation, execute the `run_screw_pinch.py` script. Requires a Python install (NumPy 1.20.2 supports either Python 3.7 or Python 3.8) with either `pip` or `poetry` for depdency installation. To install dependencies, follow the appropriate section:

### Using Poetry

With [poetry](https://python-poetry.org/) installed, dependencies can be automatically installed into a virtual environment by running the following in a command prompt:

```
poetry update
poetry run ./run_two_beam_instability.py
```

Note that Poetry will only successfully install LLVMlite if the installed system Python version is in the range 3.7-3.8. To make use of a Python installation besides the system python, first run:

```
poetry env use </path/to/python/executable>
```

### Using Pip

With [pip](https://pypi.org/project/pip/) installed, dependencies can be installed in a local virtual environment by executing the following commands:

**MacOS and Linux**:


```
python3 -m pip install virtualenv
python3 -m virtualenv .venv
source .venv/bin/activate.sh
python3 -m pip install -r requirements.txt
python3 run_two_beam_instability.py
```

**Windows:**

```
py -m pip install virtualenv
py -m virtualenv env
source .\env\Scripts\activate
py -m pip install -r requirements.txt
py run_two_beam_instability.py
```