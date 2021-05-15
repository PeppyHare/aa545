# Instructions for Running Electrostatic PIC Code

## Running the Script

Running the any of the `run_*.py` scripts requires a Python install (NumPy 1.20.2 supports either Python 3.7 or Python 3.8) with either `pip` or `poetry` for depdency installation.

- `run_langmuir_shm.py` Initializes a pair of stationary charged particles.
- `run_two_beam_instability.py` Investigates the unstable behavior of two cold, counter-streaming beams of charged particles.
- `run_leapfrog_instability.py` Performs an investigation into the leapfrog instability by violating the CFL condition.

## Configuring Parameters

There are several flags and parameters that can be modified to control the behavior of the model. The configuration is stored as an inheritable Python class. The default parameters are given in `configuration.py`. Changes to the parameters can be made by inheriting from the `Configuration` class, as shown in the various sample `run_*.py` scripts.

## Generating Plots

The `plots.py` module contains a variety of useful functions for generating plots of the particle state and history, including animations.

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