# Finite Difference Scheme Investigation

This module investigates the instability and accuracy of the following finite difference schemes, applied to a model hyperbolic equation (the linear advection equation):

- Forward-time, centered-space
- Simple upwind difference
- Lax-Friedrichs
- Lax-Wendroff

## Running the Script

To run the investigation, execute the `run_ftcs.py` script. Requires a Python install (NumPy 1.20.2 supports either Python 3.7 or Python 3.8) with either `pip` or `poetry` for depdency installation. To install dependencies, follow the appropriate section:

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