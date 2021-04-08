# Instructions for Running pic1.py

## Configuring Settings

There are several flags and parameters that can be modified to control the behavior of the model:

### Parameters

- `n`: The total number of particles modeled.
- `x_min`, `x_max`: Determine the positions of the periodic boundary in the `x` coordinate. Initial particle positions are uniformly randomly distributed within this range.
- `v_min`, `v_max`: Determine the minimum and maximum allowable initial velocity.
- `v_fwhm`: The full width at half maximum of the Maxwellian distribution used to generate the initial particle velocities.
- `dt`: Value of time step.
- `t_max`: Simulation will evolve forward in time by steps of length `dt` until this `t_max`.

### Flags

- `plot_initial_distributions`: If set to True, the script will generate histograms of the initial particle distribution as well as plot the initial state in phase space.
- `animate_phase_space`: If set to True, the script will evolve the initial distribution forward in time and produce a live animation of the particles in phase space.
- `plot_snapshots`: If set to True, the script will plot snapshots of the particles in phase space at set time intervals.
- `trace_particles`: If set to True, the script will plot traces of the particles in phase at each time step.
- `compare_ke`: If set to True, the simulation will be performed using various values of `dt`. Changes in the total kinetic energy over time will be plotted for each value of the time step.

## Running the Script

Running the `pic1.py` script requires a Python install (NumPy 1.20.2 supports either Python 3.7 or Python 3.8) with either `pip` or `poetry` for depdency installation.

### Using Poetry

With [poetry](https://python-poetry.org/) installed, dependencies can be automatically installed into a virtual environment by running the following in a command prompt:

```
poetry update
poetry run ./pic1.py
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
python3 pic1.py
```

**Windows:**

```
py -m pip install virtualenv
py -m virtualenv env
source .\env\Scripts\activate
py -m pip install -r requirements.txt
py pic1.py
```