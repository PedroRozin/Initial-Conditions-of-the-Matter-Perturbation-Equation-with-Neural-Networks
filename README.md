# Initial Conditions of the Matter Perturbation Equation using Neural Networks

This repository contains utilities, notebooks, and scripts used to study the matter perturbation equation in modified gravity and $\Lambda$ CDM, including numerical solvers and post-processing tools for quantities such as $\delta(a)$, $f(k,z)$, $\sigma_8(z)$, and $f\sigma_8(z)$.

The main result of this repo is the trained neural network in the `outputs/` folder. If you just want to use them, there is no need to clone this repository; you can just download them and use any solver as an example of use. If you want to implement the algorithm to train another NN or to verify its validity, follow the Recommended Installation section. To do this, you must also install the modified CLASS in https://github.com/PedroRozin/Modified_CLASS
If you would like to use the solver, but you are not interested in compare the $\Lambda$ CDM solutions nor the Initial Conditions with Boltzmann codes, then there is no need to install the modified CLASS.

## Project layout

- `source/`: Python modules used by the notebooks and scripts.
- `notebooks/`: Jupyter notebooks for experiments, plots, and analysis.
- `scripts/`: standalone scripts.
- `outputs/`: saved CSV files, trained models, and derived results.
- `figures/`: figure assets and documentation.

## Recommended installation

The cleanest way to use the code is to install the repository in editable mode. This makes the modules in `source/` importable without manually editing `sys.path` in every notebook.

```bash
git clone https://github.com/PedroRozin/Initial-Conditions-of-the-Matter-Perturbation-Equation-with-Neural-Networks.git
```

This downloads the repository to your machine.

```bash
cd Initial-Conditions-of-the-Matter-Perturbation-Equation-with-Neural-Networks
```

This moves the terminal into the repository root, where `pyproject.toml` lives.

### Option 1: use a local virtual environment

```bash
python -m venv .venv
```

This creates a fresh virtual environment in the local `.venv/` folder. The environment lives on your machine only; it is not committed to GitHub.

```bash
source .venv/bin/activate
```

This activates the environment so `python` and `pip` point to the isolated interpreter.

```bash
python -m pip install --upgrade pip
```

This updates `pip` inside the environment before installing the project.

```bash
pip install -r requirements.txt
```

This installs the Python dependencies used by the repository, including the custom CLASS fork that provides the `classy` module.

```bash
pip install -e .
```

This installs the repository in editable mode. Any changes you make in `source/` are immediately visible without reinstalling.

### Option 2: use an existing environment

If you already have a conda environment, a different `venv`, or a system Python installation, activate that environment first and then run:

```bash
pip install -r requirements.txt
pip install -e .
```

This installs the dependencies into your chosen environment and then installs the repository itself in editable mode.

### Option 3: no installation, temporary notebook workflow

If you do not want to install the project at all, you can temporarily add the `source/` directory to `sys.path` inside a notebook or script:

```python
import sys
sys.path.append('your_path_to_source')
```

This is convenient for quick tests, but it is less portable than installing the project with `pip install -e .`.

## Dependencies

The repository uses the following Python packages in the source code and notebooks:

- `numpy`
- `matplotlib`
- `pandas`
- `scipy`
- `torch`
- `tqdm`
- `joblib`
- `scikit-learn`
- `Cython`

The code also imports `classy` for cosmology calculations. In this repository, that dependency is provided by a modified CLASS fork stored at `https://github.com/PedroRozin/Modified_CLASS`, and `requirements.txt` installs it directly from the `class_public/` subdirectory on GitHub.

That installation step requires a working C/C++ toolchain and `make`, because CLASS is compiled from source during installation.

## If the CLASS build fails, install it manually with:

```bash
git clone https://github.com/PedroRozin/Modified_CLASS.git
cd Modified_CLASS/class_public
make clean
make -j$(nproc)
cd python
python -m pip install .
```
If this also fails, install it as you usually install the original CLASS from `https://github.com/lesgourg/class_public/tree/master`

## Usage examples


Example solver setup:

```python
import numpy as np
import main_functions as ft
from delta_solver_mg_pedro import VectorizedDeltaSolver

h = 0.68
k_array = np.logspace(np.log10(0.006), np.log10(0.2), 100) / h

solver = VectorizedDeltaSolver(
    k_array=k_array,
    h=h,
    Om_m_0=0.3,
    b=0.001,
)
```

This creates a vectorized solver using your chosen cosmological parameters and wavenumber grid.

## Running notebooks

1. Activate the environment you want to use.
2. Make sure the repository is installed with `pip install -e .` or that `source/` is on `sys.path`.
3. Open the notebooks in `notebooks/` and run the cells in order.

## Notes

- Do not commit `.venv/` to the repository.
- Keep `pyproject.toml` under version control so other users can install the project consistently.
- Large generated outputs should stay in `outputs/` only if they are meant to be shared; otherwise consider ignoring temporary files locally.


