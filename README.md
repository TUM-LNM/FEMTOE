# FEMTOE

This is a small Python project based on Jupyter containing small exercises about linear Finite
Elements. To install it locally, you need to have Python installed (>= 3.8). To set up the
project locally, create a python virtual environment, install jupyterlab and all required dependencies.

```
python -m venv venv
. venv/bin/activate
pip install jupyterlab
pip install -r requirements.txt
```

Then, you can start the Jupyter lab as follows:

```
jupyter lab
```

This should open your browser where you can open the exercises, e.g. `Locking.ipynb` or `Dynamics.ipynb`.

## Available exercises on binder

* Locking: [![Binder](images/badge_logo.svg)](https://mybinder.org/v2/gh/TUM-LNM/FEMTOE/main?labpath=Locking.ipynb)
* Dynamics: [![Binder](images/badge_logo.svg)](https://mybinder.org/v2/gh/TUM-LNM/FEMTOE/main?labpath=Dynamics.ipynb)
