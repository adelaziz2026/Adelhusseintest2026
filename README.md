# Data Science Starter

![Python](https://img.shields.io/badge/python-3.13-blue?logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-orange?logo=scikitlearn&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-deep%20learning-ee4c2c?logo=pytorch&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-notebooks-f37626?logo=jupyter&logoColor=white)

An end-to-end data science project: load data, explore, train, and evaluate.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
jupyter notebook notebooks/01_exploration_and_model.ipynb
```

`requirements.txt` has only what the example notebooks need. For the wider data science stack (SciPy, Polars, Plotly, XGBoost/LightGBM/CatBoost, statsmodels, torchvision, Optuna, SHAP and more), run `pip install -r requirements-extra.txt`.

## Layout

- `notebooks/` – Jupyter notebooks (`01` exploration and a first model, `02` ML vs. deep learning comparison)
- `src/` – reusable code: `data.py` (loading), `ml_models.py` (classical ML algorithms), `deep_learning.py` (PyTorch neural network)
- `data/` – local data; CSVs are git-ignored

The example uses scikit-learn's built-in breast cancer dataset, so no downloads are needed. Swap `load_data()` in `src/data.py` for your own data.
