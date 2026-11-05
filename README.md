# Data Science Starter

An end-to-end data science project: load data, explore, train, and evaluate.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
jupyter notebook notebooks/01_exploration_and_model.ipynb
```

## Layout

- `notebooks/` – Jupyter notebooks (`01` exploration and a first model, `02` ML vs. deep learning comparison)
- `src/` – reusable code: `data.py` (loading), `ml_models.py` (classical ML algorithms), `deep_learning.py` (PyTorch neural network)
- `data/` – local data; CSVs are git-ignored

The example uses scikit-learn's built-in breast cancer dataset, so no downloads are needed. Swap `load_data()` in `src/data.py` for your own data.
