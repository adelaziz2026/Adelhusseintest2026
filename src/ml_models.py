"""Classical machine learning models and a comparison helper."""
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier


def get_models(random_state: int = 42) -> dict:
    """Return a name -> estimator mapping of classical ML classifiers."""
    return {
        "logreg": make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)),
        "knn": make_pipeline(StandardScaler(), KNeighborsClassifier()),
        "svm": make_pipeline(StandardScaler(), SVC(random_state=random_state)),
        "decision_tree": DecisionTreeClassifier(random_state=random_state),
        "random_forest": RandomForestClassifier(n_estimators=200, random_state=random_state),
        "gradient_boosting": GradientBoostingClassifier(random_state=random_state),
    }


def compare_models(models: dict, X, y, cv: int = 5) -> pd.DataFrame:
    """Cross-validate each model and return mean/std accuracy, best first."""
    rows = []
    for name, model in models.items():
        scores = cross_val_score(model, X, y, cv=cv)
        rows.append({"model": name, "mean": scores.mean(), "std": scores.std()})
    return pd.DataFrame(rows).sort_values("mean", ascending=False).reset_index(drop=True)
