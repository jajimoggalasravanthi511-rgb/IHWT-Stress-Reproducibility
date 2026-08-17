from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np
from imblearn.ensemble import RUSBoostClassifier
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.model_selection import GroupKFold
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier


@dataclass
class WeightedVotingEnsemble:
    models: Dict[str, object]
    weights: Dict[str, float]
    classes_: np.ndarray

    def predict(self, X: np.ndarray) -> np.ndarray:
        score = np.zeros((len(X), len(self.classes_)), dtype=float)
        class_to_idx = {c: i for i, c in enumerate(self.classes_)}
        total = sum(self.weights.values()) + 1e-12
        for name, model in self.models.items():
            pred = model.predict(X)
            w = self.weights[name] / total
            for i, p in enumerate(pred):
                score[i, class_to_idx[p]] += w
        return self.classes_[np.argmax(score, axis=1)]


def build_models(cfg: dict, seed: int, n_classes: int) -> Dict[str, object]:
    rf = cfg["random_forest"]
    xg = cfg["xgboost"]
    ru = cfg["rusboost"]
    return {
        "RF": RandomForestClassifier(
            n_estimators=int(rf["n_estimators"]),
            max_depth=rf["max_depth"],
            min_samples_leaf=int(rf["min_samples_leaf"]),
            max_features=rf["max_features"],
            class_weight=rf["class_weight"],
            n_jobs=-1,
            random_state=seed,
        ),
        "XGBoost": XGBClassifier(
            n_estimators=int(xg["n_estimators"]),
            max_depth=int(xg["max_depth"]),
            learning_rate=float(xg["learning_rate"]),
            subsample=float(xg["subsample"]),
            colsample_bytree=float(xg["colsample_bytree"]),
            reg_lambda=float(xg["reg_lambda"]),
            reg_alpha=float(xg["reg_alpha"]),
            objective="multi:softprob" if n_classes > 2 else "binary:logistic",
            eval_metric="mlogloss" if n_classes > 2 else "logloss",
            n_jobs=-1,
            random_state=seed,
        ),
        "RUSBoost": RUSBoostClassifier(
            estimator=DecisionTreeClassifier(max_depth=int(ru["weak_learner_depth"]), random_state=seed),
            n_estimators=int(ru["n_estimators"]),
            learning_rate=float(ru["learning_rate"]),
            random_state=seed,
        ),
    }


def derive_inner_group_weights(X: np.ndarray, y: np.ndarray, groups: np.ndarray, model_cfg: dict, seed: int) -> dict[str, float]:
    names = ["RF", "XGBoost", "RUSBoost"]
    unique_groups = np.unique(groups)
    k = min(int(model_cfg["ensemble"]["inner_group_folds"]), len(unique_groups))
    if k < 2:
        return {n: 1.0 / len(names) for n in names}
    splitter = GroupKFold(n_splits=k)
    fold_scores = {n: [] for n in names}
    n_classes = len(np.unique(y))
    for tr, va in splitter.split(X, y, groups):
        models = build_models(model_cfg, seed, n_classes)
        for name, model in models.items():
            m = clone(model)
            m.fit(X[tr], y[tr])
            pred = m.predict(X[va])
            metric_name = str(model_cfg["ensemble"].get("weight_metric", "accuracy"))
            score = balanced_accuracy_score(y[va], pred) if metric_name == "balanced_accuracy" else accuracy_score(y[va], pred)
            fold_scores[name].append(score)
    raw = {n: max(float(np.mean(fold_scores[n])), float(model_cfg["ensemble"]["min_weight"])) for n in names}
    s = sum(raw.values())
    return {n: raw[n] / s for n in names}
