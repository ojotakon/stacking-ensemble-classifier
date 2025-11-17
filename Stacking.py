# Stacking.py


import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.datasets import load_breast_cancer

# ===========================
# Load dataset
# ===========================
data = load_breast_cancer()
X = data.data
y = data.target

# ===========================
# Base models
# ===========================
base_models = [
    ("tree", DecisionTreeClassifier(max_depth=4)),
    ("svm", SVC(kernel="rbf", probability=True))
]

meta_model = LogisticRegression()

# ===========================
# Manual Stacking Implementation
# ===========================
kf = KFold(n_splits=5, shuffle=True, random_state=42)

oof_preds = np.zeros((X.shape[0], len(base_models)))

for fold_idx, (train_idx, val_idx) in enumerate(kf.split(X)):
    print(f"Fold {fold_idx+1}")

    X_train, X_val = X[train_idx], X[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]

    fold_preds = []

    for i, (name, model) in enumerate(base_models):
        model.fit(X_train, y_train)
        val_pred = model.predict_proba(X_val)[:, 1]
        oof_preds[val_idx, i] = val_pred

# ===========================
# Train meta-learner
# ===========================
meta_model.fit(oof_preds, y)

# Evaluate meta-model
meta_pred_labels = meta_model.predict(oof_preds)
stacking_acc = accuracy_score(y, meta_pred_labels)

# ===========================
# Compare with Voting Ensemble
# ===========================
voting = VotingClassifier(estimators=base_models, voting="soft")
voting.fit(X, y)
voting_preds = voting.predict(X)
voting_acc = accuracy_score(y, voting_preds)

# ===========================
# Compare with base models
# ===========================
base_accs = {}
for name, model in base_models:
    model.fit(X, y)
    preds = model.predict(X)
    base_accs[name] = accuracy_score(y, preds)

# ===========================
# Summary
# ===========================
print("\n=== RESULTS ===")
print(f"Stacking Ensemble Accuracy: {stacking_acc:.4f}")
print(f"Voting Ensemble Accuracy:  {voting_acc:.4f}\n")

print("Base Models:")
for k, v in base_accs.items():
    print(f" - {k}: {v:.4f}")
