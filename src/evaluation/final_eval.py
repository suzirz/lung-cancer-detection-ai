"""
Module: Comprehensive Final Evaluation & Cross-Validation
File: src/evaluation/final_eval.py

Prinsip Codebase Design:
- Modul Deep: Menyembunyikan kalkulasi per-class ROC-AUC (One-vs-Rest), multi-fold cross-validation
  stabilitas, kurva ROC visual, dan laporan audit komparasi final.
- Interface sederhana: run_final_evaluation(model, X_test, y_test, class_names).
"""

import os
import json
from typing import Dict, List, Tuple
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, roc_auc_score
from sklearn.preprocessing import label_binarize
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier

from src.evaluation.metrics import calculate_metrics


def compute_roc_auc_multiclass(
    y_true: List[int],
    y_probs: np.ndarray,
    class_names: List[str],
    output_path: str = "reports/roc_auc_curve.png"
) -> Dict[str, float]:
    """
    Menghitung AUC-ROC multi-kelas (One-vs-Rest) dan menyimpan plot kurva ROC.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    num_classes = len(class_names)
    y_bin = label_binarize(y_true, classes=list(range(num_classes)))

    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    fig, ax = plt.subplots(figsize=(7, 6))
    colors = ["#2b8a3e", "#e03131", "#1971c2"]

    for i in range(num_classes):
        fpr[i], tpr[i], _ = roc_curve(y_bin[:, i], y_probs[:, i])
        roc_auc[class_names[i]] = float(auc(fpr[i], tpr[i]))
        ax.plot(
            fpr[i], tpr[i], color=colors[i % len(colors)], lw=2,
            label=f"ROC {class_names[i]} (AUC = {roc_auc[class_names[i]]:.3f})"
        )

    # Macro AUC
    macro_auc = float(roc_auc_score(y_bin, y_probs, average="macro", multi_class="ovr"))
    roc_auc["macro_auc"] = macro_auc

    ax.plot([0, 1], [0, 1], "k--", lw=1.5, label="Random Guess (AUC = 0.500)")
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate (1 - Specificity)", fontsize=11)
    ax.set_ylabel("True Positive Rate (Sensitivity / Recall)", fontsize=11)
    ax.set_title("Multi-Class ROC-AUC Curves — Lung Cancer Diagnosis", fontsize=13, fontweight="bold", pad=12)
    ax.legend(loc="lower right", fontsize=10)
    ax.grid(alpha=0.4)

    fig.tight_layout()
    plt.savefig(output_path, dpi=100)
    plt.close(fig)

    return roc_auc


def run_cross_validation_audit(
    X: np.ndarray,
    y: np.ndarray,
    n_splits: int = 5,
    random_state: int = 42
) -> Dict[str, float]:
    """
    Runs Stratified K-Fold Cross-Validation evaluating Macro Recall, Macro F1, and Accuracy.
    """
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    clf = RandomForestClassifier(n_estimators=100, max_depth=10, class_weight="balanced", random_state=random_state, n_jobs=1)

    scores_recall = cross_val_score(clf, X, y, cv=cv, scoring="recall_macro", n_jobs=1)
    scores_f1 = cross_val_score(clf, X, y, cv=cv, scoring="f1_macro", n_jobs=1)
    scores_acc = cross_val_score(clf, X, y, cv=cv, scoring="accuracy", n_jobs=1)

    return {
        "cv_folds": n_splits,
        "mean_recall_macro": float(np.mean(scores_recall)),
        "std_recall_macro": float(np.std(scores_recall)),
        "mean_f1_macro": float(np.mean(scores_f1)),
        "std_f1_macro": float(np.std(scores_f1)),
        "mean_accuracy": float(np.mean(scores_acc)),
        "std_accuracy": float(np.std(scores_acc)),
    }
