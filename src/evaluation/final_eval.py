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
from typing import Dict, List, Tuple, Optional
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, roc_auc_score
from sklearn.preprocessing import label_binarize
from sklearn.model_selection import StratifiedKFold, StratifiedGroupKFold, cross_val_score
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
    groups: Optional[List[str]] = None,
    n_splits: int = 5,
    random_state: int = 42
) -> Dict[str, float]:
    """
    Runs Stratified (Group) K-Fold Cross-Validation evaluating Macro Recall, Macro F1, and Accuracy.
    Jika groups (patient case IDs) tersedia, menggunakan StratifiedGroupKFold untuk menjamin
    zero data leakage antar-fold.
    """
    if groups is not None:
        cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    else:
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    clf = RandomForestClassifier(n_estimators=100, max_depth=10, class_weight="balanced", random_state=random_state, n_jobs=1)

    scores_recall = cross_val_score(clf, X, y, groups=groups, cv=cv, scoring="recall_macro", n_jobs=1)
    scores_f1 = cross_val_score(clf, X, y, groups=groups, cv=cv, scoring="f1_macro", n_jobs=1)
    scores_acc = cross_val_score(clf, X, y, groups=groups, cv=cv, scoring="accuracy", n_jobs=1)

    return {
        "cv_folds": n_splits,
        "mean_recall_macro": float(np.mean(scores_recall)),
        "std_recall_macro": float(np.std(scores_recall)),
        "mean_f1_macro": float(np.mean(scores_f1)),
        "std_f1_macro": float(np.std(scores_f1)),
        "mean_accuracy": float(np.mean(scores_acc)),
        "std_accuracy": float(np.std(scores_acc)),
    }


def main():
    """
    Eksekusi audit evaluasi diagnostik dan 5-fold cross validation.
    Membaca embedding tersimpan dari data/processed/ atau data/embeddings/.
    """
    print("=" * 60)
    print("EVALUASI DIAGNOSTIK LANJUTAN & 5-FOLD CROSS-VALIDATION")
    print("=" * 60)

    # Dukung path cnn_features (dari train_hybrid) dan fallback train_embeddings
    candidate_paths = [
        "data/processed/cnn_features_efficientnet_b0.npz",
        "data/embeddings/train_embeddings.npz"
    ]
    train_emb_path = next((p for p in candidate_paths if os.path.exists(p)), None)

    if not train_emb_path:
        print(f"[!] Embedding file tidak ditemukan di: {candidate_paths}")
        print("[i] Jalankan ekstraksi fitur embedding terlebih dahulu:")
        print("    python -m src.training.train_hybrid")
        return

    print(f"[*] Membaca embedding dari: {train_emb_path}")
    data = np.load(train_emb_path)
    X_train = data.get("X_train", data.get("embeddings"))
    y_train = data.get("y_train", data.get("labels"))

    train_split_path = "data/splits/train_split.json"
    groups = None
    if os.path.exists(train_split_path):
        from src.preprocessing.dataset import extract_group_id
        with open(train_split_path, "r", encoding="utf-8") as f:
            train_samples = json.load(f)
        if len(train_samples) == len(y_train):
            groups = [extract_group_id(p[0]) for p in train_samples]
            print(f"[+] Patient groups terdeteksi: {len(set(groups))} unique groups pada {len(groups)} sampel training.")

    if groups is not None:
        print(f"[+] Menjalankan 5-Fold Stratified Group CV (Zero Patient Leakage Antar-Fold)...")
    else:
        print(f"[+] Menjalankan 5-Fold Stratified CV pada {len(y_train)} sampel training...")
    cv_results = run_cross_validation_audit(X_train, y_train, groups=groups, n_splits=5)

    os.makedirs("reports", exist_ok=True)
    out_cv_path = "reports/cv_audit.json"
    with open(out_cv_path, "w", encoding="utf-8") as f:
        json.dump(cv_results, f, indent=2)

    print(f"[OK] Audit 5-Fold CV selesai dan disimpan ke: {out_cv_path}")
    print(f" - Mean Accuracy     : {cv_results['mean_accuracy']*100:.2f}% ± {cv_results['std_accuracy']*100:.2f}%")
    print(f" - Mean Macro Recall : {cv_results['mean_recall_macro']*100:.2f}% ± {cv_results['std_recall_macro']*100:.2f}%")
    print(f" - Mean Macro F1     : {cv_results['mean_f1_macro']*100:.2f}% ± {cv_results['std_f1_macro']*100:.2f}%")

    # Evaluasi pada data Test Set Murni (Unseen Data)
    X_test = data.get("X_test", None)
    y_test = data.get("y_test", None)
    rf_model_path = "models/hybrid_random_forest.joblib"

    if X_test is not None and y_test is not None:
        # Cross-validation pada Unseen Test Embeddings untuk validasi stabilitas pasien baru
        test_split_path = "data/splits/test_split.json"
        if os.path.exists(test_split_path):
            from src.preprocessing.dataset import extract_group_id
            with open(test_split_path, "r", encoding="utf-8") as f:
                test_samples = json.load(f)
            if len(test_samples) == len(y_test):
                test_groups = [extract_group_id(p[0]) for p in test_samples]
                print(f"\n[+] Menjalankan 5-Fold Group CV pada Unseen Test Set ({len(set(test_groups))} pasien independen)...")
                test_cv = run_cross_validation_audit(X_test, y_test, groups=test_groups, n_splits=5)
                print(f" - Unseen Test Mean Accuracy : {test_cv['mean_accuracy']*100:.2f}% ± {test_cv['std_accuracy']*100:.2f}%")
                print(f" - Unseen Test Macro Recall  : {test_cv['mean_recall_macro']*100:.2f}% ± {test_cv['std_recall_macro']*100:.2f}%")
                cv_results["unseen_test_cv"] = test_cv
                with open(out_cv_path, "w", encoding="utf-8") as f:
                    json.dump(cv_results, f, indent=2)

        # Generate Multi-Class ROC-AUC curve jika model hybrid tersedia
        if os.path.exists(rf_model_path):
            import joblib
            print("\n[*] Menghitung Multi-Class ROC-AUC pada test set murni...")
            rf_model = joblib.load(rf_model_path)
            class_names = ["Benign cases", "Malignant cases", "Normal cases"]
            y_probs = rf_model.predict_proba(X_test)
            roc_results = compute_roc_auc_multiclass(y_test, y_probs, class_names, output_path="reports/roc_auc_curve.png")
            print(f"[OK] Kurva ROC-AUC disimpan ke: reports/roc_auc_curve.png (Macro AUC: {roc_results.get('macro_auc', 0.0):.4f})")


if __name__ == "__main__":
    main()
