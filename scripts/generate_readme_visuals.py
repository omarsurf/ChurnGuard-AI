#!/usr/bin/env python3
"""Generate visualizations for README.md.

Run from project root:
    python scripts/generate_readme_visuals.py
"""

import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import ConfusionMatrixDisplay, RocCurveDisplay

# Paths
PROJECT_ROOT = Path(__file__).parent.parent
ASSETS_DIR = PROJECT_ROOT / "assets"
MODELS_DIR = PROJECT_ROOT / "models"
DATA_DIR = PROJECT_ROOT / "data" / "processed"

ASSETS_DIR.mkdir(exist_ok=True)

# Style settings
plt.style.use("seaborn-v0_8-whitegrid")
COLORS = {"primary": "#2563eb", "secondary": "#7c3aed", "accent": "#059669", "warning": "#d97706"}


def load_artifacts():
    """Load model, preprocessor, and test data."""
    model = joblib.load(MODELS_DIR / "best_model.joblib")

    X_test = np.load(DATA_DIR / "X_test_processed.npy")
    y_test = np.load(DATA_DIR / "y_test.npy")

    return model, X_test, y_test


def plot_roc_curve(model, X_test, y_test):
    """Generate ROC curve plot."""
    fig, ax = plt.subplots(figsize=(8, 6))

    RocCurveDisplay.from_estimator(
        model, X_test, y_test,
        ax=ax,
        color=COLORS["primary"],
        lw=2,
        name="Logistic Regression"
    )

    # Add diagonal reference line
    ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5, label="Random Classifier")

    # Add ROC-AUC annotation
    from sklearn.metrics import roc_auc_score
    y_proba = model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, y_proba)
    ax.annotate(
        f"ROC-AUC = {auc:.3f}",
        xy=(0.6, 0.3),
        fontsize=14,
        fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor=COLORS["primary"])
    )

    ax.set_title("ROC Curve - Production Model", fontsize=14, fontweight="bold")
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.legend(loc="lower right")

    plt.tight_layout()
    plt.savefig(ASSETS_DIR / "roc_curve.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved: assets/roc_curve.png")


def plot_confusion_matrix(model, X_test, y_test, threshold=0.25):
    """Generate confusion matrix at optimal threshold."""
    y_proba = model.predict_proba(X_test)[:, 1]
    y_pred = (y_proba >= threshold).astype(int)

    fig, ax = plt.subplots(figsize=(7, 6))

    ConfusionMatrixDisplay.from_predictions(
        y_test, y_pred,
        ax=ax,
        cmap="Blues",
        display_labels=["No Churn", "Churn"],
        colorbar=False
    )

    ax.set_title(f"Confusion Matrix (Threshold = {threshold})", fontsize=14, fontweight="bold")

    # Add annotation
    tn, fp, fn, tp = (
        ((y_test == 0) & (y_pred == 0)).sum(),
        ((y_test == 0) & (y_pred == 1)).sum(),
        ((y_test == 1) & (y_pred == 0)).sum(),
        ((y_test == 1) & (y_pred == 1)).sum()
    )

    plt.figtext(
        0.5, -0.05,
        f"TP={tp} | FP={fp} | FN={fn} | TN={tn}  |  Recall={tp/(tp+fn):.1%}  |  Precision={tp/(tp+fp):.1%}",
        ha="center", fontsize=10, style="italic"
    )

    plt.tight_layout()
    plt.savefig(ASSETS_DIR / "confusion_matrix.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved: assets/confusion_matrix.png")


def plot_threshold_analysis():
    """Generate threshold analysis plot from CSV."""
    threshold_file = MODELS_DIR / "threshold_analysis_val.csv"

    if not threshold_file.exists():
        print(f"Warning: {threshold_file} not found, skipping threshold plot")
        return

    df = pd.read_csv(threshold_file)

    # Normalize column names to lowercase
    df.columns = df.columns.str.lower()

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Plot 1: Precision-Recall Trade-off
    ax1 = axes[0]
    ax1.plot(df["threshold"], df["precision"], label="Precision", color=COLORS["primary"], lw=2, marker="o", markersize=4)
    ax1.plot(df["threshold"], df["recall"], label="Recall", color=COLORS["secondary"], lw=2, marker="s", markersize=4)
    ax1.plot(df["threshold"], df["f1_score"], label="F1 Score", color=COLORS["accent"], lw=2, marker="^", markersize=4)

    # Mark optimal threshold
    ax1.axvline(x=0.25, color=COLORS["warning"], linestyle="--", lw=2, label="Optimal (0.25)")
    ax1.axhline(y=0.70, color="gray", linestyle=":", alpha=0.5)
    ax1.axhline(y=0.50, color="gray", linestyle=":", alpha=0.5)

    ax1.set_xlabel("Threshold", fontsize=12)
    ax1.set_ylabel("Score", fontsize=12)
    ax1.set_title("Precision-Recall Trade-off", fontsize=14, fontweight="bold")
    ax1.legend(loc="center left")
    ax1.set_xlim(0.15, 0.90)
    ax1.set_ylim(0, 1)
    ax1.grid(True, alpha=0.3)

    # Plot 2: Net Value
    ax2 = axes[1]
    ax2.bar(df["threshold"], df["net_value"], color=COLORS["primary"], alpha=0.7, width=0.04)
    ax2.axvline(x=0.25, color=COLORS["warning"], linestyle="--", lw=2, label="Optimal (0.25)")

    # Highlight optimal
    optimal_idx = df[df["threshold"] == 0.25].index
    if len(optimal_idx) > 0:
        optimal_value = df.loc[optimal_idx[0], "net_value"]
        ax2.bar([0.25], [optimal_value], color=COLORS["warning"], alpha=0.9, width=0.04)
        ax2.annotate(
            f"${optimal_value:,.0f}",
            xy=(0.25, optimal_value),
            xytext=(0.35, optimal_value * 0.9),
            fontsize=12,
            fontweight="bold",
            arrowprops=dict(arrowstyle="->", color=COLORS["warning"])
        )

    ax2.set_xlabel("Threshold", fontsize=12)
    ax2.set_ylabel("Net Value ($)", fontsize=12)
    ax2.set_title("Business Value by Threshold", fontsize=14, fontweight="bold")
    ax2.legend()
    ax2.set_xlim(0.15, 0.90)
    ax2.grid(True, alpha=0.3, axis="y")

    # Format y-axis as currency
    ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f"${x/1000:.0f}K"))

    plt.tight_layout()
    plt.savefig(ASSETS_DIR / "threshold_analysis.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved: assets/threshold_analysis.png")


def plot_feature_importance(model):
    """Generate feature importance plot."""
    # Load registry for feature names
    registry_file = MODELS_DIR / "registry.json"

    if not registry_file.exists():
        print("Warning: registry.json not found, skipping feature importance plot")
        return

    with open(registry_file) as f:
        registry = json.load(f)

    # Get feature importance from registry
    if "models" in registry and len(registry["models"]) > 0:
        prod_model = registry["models"][0]
        importance = prod_model.get("feature_importance", {})
        features = prod_model.get("input_features", [])
    else:
        print("Warning: No model in registry, skipping feature importance plot")
        return

    if not importance:
        print("Warning: No feature importance in registry, skipping plot")
        return

    # Create DataFrame and sort
    df = pd.DataFrame({
        "feature": list(importance.keys()),
        "importance": list(importance.values())
    })
    df = df.sort_values("importance", ascending=True).tail(15)  # Top 15

    # Clean feature names
    df["feature"] = df["feature"].str.replace("num__", "").str.replace("cat__", "")

    fig, ax = plt.subplots(figsize=(10, 8))

    colors = [COLORS["primary"] if v > 0.5 else COLORS["secondary"] for v in df["importance"]]
    bars = ax.barh(df["feature"], df["importance"], color=colors, alpha=0.8)

    ax.set_xlabel("Absolute Coefficient (L1 Regularization)", fontsize=12)
    ax.set_title("Top 15 Feature Importances", fontsize=14, fontweight="bold")
    ax.grid(True, alpha=0.3, axis="x")

    # Add value labels
    for bar, val in zip(bars, df["importance"]):
        ax.text(val + 0.02, bar.get_y() + bar.get_height()/2,
                f"{val:.2f}", va="center", fontsize=9)

    plt.tight_layout()
    plt.savefig(ASSETS_DIR / "feature_importance.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved: assets/feature_importance.png")


def plot_pipeline_diagram():
    """Generate simple pipeline flow diagram."""
    fig, ax = plt.subplots(figsize=(14, 4))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 4)
    ax.axis("off")

    # Stage boxes
    stages = [
        {"name": "PREPARE", "x": 1, "color": "#3b82f6", "desc": "Data Validation\nFeature Engineering\n14 Features"},
        {"name": "TRAIN", "x": 4.5, "color": "#8b5cf6", "desc": "Multi-Candidate\nModel Selection\nRegistry"},
        {"name": "EVALUATE", "x": 8, "color": "#10b981", "desc": "Threshold Search\nQuality Gates\nNet Value"},
        {"name": "PREDICT", "x": 11.5, "color": "#f59e0b", "desc": "Batch Inference\nDrift Detection\nStrict Mode"},
    ]

    for stage in stages:
        # Box
        rect = plt.Rectangle((stage["x"] - 1.2, 0.8), 2.4, 2.4,
                             facecolor=stage["color"], alpha=0.15,
                             edgecolor=stage["color"], linewidth=2,
                             joinstyle="round")
        ax.add_patch(rect)

        # Title
        ax.text(stage["x"], 2.8, stage["name"], ha="center", va="center",
               fontsize=12, fontweight="bold", color=stage["color"])

        # Description
        ax.text(stage["x"], 1.8, stage["desc"], ha="center", va="center",
               fontsize=9, color="#374151", linespacing=1.4)

    # Arrows
    for i in range(len(stages) - 1):
        ax.annotate("",
                   xy=(stages[i+1]["x"] - 1.4, 2),
                   xytext=(stages[i]["x"] + 1.4, 2),
                   arrowprops=dict(arrowstyle="->", color="#6b7280", lw=2))

    # Config box at top
    ax.text(7, 3.7, "config/default.yaml", ha="center", va="center",
           fontsize=10, style="italic", color="#6b7280",
           bbox=dict(boxstyle="round,pad=0.3", facecolor="#f3f4f6", edgecolor="#d1d5db"))

    plt.tight_layout()
    plt.savefig(ASSETS_DIR / "pipeline_diagram.png", dpi=150, bbox_inches="tight",
                facecolor="white", edgecolor="none")
    plt.close()
    print("Saved: assets/pipeline_diagram.png")


def main():
    print("Generating README visuals...\n")

    # Load artifacts
    try:
        model, X_test, y_test = load_artifacts()
        print("Loaded model and test data\n")
    except Exception as e:
        print(f"Error loading artifacts: {e}")
        print("Generating only static plots...\n")
        model, X_test, y_test = None, None, None

    # Generate plots
    if model is not None:
        plot_roc_curve(model, X_test, y_test)
        plot_confusion_matrix(model, X_test, y_test, threshold=0.25)
        plot_feature_importance(model)

    plot_threshold_analysis()
    plot_pipeline_diagram()

    print("\nDone! Visuals saved to assets/")


if __name__ == "__main__":
    main()
