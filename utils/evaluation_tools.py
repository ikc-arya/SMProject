import pandas as pd
import numpy as np
from sklearn.metrics import (
    confusion_matrix, precision_recall_curve, roc_curve,
    average_precision_score
)
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def evaluate_multiclass(
    y_true_df,
    y_pred_df,
    characters
):
    """
    Evaluate multi-character predictions.
    
    Features:
    1. One figure: all characters' confusion matrices as subplots
    2. Overlay PR curve for all characters
    3. Overlay ROC curve for all characters with dashed diagonal
    4. Returns per-character MAP and overall MAP
    """
    metrics_dict = {}

    # For overlay plots
    fig_pr_comb = go.Figure()
    fig_roc_comb = go.Figure()

    # -----------------------------
    # Confusion matrix
    # -----------------------------
    n_chars = len(characters)
    fig_cm = make_subplots(
        rows=1, cols=n_chars,
        subplot_titles=characters,
        horizontal_spacing=0.05
    )

    for i, char in enumerate(characters, start=1):
        y_true = y_true_df[char].values
        y_pred = y_pred_df[f"{char}_present"].values

        cm = confusion_matrix(y_true, y_pred)
        cm_text = np.array([[f"{v}" for v in row] for row in cm])

        fig_cm.add_trace(
            go.Heatmap(
                z=cm,
                x=["Pred 0", "Pred 1"],
                y=["Actual 0", "Actual 1"],
                text=cm_text,
                texttemplate="%{text}",
                colorscale="Blues",
                showscale=False,
            ),
            row=1, col=i
        )

        # Rotate y-axis labels to vertical
        fig_cm.update_yaxes(tickangle=270, row=1, col=i)

        # ---- PR / ROC overlay prep ----
        precision, recall, _ = precision_recall_curve(y_true, y_pred)
        avg_prec = average_precision_score(y_true, y_pred)
        metrics_dict[char] = {"MAP": avg_prec}
        fig_pr_comb.add_trace(
            go.Scatter(x=recall, y=precision, mode="lines+markers", name=f"{char}")
        )

        fpr, tpr, _ = roc_curve(y_true, y_pred)
        fig_roc_comb.add_trace(
            go.Scatter(x=fpr, y=tpr, mode="lines+markers", name=f"{char}")
        )

    # -----------------------------
    # Display confusion matrices
    # -----------------------------
    fig_cm.update_layout(
        title_text="Confusion Matrices per Character",
        width=800,
        height=400,
        showlegend=False
    )
    fig_cm.show()

    # -----------------------------
    # Overlay Precision-Recall
    # -----------------------------
    fig_pr_comb.update_layout(
        title="Precision-Recall Overlay (all characters)",
        xaxis_title="Recall",
        yaxis_title="Precision",
        width=800,
        height=400
    )
    fig_pr_comb.show()

    # -----------------------------
    # Overlay ROC
    # -----------------------------
    fig_roc_comb.add_trace(
        go.Scatter(x=[0,1], y=[0,1], mode="lines", line=dict(dash="dash"), name = 'Random')
    )
    fig_roc_comb.update_layout(
        title="ROC Curve Overlay (all characters)",
        xaxis_title="FPR",
        yaxis_title="TPR",
        width=800,
        height=400
    )
    fig_roc_comb.show()

    # -----------------------------
    # MAP
    # -----------------------------
    overall_map = np.mean([m['MAP'] for m in metrics_dict.values()])
    print("Mean Average Precision (MAP) per character:")
    for char, m in metrics_dict.items():
        print(f"{char}: MAP={m['MAP']:.3f}")
    print(f"\nOverall MAP (all characters): {overall_map:.3f}")

    return metrics_dict, overall_map