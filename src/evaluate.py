"""Evaluation metrics.

For cancer detection, plain accuracy is misleading on imbalanced data, so we
also report ROC-AUC, sensitivity (recall on malignant) and specificity.

Metrics can additionally be sliced by BI-RADS breast density: dense breasts are
harder to read, so per-density numbers reveal whether the model is silently
failing on a subgroup that an aggregate score would hide. Because the per-stratum
positive counts at densities 1 and 4 are small, the non-dense (A/B = BI-RADS 1-2)
vs dense (C/D = BI-RADS 3-4) grouping is the primary subgroup result and the
four-way breakdown is supplementary.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import (accuracy_score, confusion_matrix,
                             roc_auc_score)

# Primary subgroup grouping: BI-RADS A/B (fatty/scattered) vs C/D (dense).
DEFAULT_DENSITY_GROUPS = {"non-dense (A/B)": (1, 2), "dense (C/D)": (3, 4)}


def _binary_metrics(y_true, y_pred, y_prob, pos):
    """Metrics + confusion matrix for one (sub)set of samples.

    Returns n / n_pos so the sample size behind each number is always visible.
    Rate metrics are NaN (not 0) when their denominator is empty, so an
    undefined value is never mistaken for a real score.
    """
    y_true_bin = (y_true == pos).astype(int)
    y_pred_bin = (y_pred == pos).astype(int)
    n = int(len(y_true))

    m = {"n": n, "n_pos": int(y_true_bin.sum())}
    m["accuracy"] = accuracy_score(y_true, y_pred) if n else float("nan")
    # AUC is undefined if only one class is present in this slice.
    try:
        m["auc"] = roc_auc_score(y_true_bin, y_prob)
    except ValueError:
        m["auc"] = float("nan")

    tn, fp, fn, tp = confusion_matrix(
        y_true_bin, y_pred_bin, labels=[0, 1]).ravel()
    m["tn"], m["fp"], m["fn"], m["tp"] = int(tn), int(fp), int(fn), int(tp)
    m["sensitivity"] = tp / (tp + fn) if (tp + fn) else float("nan")
    m["specificity"] = tn / (tn + fp) if (tn + fp) else float("nan")
    return m


@torch.no_grad()
def evaluate_model(model, loader, device, cfg, by_density=False,
                   density_groups=DEFAULT_DENSITY_GROUPS):
    """Run the model over `loader` and return metrics.

    With `by_density=True`, also returns:
      - `per_density`:       metrics keyed by each BI-RADS density value (1-4)
      - `per_density_group`: metrics keyed by the A/B vs C/D grouping
    Both read the `density` column from the loader's manifest (loader.dataset.df)
    and align it to predictions by row order, so they require a NON-shuffled,
    non-drop_last loader (which is how make_loader builds the test loader).
    """
    model.eval()
    all_labels, all_preds, all_probs = [], [], []
    pos = cfg.evaluate.positive_class

    for images, labels in loader:
        images = images.to(device)
        logits = model(images)
        probs = F.softmax(logits, dim=1)[:, pos].cpu().numpy()
        preds = logits.argmax(dim=1).cpu().numpy()
        all_probs.extend(probs)
        all_preds.extend(preds)
        all_labels.extend(labels.numpy())

    y_true = np.array(all_labels)
    y_pred = np.array(all_preds)
    y_prob = np.array(all_probs)

    metrics = _binary_metrics(y_true, y_pred, y_prob, pos)

    if by_density:
        df = getattr(loader.dataset, "df", None)
        if df is None or "density" not in df.columns:
            raise ValueError(
                "by_density=True requires the loader's manifest to carry a "
                "'density' column. Rebuild train/val/test.csv with density.")
        densities = df["density"].to_numpy()
        # Row-order alignment is only valid for a sequential (non-shuffled,
        # non-drop_last) loader; guard against silent misalignment.
        if len(densities) != len(y_true):
            raise ValueError(
                f"density rows ({len(densities)}) != predictions ({len(y_true)}); "
                "per-density metrics need a non-shuffled, drop_last=False loader.")

        metrics["per_density"] = {
            int(d): _binary_metrics(
                y_true[densities == d], y_pred[densities == d],
                y_prob[densities == d], pos)
            for d in sorted(np.unique(densities))
        }

        grouped = {}
        for name, vals in density_groups.items():
            mask = np.isin(densities, list(vals))
            if mask.any():
                grouped[name] = _binary_metrics(
                    y_true[mask], y_pred[mask], y_prob[mask], pos)
        metrics["per_density_group"] = grouped

    return metrics


def _print_table(title, rows):
    """rows: list of (label, metrics_dict)."""
    print(title)
    print(f"  {'stratum':<18}{'N':>5}{'pos':>5}{'AUC':>8}{'sens':>8}"
          f"{'spec':>8}    {'TN':>4}{'FP':>4}{'FN':>4}{'TP':>4}")
    for label, m in rows:
        print(f"  {label:<18}{m['n']:>5}{m['n_pos']:>5}"
              f"{m['auc']:>8.3f}{m['sensitivity']:>8.3f}{m['specificity']:>8.3f}"
              f"    {m['tn']:>4}{m['fp']:>4}{m['fn']:>4}{m['tp']:>4}")


def print_metrics(metrics):
    """Print aggregate, primary grouped, and supplementary four-way metrics.

    `pos`=positive count (malignant). CM columns are TN FP FN TP with malignant
    as positive. A/B = BI-RADS 1-2 (non-dense), C/D = BI-RADS 3-4 (dense).
    """
    _print_table("AGGREGATE (all densities)", [("all", metrics)])

    if "per_density_group" in metrics:
        print()
        _print_table(
            "PRIMARY -- non-dense (A/B) vs dense (C/D)",
            list(metrics["per_density_group"].items()))

    if "per_density" in metrics:
        print()
        _print_table(
            "SUPPLEMENTARY -- per BI-RADS density (small N at 1 & 4; "
            "interpret with care)",
            [(f"density {d}", m) for d, m in metrics["per_density"].items()])
