"""
metrics_utils.py
================
Shared evaluation metrics for the revised CAMPL pipeline.

Provides:
  - spectrum-level UAR / UF1 / accuracy / confusion matrix
  - per-class precision / recall / F1 / specificity
  - macro AUC (one-vs-rest) when class probabilities are available
  - bootstrap 95% confidence intervals for UAR / UF1 / accuracy
  - patient-level aggregation (mean predicted probability per patient) with
    patient-level UAR / UF1 / accuracy / confusion matrix
"""

import numpy as np
from sklearn.metrics import (
    recall_score, f1_score, precision_score, confusion_matrix, roc_auc_score,
)


def per_class_specificity(cm):
    """Specificity (true negative rate) per class from a confusion matrix."""
    cm = np.asarray(cm, dtype=np.float64)
    total = cm.sum()
    tp = np.diag(cm)
    fn = cm.sum(axis=1) - tp
    fp = cm.sum(axis=0) - tp
    tn = total - tp - fn - fp
    with np.errstate(divide='ignore', invalid='ignore'):
        spec = np.where((tn + fp) > 0, tn / (tn + fp), np.nan)
    return spec


def bootstrap_ci(y_true, y_pred, n_bootstrap=1000, seed=42, confidence=0.95):
    """
    Bootstrap confidence intervals for UAR / UF1 / accuracy.
    Resamples spectrum indices with replacement.
    Returns dict: metric -> (point_estimate, ci_low, ci_high).
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    rng = np.random.default_rng(seed)
    n = len(y_true)

    def _stats(idx):
        return (
            recall_score(y_true[idx], y_pred[idx], average='macro', zero_division=0),
            f1_score(y_true[idx], y_pred[idx], average='macro', zero_division=0),
            float(np.mean(y_true[idx] == y_pred[idx])),
        )

    point = _stats(np.arange(n))
    boot = np.array([_stats(rng.integers(0, n, n)) for _ in range(n_bootstrap)])
    alpha = 1.0 - confidence
    lo = np.nanpercentile(boot, 100 * alpha / 2, axis=0)
    hi = np.nanpercentile(boot, 100 * (1 - alpha / 2), axis=0)

    names = ['uar', 'uf1', 'accuracy']
    return {name: (float(p), float(l), float(h))
            for name, p, l, h in zip(names, point, lo, hi)}


def macro_auc(y_true, probs, num_classes):
    """Macro one-vs-rest AUC; NaN if not computable (e.g. missing classes)."""
    if probs is None:
        return float('nan')
    y_true = np.asarray(y_true)
    probs = np.asarray(probs)
    try:
        if num_classes == 2:
            return float(roc_auc_score(y_true, probs[:, 1]))
        present = np.unique(y_true)
        if len(present) < 2:
            return float('nan')
        return float(roc_auc_score(
            y_true, probs[:, :num_classes], multi_class='ovr', average='macro',
            labels=np.arange(num_classes)))
    except Exception:
        return float('nan')


def aggregate_by_patient(y_true, probs, groups):
    """
    Aggregate spectrum-level predicted probabilities by patient (mean
    probability). Returns (patient_ids, patient_y_true, patient_probs,
    patient_y_pred). Patient ground truth = majority/first label (validated
    upstream to be unique per patient when possible).
    """
    y_true = np.asarray(y_true)
    probs = np.asarray(probs)
    groups = np.asarray(groups)

    pids, y_out, p_out = [], [], []
    for pid in np.unique(groups):
        mask = groups == pid
        labels, counts = np.unique(y_true[mask], return_counts=True)
        pids.append(pid)
        y_out.append(int(labels[np.argmax(counts)]))
        p_out.append(probs[mask].mean(axis=0))
    p_out = np.stack(p_out, axis=0)
    return np.array(pids), np.array(y_out), p_out, p_out.argmax(axis=1)


def compute_extended_metrics(y_true, y_pred, probs=None, groups=None,
                             num_classes=None, n_bootstrap=1000, seed=42):
    """
    Compute the full metric suite. Returns a nested dict:
      {
        'spectrum': {...},         # spectrum-level metrics
        'patient': {...} or None,  # patient-level metrics (if groups given)
      }
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if num_classes is None:
        num_classes = int(max(y_true.max(), y_pred.max())) + 1

    labels = np.arange(num_classes)
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    uar = recall_score(y_true, y_pred, average='macro', zero_division=0)
    uf1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
    acc = float(np.mean(y_true == y_pred))
    prec_pc = precision_score(y_true, y_pred, average=None, labels=labels, zero_division=0)
    rec_pc = recall_score(y_true, y_pred, average=None, labels=labels, zero_division=0)
    f1_pc = f1_score(y_true, y_pred, average=None, labels=labels, zero_division=0)
    spec_pc = per_class_specificity(cm)
    auc = macro_auc(y_true, probs, num_classes)
    ci = bootstrap_ci(y_true, y_pred, n_bootstrap=n_bootstrap, seed=seed)

    spectrum = {
        'uar': float(uar), 'uf1': float(uf1), 'accuracy': acc,
        'confusion_matrix': cm,
        'per_class_precision': prec_pc,
        'per_class_recall': rec_pc,
        'per_class_f1': f1_pc,
        'per_class_specificity': spec_pc,
        'macro_auc': auc,
        'bootstrap_ci': ci,
    }

    # Patient-level metrics are reported whenever patient grouping is
    # available — including the degenerate case of exactly one spectrum per
    # patient (aggregation is then the identity, and the patient-level
    # numbers still document the cohort size).
    patient = None
    if groups is not None and probs is not None:
        pids, py_true, pprobs, py_pred = aggregate_by_patient(y_true, probs, groups)
        pcm = confusion_matrix(py_true, py_pred, labels=labels)
        patient = {
            'n_patients': len(pids),
            'uar': float(recall_score(py_true, py_pred, average='macro', zero_division=0)),
            'uf1': float(f1_score(py_true, py_pred, average='macro', zero_division=0)),
            'accuracy': float(np.mean(py_true == py_pred)),
            'confusion_matrix': pcm,
            'macro_auc': macro_auc(py_true, pprobs, num_classes),
        }

    return {'spectrum': spectrum, 'patient': patient}


def format_metrics_report(metrics, title='Evaluation', num_classes=None):
    """Render the metric dict from compute_extended_metrics as a text report."""
    lines = []
    lines.append(f"===== {title} =====")
    spec = metrics['spectrum']
    cm = spec['confusion_matrix']
    if num_classes is None:
        num_classes = cm.shape[0]

    lines.append("-- Spectrum-level --")
    lines.append(f"UAR      : {spec['uar']:.4f}")
    lines.append(f"UF1      : {spec['uf1']:.4f}")
    lines.append(f"Accuracy : {spec['accuracy']:.4f}")
    if not np.isnan(spec['macro_auc']):
        lines.append(f"Macro AUC: {spec['macro_auc']:.4f}")
    lines.append("Bootstrap 95% CI:")
    for name in ['uar', 'uf1', 'accuracy']:
        p, lo, hi = spec['bootstrap_ci'][name]
        lines.append(f"  {name:9s}: {p:.4f} [{lo:.4f}, {hi:.4f}]")

    lines.append("Per-class metrics:")
    lines.append(f"  {'class':>6s} {'prec':>8s} {'recall':>8s} {'f1':>8s} {'spec':>8s}")
    for c in range(num_classes):
        lines.append(
            f"  {c:>6d} {spec['per_class_precision'][c]:>8.4f} "
            f"{spec['per_class_recall'][c]:>8.4f} {spec['per_class_f1'][c]:>8.4f} "
            f"{spec['per_class_specificity'][c]:>8.4f}")
    lines.append(f"Confusion matrix:\n{cm}")

    # if metrics.get('patient') is not None:
    #     pat = metrics['patient']
    #     lines.append("-- Patient-level (mean probability aggregation) --")
    #     lines.append(f"Patients : {pat['n_patients']}")
    #     lines.append(f"UAR      : {pat['uar']:.4f}")
    #     lines.append(f"UF1      : {pat['uf1']:.4f}")
    #     lines.append(f"Accuracy : {pat['accuracy']:.4f}")
    #     if not np.isnan(pat['macro_auc']):
    #         lines.append(f"Macro AUC: {pat['macro_auc']:.4f}")
    #     lines.append(f"Confusion matrix:\n{pat['confusion_matrix']}")

    return "\n".join(lines)
