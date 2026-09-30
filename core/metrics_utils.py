
import numpy as np
from sklearn.metrics import (
    recall_score, f1_score, precision_score, confusion_matrix, roc_auc_score,
)


def per_class_specificity(cm):
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

    return "\n".join(lines)
