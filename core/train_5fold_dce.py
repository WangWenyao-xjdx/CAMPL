import os
import copy
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import recall_score

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau

from LoadData import load_data
from config import Config
from metrics_utils import compute_extended_metrics, format_metrics_report


def get_model_class(model_name):
    if model_name == 'CNN4':
        from models.CNN4 import CNN4
        return CNN4
    elif model_name == 'GoogleNet':
        from models.GoogleNet import GoogleNet1D
        return GoogleNet1D
    elif model_name == 'ResNet9':
        from models.ResNet9 import ResNet9_1D
        return ResNet9_1D
    elif model_name == 'ResNet18':
        from models.ResNet18 import ResNet18_1D
        return ResNet18_1D
    elif model_name == 'VGG16':
        from models.VGG16 import VGG16_1D
        return VGG16_1D
    else:
        raise ValueError(f"error model name: {model_name}")


class TensorDataset(Dataset):
    def __init__(self, X, y, group_codes=None):
        self.X = X
        self.y = y
        if group_codes is None:
            group_codes = torch.zeros(len(y), dtype=torch.long)
        self.group_codes = group_codes
    def __len__(self):
        return len(self.y)
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx], self.group_codes[idx]


def factorize_groups(groups):
    codes, uniques = pd.factorize(np.asarray(groups))
    return torch.tensor(codes, dtype=torch.long), list(uniques)



@torch.no_grad()
def initialize_prototypes(model, train_loader, num_classes, device, use_finch=True,
                          distance='cosine', max_per_class=None, min_cluster_samples=1):
    model.eval()
    class_features = [[] for _ in range(num_classes)]

    for batch in train_loader:
        images, labels = batch[0], batch[1]
        images = images.to(device)
        _, features_p, _, _, _ = model(images)
        features_p = features_p.cpu()

        for i in range(len(labels)):
            label = labels[i].item()
            class_features[label].append(features_p[i])

    feat_dim = features_p.shape[1]
    all_prototypes = []
    all_labels = []

    for c in range(num_classes):
        if len(class_features[c]) == 0:
            proto = torch.randn(1, feat_dim) * 0.01
            all_prototypes.append(proto)
            all_labels.extend([c])
            print(f"  Class {c}: 0 samples -> 1 random prototype")
            continue

        stacked = torch.stack(class_features[c]).numpy().astype(np.float32)  # [n_c, feat_dim]

        if use_finch and len(stacked) >= 3:
            try:
                from finch import FINCH
                c_partitions, num_clust, _ = FINCH(
                    stacked, req_clust=None, distance=distance,
                    ensure_early_exit=True, verbose=False
                )
                col = 0
                if max_per_class is not None:
                    for j in range(c_partitions.shape[1]):
                        if len(np.unique(c_partitions[:, j])) <= max_per_class:
                            col = j
                            break
                    else:
                        col = -1   
                if col >= 0:
                    partition_labels = c_partitions[:, col]
                else:
                    finest = c_partitions[:, 0]
                    uniq, cnts = np.unique(finest, return_counts=True)
                    keep = uniq[np.argsort(-cnts)[:max_per_class]]
                    partition_labels = np.where(np.isin(finest, keep), finest, -1)
                unique_labels = np.unique(partition_labels)

                protos_for_class = []
                dropped_small = 0
                for ul in unique_labels:
                    mask = partition_labels == ul
                    if ul == -1 or mask.sum() < min_cluster_samples:
                        dropped_small += int(mask.sum())
                        continue
                    cluster_feats = stacked[mask]
                    proto = cluster_feats.mean(axis=0)
                    protos_for_class.append(torch.from_numpy(proto).float())

                if len(protos_for_class) == 0:
                    proto = torch.from_numpy(stacked.mean(axis=0)).float().unsqueeze(0)
                    all_prototypes.append(proto)
                    all_labels.extend([c])
                    continue

                class_protos = torch.stack(protos_for_class)  # [n_protos_c, feat_dim]
                all_prototypes.append(class_protos)
                all_labels.extend([c] * class_protos.shape[0])
                print(f"  Class {c}: {len(stacked)} samples -> {class_protos.shape[0]} prototypes "
                      f"(FINCH partition col {col}, clusters: {num_clust}, "
                      f"filtered-out samples: {dropped_small})")
            except Exception as e:
                print(f"  Class {c}: FINCH failed ({e}), fallback to mean prototype")
                proto = torch.from_numpy(stacked.mean(axis=0)).float().unsqueeze(0)
                all_prototypes.append(proto)
                all_labels.extend([c])
        else:
            proto = torch.from_numpy(stacked.mean(axis=0)).float().unsqueeze(0)
            all_prototypes.append(proto)
            all_labels.extend([c])
            if not use_finch:
                print(f"  Class {c}: {len(stacked)} samples -> 1 mean prototype")

    prototypes = torch.cat(all_prototypes, dim=0).to(device)  # [total_prototypes, feat_dim]
    prototype_labels = torch.tensor(all_labels, dtype=torch.long, device=device)

    counts = torch.zeros(num_classes, dtype=torch.long, device=device)
    for c in range(num_classes):
        counts[c] = (prototype_labels == c).sum()

    return prototypes, prototype_labels, counts


def prototypical_loss_multi(features_p, prototypes, prototype_labels, labels):

    feat_sq = torch.sum(features_p ** 2, dim=1, keepdim=True)         
    proto_sq = torch.sum(prototypes ** 2, dim=1, keepdim=True).t()      
    cross = 2 * torch.matmul(features_p, prototypes.t())               
    distances = feat_sq + proto_sq - cross                           

    num_classes = int(prototype_labels.max().item()) + 1
    batch_size = features_p.shape[0]

    mask = torch.zeros(num_classes, prototypes.shape[0], device=features_p.device, dtype=features_p.dtype)
    valid_idx = torch.arange(prototypes.shape[0], device=features_p.device)
    mask[prototype_labels, valid_idx] = 1.0

    dist_expanded = distances.unsqueeze(1).expand(batch_size, num_classes, -1)
    mask_expanded = mask.unsqueeze(0).expand(batch_size, -1, -1)

    dist_masked = dist_expanded.clone()
    dist_masked[mask_expanded == 0] = float('inf')
    class_dist, _ = dist_masked.min(dim=2)  # [batch, num_classes]

    log_p_y = F.log_softmax(-class_dist, dim=1)
    loss = F.nll_loss(log_p_y, labels)
    return loss


def _cosine_logits(features, prototypes):
    features_norm = F.normalize(features, p=2, dim=1)
    proto_norm = F.normalize(prototypes, p=2, dim=1)
    return torch.matmul(features_norm, proto_norm.t())


def _multi_positive_log_numerator(logits, prototype_labels, labels):

    neg_inf = torch.finfo(logits.dtype).min
    pos_mask = prototype_labels.unsqueeze(0) == labels.unsqueeze(1)  # [B, P]
    pos_logits = torch.where(pos_mask, logits, torch.full_like(logits, neg_inf))
    log_sum_pos = torch.logsumexp(pos_logits, dim=1)  # [B]
    n_pos = pos_mask.sum(dim=1).clamp(min=1).to(logits.dtype)  # [B]
    return log_sum_pos - torch.log(n_pos)


def _log_denominator(logits, prototype_labels, num_classes, class_balanced):

    if not class_balanced:
        return torch.logsumexp(logits, dim=1)  # [B]

    neg_inf = torch.finfo(logits.dtype).min
    P = logits.shape[1]
    onehot = F.one_hot(prototype_labels, num_classes)  # [P, C] (0/1)
    counts = onehot.sum(dim=0)  # [C]

    expanded = logits.unsqueeze(2).expand(-1, -1, num_classes)        # [B, P, C]
    belongs = onehot.unsqueeze(0).bool().expand(logits.shape[0], -1, -1)
    masked = torch.where(belongs, expanded, torch.full_like(expanded, neg_inf))
    per_class_lse = torch.logsumexp(masked, dim=1)                    # [B, C]
    per_class_lse = per_class_lse - torch.log(counts.clamp(min=1).to(logits.dtype)).unsqueeze(0)

    valid = counts.unsqueeze(0) > 0                                   # [1, C]
    per_class_lse = torch.where(valid, per_class_lse,
                                torch.full_like(per_class_lse, neg_inf))
    return torch.logsumexp(per_class_lse, dim=1)                      # [B]


def infoNCE_loss_multi(features, prototypes, prototype_labels, labels,
                       temperature, class_balanced=True):
    sim = _cosine_logits(features, prototypes)            # [-1, 1]
    logits = sim / temperature
    num_classes = int(prototype_labels.max().item()) + 1

    log_num = _multi_positive_log_numerator(logits, prototype_labels, labels)
    log_den = _log_denominator(logits, prototype_labels, num_classes, class_balanced)
    loss = log_den - log_num                               # = -log(num/den)
    return loss.mean()


def infoNCE_alpha_loss_multi(features, prototypes, prototype_labels, labels,
                             temperature, alpha, cosine_shift=True,
                             class_balanced=True):
    sim = _cosine_logits(features, prototypes)            # [-1, 1]
    if cosine_shift:
        s01 = (sim + 1.0) / 2.0                            # [0, 1]
        powered = s01.pow(alpha)
    else:
        powered = sim.clamp(min=0.0, max=1.0).pow(alpha)
    logits = powered / temperature

    num_classes = int(prototype_labels.max().item()) + 1
    log_num = _multi_positive_log_numerator(logits, prototype_labels, labels)
    log_den = _log_denominator(logits, prototype_labels, num_classes, class_balanced)
    loss = log_den - log_num
    return loss.mean()



def compute_train_loss(log_probs, log_probs_p, features_p, centers, labels, cfg,
                       proto_labels=None):

    if cfg.METHODS == 4:
        loss_info_alpha = infoNCE_alpha_loss_multi(
            features_p, centers, proto_labels, labels,
            cfg.TEMPERATURE, cfg.ALPHA,
            cosine_shift=getattr(cfg, 'COSINE_SHIFT', True),
            class_balanced=getattr(cfg, 'CLASS_BALANCED_INFONCE', True))
        loss = cfg.L1 * loss1 + cfg.L2 * loss_info_alpha
    else:
        raise ValueError(f"error METHODS: {cfg.METHODS}")

    return loss


def compute_eval_loss(log_probs, log_probs_p, features_p, centers, labels, cfg,
                      proto_labels=None):
    return compute_train_loss(log_probs, log_probs_p, features_p, centers,
                              labels, cfg, proto_labels)


def _use_baseline_head(cfg):
    return getattr(cfg, 'METHODS', 2) == -1


def train_one_epoch(model, train_loader, optimizer, device, cfg):
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for images, labels, _ in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()

        log_probs, features_p, centers, dists, log_probs_p = model(images)
        proto_labels = model.dce.prototype_labels if hasattr(model.dce, 'prototype_labels') else None
        loss = compute_train_loss(log_probs, log_probs_p, features_p, centers,
                                  labels, cfg, proto_labels)

        loss.backward()
        optimizer.step()

        head = log_probs if _use_baseline_head(cfg) else log_probs_p
        _, predicted = torch.max(head.data, 1)
        total_samples += labels.size(0)
        total_correct += (predicted == labels).sum().item()
        total_loss += loss.item() * labels.size(0)

    return total_loss / total_samples, total_correct / total_samples


@torch.no_grad()
def evaluate(model, loader, device, cfg):
    
    model.eval()
    total_loss = 0.0
    all_labels = []
    all_preds = []
    all_probs = []
    all_groups = []

    for images, labels, group_codes in loader:
        images, labels = images.to(device), labels.to(device)
        log_probs, features_p, centers, dists, log_probs_p = model(images)
        proto_labels = model.dce.prototype_labels if hasattr(model.dce, 'prototype_labels') else None
        loss = compute_eval_loss(log_probs, log_probs_p, features_p, centers,
                                 labels, cfg, proto_labels)
        total_loss += loss.item() * labels.size(0)

        head = log_probs if _use_baseline_head(cfg) else log_probs_p
        probs = torch.exp(head)  # log_softmax -> softmax
        _, predicted = torch.max(head.data, 1)

        all_labels.append(labels.cpu().numpy())
        all_preds.append(predicted.cpu().numpy())
        all_probs.append(probs.cpu().numpy())
        all_groups.append(group_codes.numpy())

    all_labels = np.concatenate(all_labels)
    all_preds = np.concatenate(all_preds)
    all_probs = np.concatenate(all_probs)
    all_groups = np.concatenate(all_groups)
    avg_loss = total_loss / max(len(all_labels), 1)

    return {
        'loss': avg_loss,
        'labels': all_labels,
        'preds': all_preds,
        'probs': all_probs,
        'group_codes': all_groups,
    }


def build_model(cfg, num_classes, seq_length, device):
    ModelClass = get_model_class(cfg.MODEL_NAME)
    model_kwargs = {
        'in_channels': cfg.IN_CHANNELS,
        'num_classes': num_classes,
        'num_hidden_units': cfg.NUM_HIDDEN_UNITS,
    }
    if cfg.MODEL_NAME in ['CNN4', 'VGG16']:
        model_kwargs['input_length'] = seq_length
    if cfg.MODEL_NAME == 'GoogleNet':
        model_kwargs['aux_logits'] = False
    model = ModelClass(**model_kwargs).to(device)
    if hasattr(model, 'dce') and hasattr(model.dce, 'configure_tau'):
        model.dce.configure_tau(
            tau_init=getattr(cfg, 'DCE_TAU_INIT', 1.0),
            learnable=getattr(cfg, 'DCE_TAU_LEARNABLE', True),
            tau_min=getattr(cfg, 'DCE_TAU_MIN', 0.05),
            tau_max=getattr(cfg, 'DCE_TAU_MAX', 5.0))
    return model



def run_5fold_cv(cfg=None, fold_callback=None):

    if cfg is None:
        cfg = Config
        cfg.SAVE_DIR = os.path.join(cfg.SAVE_DIR, cfg.DATASET, cfg.MODEL_NAME, f'method_{cfg.METHODS}')
        os.makedirs(cfg.SAVE_DIR, exist_ok=True)
    else:
        os.makedirs(cfg.SAVE_DIR, exist_ok=True)

    device = torch.device(cfg.DEVICE if torch.cuda.is_available() else 'cpu')
    

    X, y, groups = load_data(cfg.DATA_PATH, has_patient_id=True)
    if y.min() != 0:
        y = y - y.min()
    num_classes = int(y.max().item()) + 1 if cfg.NUM_CLASSES is None else cfg.NUM_CLASSES
    seq_length = X.shape[2]
    group_codes_all, unique_groups = factorize_groups(groups)
    n_patients = len(unique_groups)


    if n_patients < 5:
        raise ValueError(f"({n_patients}) ＜ 5")

    torch.manual_seed(cfg.SEED)
    np.random.seed(cfg.SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(cfg.SEED)
        torch.backends.cudnn.deterministic = True

    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=cfg.SEED)

    fold_results = []
    best_uar = -1.0
    best_fold_idx = -1
    best_model_state = None
    best_centers_state = None
    best_proto_labels_state = None
    best_proto_counts_state = None
    best_train_data = None
    best_val_data = None
    best_val_preds = None

    groups_np = np.asarray(groups)

    for fold, (train_idx, val_idx) in enumerate(
            sgkf.split(X.numpy(), y.numpy(), groups_np)):
        print(f"\n{'='*60}")
        print(f"Fold {fold + 1} / 5")
        print(f"{'='*60}")

        train_pids = set(groups_np[train_idx].tolist())
        val_pids = set(groups_np[val_idx].tolist())
        overlap = train_pids & val_pids
        if overlap:
            raise RuntimeError(
                f"Fold {fold + 1}: train/val has same id: "
                f"{sorted(overlap)[:10]}")

        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        g_train, g_val = group_codes_all[train_idx], group_codes_all[val_idx]

        train_loader = DataLoader(TensorDataset(X_train, y_train, g_train),
                                  batch_size=cfg.BATCH_SIZE, shuffle=True,
                                  drop_last=False, num_workers=0)
        val_loader = DataLoader(TensorDataset(X_val, y_val, g_val),
                                batch_size=cfg.BATCH_SIZE, shuffle=False,
                                drop_last=False, num_workers=0)

        model = build_model(cfg, num_classes, seq_length, device)

        
        if cfg.METHODS == -1:
            proto_counts_list = [0] * num_classes
        else:
           
            warmup_epochs = getattr(cfg, 'FINCH_WARMUP_EPOCHS', 0)
            if warmup_epochs > 0:
                warmup_backbone(model, train_loader, device, cfg, warmup_epochs)
            use_multi = getattr(cfg, 'USE_MULTI_PROTOTYPE', True)
            finch_dist = getattr(cfg, 'FINCH_DISTANCE', 'cosine')
            max_proto = getattr(cfg, 'MAX_PROTOTYPES_PER_CLASS', None)
            min_proto_samples = getattr(cfg, 'MIN_SAMPLES_PER_PROTOTYPE', 1)
            prototypes_data, proto_labels, proto_counts = initialize_prototypes(
                model, train_loader, num_classes, device,
                use_finch=use_multi, distance=finch_dist,
                max_per_class=max_proto, min_cluster_samples=min_proto_samples
            )
            model.dce.set_prototypes(prototypes_data, proto_labels, proto_counts)
            proto_counts_list = proto_counts.cpu().tolist()
           
        optimizer = torch.optim.Adam(model.parameters(),
                                     lr=cfg.LR, weight_decay=cfg.WEIGHT_DECAY)
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1,
                                      patience=max(1, cfg.PATIENCE // 2))

     
        best_fold_uar = -1.0
        best_fold_loss = float('inf')
        best_fold_state = None
        epochs_no_improve = 0

        for epoch in range(cfg.EPOCHS):
            train_loss, train_acc = train_one_epoch(
                model, train_loader, optimizer, device, cfg
            )
            val_out = evaluate(model, val_loader, device, cfg)
            val_loss = val_out['loss']
            val_uar = recall_score(val_out['labels'], val_out['preds'],
                                   average='macro', zero_division=0)

            scheduler.step(val_loss)

            print(f"Epoch {epoch+1:03d} | "
                  f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
                  f"Val Loss: {val_loss:.4f} UAR: {val_uar:.4f}")

            if val_uar > best_fold_uar or (val_uar == best_fold_uar and val_loss < best_fold_loss):
                best_fold_uar = val_uar
                best_fold_loss = val_loss
                best_fold_state = copy.deepcopy(model.state_dict())
                epochs_no_improve = 0
                print(f"  -> New best (Val UAR={val_uar:.4f})")
            else:
                epochs_no_improve += 1

            if epochs_no_improve >= cfg.PATIENCE:
                print(f"  -> Early stopping triggered at epoch {epoch+1}")
                break

        model.load_state_dict(best_fold_state)
        val_out = evaluate(model, val_loader, device, cfg)
        val_metrics = compute_extended_metrics(
            val_out['labels'], val_out['preds'], probs=val_out['probs'],
            groups=val_out['group_codes'], num_classes=num_classes,
            n_bootstrap=getattr(cfg, 'N_BOOTSTRAP', 1000),
            seed=getattr(cfg, 'BOOTSTRAP_SEED', 42))
        val_uar = val_metrics['spectrum']['uar']
        val_uf1 = val_metrics['spectrum']['uf1']
        val_acc = val_metrics['spectrum']['accuracy']
        val_loss = val_out['loss']


        cur_tau = (model.dce.get_tau() if hasattr(model, 'dce')
                   and hasattr(model.dce, 'get_tau') else None)
        if cur_tau is not None:

        fold_results.append({
            'fold': fold + 1,
            'uar': val_uar,
            'dce_tau': cur_tau,
            'uf1': val_uf1,
            'acc': val_acc,
            'loss': val_loss,
            'cm': val_metrics['spectrum']['confusion_matrix'],
            'proto_counts': proto_counts_list,
            'metrics': val_metrics,
        })

        if fold_callback is not None:
            fold_callback(fold + 1, fold_results[-1])

        if val_uar > best_uar:
            best_uar = val_uar
            best_fold_idx = fold + 1
            best_model_state = best_fold_state
            if cfg.METHODS != -1 and hasattr(model.dce, 'prototypes'):
                best_centers_state = model.dce.prototypes.detach().cpu().clone()
                best_proto_labels_state = model.dce.prototype_labels.detach().cpu().clone()
                best_proto_counts_state = model.dce.prototype_counts.detach().cpu().clone()
            best_train_data = (X_train.numpy(), y_train.numpy(), groups_np[train_idx])
            best_val_data = (X_val.numpy(), y_val.numpy(), groups_np[val_idx])
            best_val_preds = (np.asarray(val_out['labels']),
                              np.asarray(val_out['preds']),
                              np.asarray(val_out['probs']),
                              np.asarray(groups_np[val_idx]).astype(str))

    print(f"\n{'='*60}")
    print(f"Mean Val UAR: {np.mean([r['uar'] for r in fold_results]):.4f} "
          f"± {np.std([r['uar'] for r in fold_results]):.4f}")
    print(f"{'='*60}")

    model_path = os.path.join(cfg.SAVE_DIR, 'best_model.pth')
    torch.save({
        'model_state_dict': best_model_state,
        'model_name': cfg.MODEL_NAME,
        'num_classes': num_classes,
        'num_hidden_units': cfg.NUM_HIDDEN_UNITS,
        'in_channels': cfg.IN_CHANNELS,
        'methods': cfg.METHODS,
        'uar': best_uar,
        'fold': best_fold_idx,
    }, model_path)

    np.savez(os.path.join(cfg.SAVE_DIR, 'best_fold_train.npz'),
             X=best_train_data[0], y=best_train_data[1],
             groups=best_train_data[2], fold=best_fold_idx)
    np.savez(os.path.join(cfg.SAVE_DIR, 'best_fold_val.npz'),
             X=best_val_data[0], y=best_val_data[1],
             groups=best_val_data[2], fold=best_fold_idx)
    np.savez(os.path.join(cfg.SAVE_DIR, 'best_fold_val_predictions.npz'),
             labels=best_val_preds[0], preds=best_val_preds[1],
             probs=best_val_preds[2], groups=best_val_preds[3],
             fold=best_fold_idx)

    if best_centers_state is not None:
        np.savez(os.path.join(cfg.SAVE_DIR, 'best_fold_centers.npz'),
                 centers=best_centers_state.numpy(),
                 prototype_labels=best_proto_labels_state.numpy(),
                 prototype_counts=best_proto_counts_state.numpy(),
                 fold=best_fold_idx)

    return fold_results, best_fold_idx, num_classes



def save_cv_summary(fold_results, best_fold_idx, cfg):
    uars = [r['uar'] for r in fold_results]
    uf1s = [r['uf1'] for r in fold_results]
    accs = [r['acc'] for r in fold_results]

    results_path = os.path.join(cfg.SAVE_DIR, 'results.txt')
    with open(results_path, 'w', encoding='utf-8') as f:
        f.write("===== Experiment Configuration =====\n")
        f.write(f"Dataset: {cfg.DATASET}\n")
        f.write(f"Model: {cfg.MODEL_NAME}\n")
        f.write(f"Method: {cfg.METHODS}\n")
        f.write(f"L1: {cfg.L1}, L2: {cfg.L2}\n")
        f.write(f"Temperature: {cfg.TEMPERATURE}, Alpha: {cfg.ALPHA}\n")
        f.write(f"COSINE_SHIFT: {getattr(cfg, 'COSINE_SHIFT', True)}, "
                f"CLASS_BALANCED_INFONCE: {getattr(cfg, 'CLASS_BALANCED_INFONCE', True)}\n")
        f.write(f"Epochs: {cfg.EPOCHS}, BatchSize: {cfg.BATCH_SIZE}, LR: {cfg.LR}\n")
        f.write(f"NumHiddenUnits: {cfg.NUM_HIDDEN_UNITS}\n")
        f.write(f"MultiPrototype: {getattr(cfg, 'USE_MULTI_PROTOTYPE', True)}\n")
        f.write("=" * 40 + "\n\n")

        f.write("===== 5-Fold Cross Validation Summary (patient-level) =====\n")
        f.write(f"Best Fold: Fold {best_fold_idx} (UAR={max(uars):.4f})\n\n")

        for r in fold_results:
            f.write(f"Fold {r['fold']}:\n")
            f.write(f"  Per-class prototype counts: {r['proto_counts']}\n")
            f.write(format_metrics_report(r['metrics'],
                                          title=f"Fold {r['fold']} Validation")
                    + "\n\n")

        f.write("----- Overall Statistics (validation) -----\n")
        f.write(f"Mean UAR : {np.mean(uars):.4f} ± {np.std(uars):.4f}\n")
        f.write(f"Mean UF1 : {np.mean(uf1s):.4f} ± {np.std(uf1s):.4f}\n")
        f.write(f"Mean Acc : {np.mean(accs):.4f} ± {np.std(accs):.4f}\n")



def test_best_model(num_classes, cfg=None):

    if cfg is None:
        cfg = Config
        expected_suffix = f'method_{cfg.METHODS}'
        if expected_suffix not in cfg.SAVE_DIR:
            cfg.SAVE_DIR = os.path.join(cfg.SAVE_DIR, cfg.DATASET, cfg.MODEL_NAME, expected_suffix)
            os.makedirs(cfg.SAVE_DIR, exist_ok=True)
    else:
        if not os.path.exists(cfg.SAVE_DIR):
            os.makedirs(cfg.SAVE_DIR, exist_ok=True)

    if cfg.TEST_DATA_PATH is None or not os.path.exists(cfg.TEST_DATA_PATH):
        return None

    device = torch.device(cfg.DEVICE if torch.cuda.is_available() else 'cpu')

    X_test, y_test, groups_test = load_data(cfg.TEST_DATA_PATH, has_patient_id=True)
    if y_test.min() != 0:
        y_test = y_test - y_test.min()
    g_codes, _ = factorize_groups(groups_test)

    test_loader = DataLoader(TensorDataset(X_test, y_test, g_codes),
                             batch_size=cfg.BATCH_SIZE,
                             shuffle=False, drop_last=False, num_workers=0)

    model = build_model(cfg, num_classes, X_test.shape[2], device)

    model_path = os.path.join(cfg.SAVE_DIR, 'best_model.pth')
    checkpoint = torch.load(model_path, map_location=device)
    state_dict = checkpoint['model_state_dict']

    if 'dce.prototypes' in state_dict and state_dict['dce.prototypes'].shape[0] > 0:
        proto = state_dict['dce.prototypes']
        labels = state_dict['dce.prototype_labels']
        counts = state_dict['dce.prototype_counts']
        model.dce.set_prototypes(proto, labels, counts)

    model.load_state_dict(state_dict)

    test_out = evaluate(model, test_loader, device, cfg)
    test_metrics = compute_extended_metrics(
        test_out['labels'], test_out['preds'], probs=test_out['probs'],
        groups=test_out['group_codes'], num_classes=num_classes,
        n_bootstrap=getattr(cfg, 'N_BOOTSTRAP', 1000),
        seed=getattr(cfg, 'BOOTSTRAP_SEED', 42))

    report = format_metrics_report(test_metrics, title="Independent Test Set Results",
                                   num_classes=num_classes)

    results_path = os.path.join(cfg.SAVE_DIR, 'results.txt')
    with open(results_path, 'a', encoding='utf-8') as f:
        f.write("\n" + report + "\n")

    pred_path = os.path.join(cfg.SAVE_DIR, 'test_predictions.npz')
    np.savez(pred_path,
             labels=np.asarray(test_out['labels']),
             preds=np.asarray(test_out['preds']),
             probs=np.asarray(test_out['probs']),
             groups=np.asarray(groups_test).astype(str))

    print(f"\n{report}")

    return test_metrics


def create_cfg_from_search_config(search_config):
    from types import SimpleNamespace
    cfg = SimpleNamespace()
    for attr in [a for a in dir(Config) if not a.startswith('_') and not callable(getattr(Config, a))]:
        setattr(cfg, attr, getattr(Config, attr))
    for key, value in search_config.items():
        setattr(cfg, key, value)
    return cfg


def warmup_backbone(model, train_loader, device, cfg, warmup_epochs):

    if warmup_epochs <= 0:
        return
    was_training = model.training
    model.train()
    warmup_opt = torch.optim.Adam(model.parameters(), lr=cfg.LR,
                                  weight_decay=cfg.WEIGHT_DECAY)
    for ep in range(warmup_epochs):
        total_loss, total_correct, total_n = 0.0, 0, 0
        for images, labels, _g in train_loader:
            images, labels = images.to(device), labels.to(device)
            warmup_opt.zero_grad()
            log_probs, _fp, _c, _d, _lp = model(images)
            loss = F.nll_loss(log_probs, labels)   
            loss.backward()
            warmup_opt.step()
            total_loss += loss.item() * labels.size(0)
            total_correct += (log_probs.argmax(dim=1) == labels).sum().item()
            total_n += labels.size(0)
        print(f"  [warmup] epoch {ep+1}/{warmup_epochs} "
              f"loss={total_loss/total_n:.4f} acc={total_correct/total_n:.4f}")
    if not was_training:
        model.eval()


class Fold1Pruned(Exception):
    pass


def ray_trainable(search_config):
    import importlib
    from ray import tune

    _train_module = importlib.import_module('train_5fold_dce')

    cfg = _train_module.create_cfg_from_search_config(search_config)

    try:
        trial_dir = tune.get_trial_dir()
    except AttributeError:
        trial_dir = tune.get_context().get_trial_dir()

    cfg.SAVE_DIR = trial_dir

    prune_enable = getattr(cfg, 'RAY_PRUNE_FOLD1', False)
    min_fold1_uar = getattr(cfg, 'RAY_MIN_FOLD1_UAR', 0.60)

    def _fold1_guard(fold_idx, fold_result):
        if prune_enable and fold_idx == 1 and fold_result['uar'] < min_fold1_uar:
            raise _train_module.Fold1Pruned(
                f"fold 1 val UAR={fold_result['uar']:.4f} < {min_fold1_uar:.2f}，"

    fold_callback = _fold1_guard if prune_enable else None
    fold_results, best_fold_idx, num_classes = _train_module.run_5fold_cv(
        cfg, fold_callback=fold_callback)
    _train_module.save_cv_summary(fold_results, best_fold_idx, cfg)

    mean_val_uar = float(np.mean([r['uar'] for r in fold_results]))
    best_fold_val_uar = float(max(r['uar'] for r in fold_results))

   

    return {
        "mean_val_uar": mean_val_uar,       
        "best_fold_val_uar": best_fold_val_uar, 
        
    }


def run_ray_tune():
    try:
        from ray import tune
        from ray.tune.search.optuna import OptunaSearch
        from optuna.samplers import TPESampler
    except ImportError as e:
        raise ImportError(
            "Ray Tune 路径需要安装依赖: pip install 'ray[tune]' optuna"
        ) from e
    from pathlib import Path

    cfg = Config

   
    if getattr(cfg, 'RAY_SPACE_MODE', 'continuous') == 'discrete':
        search_space = {
            "BATCH_SIZE": tune.choice([16, 32, 64, 128, 256]),
            "LR": tune.choice([1e-4, 5e-4, 1e-3, 5e-3, 1e-5,5e-5]),
        }
        if cfg.METHODS >= 2:
            search_space["L1"] = tune.choice([1e-2, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
            search_space["L2"] = tune.choice([1e-2, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
        if cfg.METHODS >= 3:
            search_space["TEMPERATURE"] = tune.choice([0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
        if cfg.METHODS >= 4:
            search_space["ALPHA"] = tune.choice([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
    else:
        search_space = {
            "BATCH_SIZE": tune.choice([16, 32, 64, 128, 256, 512]),
            "LR": tune.loguniform(1e-5, 1e-2),
        }
        if cfg.METHODS >= 2:
            search_space["L1"] = tune.uniform(1e-4, 1.0)
            search_space["L2"] = tune.uniform(1e-4, 1.0)
        if cfg.METHODS >= 3:
            search_space["TEMPERATURE"] = tune.uniform(0.01, 1.0)
        if cfg.METHODS >= 4:
            search_space["ALPHA"] = tune.uniform(1e-4, 0.9999)

    if getattr(cfg, 'RAY_SEARCH_HIDDEN_UNITS', False):
        search_space["NUM_HIDDEN_UNITS"] = tune.choice([8, 16, 32, 64, 128])

    if getattr(cfg, 'RAY_SEARCH_DCE_TAU', False):
        search_space["DCE_TAU_INIT"] = tune.choice(
            [0.05, 0.07, 0.1, 0.15, 0.2, 0.3, 0.5, 0.7, 1.0])

    print(f"Ray Tune 搜索空间 (METHODS={cfg.METHODS}):")
    for k, v in search_space.items():
        print(f"  {k}: {v}")
    objective = getattr(cfg, 'RAY_SEARCH_OBJECTIVE', 'best_fold')
    metric_name = {'best_fold': 'best_fold_val_uar', 'mean': 'mean_val_uar'}[objective]
    

    ray_dir = os.path.abspath(os.path.join(cfg.SAVE_DIR, 'ray_results'))
    os.makedirs(ray_dir, exist_ok=True)
    storage_uri = Path(ray_dir).as_uri()

    sampler = TPESampler(n_startup_trials=cfg.RAY_RANDOM_STEPS, seed=cfg.SEED)

    analysis = tune.run(
        ray_trainable,
        config=search_space,
        metric=metric_name,
        mode="max",
        num_samples=cfg.RAY_SEARCH_ITER,
        resources_per_trial={"cpu": 2, "gpu": 1 if cfg.DEVICE == 'cuda' else 0},
        verbose=1,
        storage_path=storage_uri,
        raise_on_failed_trial=False,
        max_failures=1,
        search_alg=OptunaSearch(sampler=sampler),
    )

    completed_trials = [t for t in analysis.trials if t.status == "TERMINATED"]
    if not completed_trials:
        print(f"\n{'='*60}")
        print(f"{'='*60}")
        return analysis

    try:
        trials_csv = os.path.join(cfg.SAVE_DIR, 'ray_trials_summary.csv')
        df = analysis.dataframe(metric=metric_name, mode="max")
        df.to_csv(trials_csv, index=False)
        print(f"trial saved")
    except Exception as e:
        print(f"trial error: {e}")

    best_trial = max(completed_trials,
                     key=lambda t: t.last_result.get(metric_name, -1))
    best_config = best_trial.config
    best_search_uar = best_trial.last_result.get(metric_name, -1)

    summary_dir = os.path.join(
        cfg.SAVE_DIR, cfg.DATASET, cfg.MODEL_NAME, f'method_{cfg.METHODS}'
    )
    os.makedirs(summary_dir, exist_ok=True)

    final_cfg = create_cfg_from_search_config(best_config)
    final_cfg.SAVE_DIR = summary_dir
    fold_results, best_fold_idx, num_classes = run_5fold_cv(final_cfg)
    save_cv_summary(fold_results, best_fold_idx, final_cfg)
    test_metrics = test_best_model(num_classes, final_cfg)

    summary_path = os.path.join(summary_dir, 'ray_best_config.txt')
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write("===== Ray Tune Best Configuration =====\n")
        f.write(f"Dataset: {cfg.DATASET}\n")
        f.write(f"Model: {cfg.MODEL_NAME}\n")
        f.write(f"Method: {cfg.METHODS}\n")
        f.write(f"Search Iterations: {cfg.RAY_SEARCH_ITER}\n")
        f.write(f"Random Steps (TPESampler n_startup_trials): {cfg.RAY_RANDOM_STEPS}\n")
        f.write(f"Search objective: {metric_name} "
                f"({'best of 5 folds validation UAR' if objective == 'best_fold' else 'mean of 5 folds validation UAR'})\n")
        f.write(f"(test set metrics recorded per trial for diagnosis only, never used in selection)\n\n")
        f.write(f"Best Trial: {best_trial.trial_id}\n")
        f.write(f"Best search {metric_name}: {best_search_uar:.4f}\n")
        f.write(f"Retrained mean val UAR  : "
                f"{np.mean([r['uar'] for r in fold_results]):.4f}\n")
        f.write(f"Retrained best-fold val UAR: "
                f"{max(r['uar'] for r in fold_results):.4f}\n")
        
        for k, v in best_config.items():
            f.write(f"  {k}: {v}\n")

    return analysis


if __name__ == '__main__':
    if Config.USE_RAY:
        run_ray_tune()
    else:
        fold_results, best_fold_idx, num_classes = run_5fold_cv()
        save_cv_summary(fold_results, best_fold_idx, Config)
        test_best_model(num_classes)
