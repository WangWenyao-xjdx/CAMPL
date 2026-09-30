"""
train_5fold_dce.py
==================
单机五折交叉验证 + DCE Loss 训练脚本 — 多原型版本（修订版）。

修订要点（对应审稿意见）：
1. 五折使用 StratifiedGroupKFold（groups = patient_id），每折强断言
   train/val 患者集合不相交，保证患者级独立性。
2. InfoNCE / InfoNCE-alpha 数值稳定（logsumexp + max 减法）；
   InfoNCE-alpha 默认先将余弦相似度从 [-1,1] 平移到 [0,1]（s01=(s+1)/2）
   再 pow(alpha)（Config.COSINE_SHIFT 开关）。
3. InfoNCE 分子为多样本正例的 log-mean-exp（同类所有原型均为正样本但
   取平均），分母支持按类均衡（Config.CLASS_BALANCED_INFONCE），避免
   FINCH 原型多的类在分子或分母中被过度加权。
4. compute_eval_loss 与 compute_train_loss 使用完全相同的 L1/L2 权重。
5. METHODS=-1 新增基线：仅使用普通分类头（fc）的交叉熵。
6. 评估指标扩展：spectrum 级 UAR/UF1/Acc/混淆矩阵/每类 precision/recall/
   F1/specificity/macro AUC/bootstrap 95% CI；有患者 id 时输出患者级聚合指标。
7. 正确保存原型：best_fold_centers.npz 含 centers / prototype_labels /
   prototype_counts；results.txt 记录每折每类原型数。
8. Ray Tune：trial 只优化验证集 mean UAR，绝不接触测试集；
   OptunaSearch + TPESampler(n_startup_trials=Config.RAY_RANDOM_STEPS)；
   搜索结束后用最优配置重跑一次五折，并只对测试集评估一次。
"""

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


# ============================ 【动态导入模型】 ============================

def get_model_class(model_name):
    """根据配置名动态返回模型类"""
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
    elif model_name == 'AttentionCNN1D':
        from models.AttentionCNN1D import AttentionCNN1D
        return AttentionCNN1D
    elif model_name == 'Transformer1D':
        from models.Transformer1D import Transformer1D
        return Transformer1D
    elif model_name == 'Mamba1D':
        from models.Mamba1D import Mamba1D
        return Mamba1D
    else:
        raise ValueError(f"不支持的模型名称: {model_name}")


# ============================ 工具类 ============================

class TensorDataset(Dataset):
    """返回 (X, y, group_code)；group_code 为患者 id 的整数编码。"""
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
    """把患者 id 数组编码为整数 code，返回 (codes_tensor, unique_ids)。"""
    codes, uniques = pd.factorize(np.asarray(groups))
    return torch.tensor(codes, dtype=torch.long), list(uniques)


# ============================ 多原型初始化（FINCH） ============================

@torch.no_grad()
def initialize_prototypes(model, train_loader, num_classes, device, use_finch=True,
                          distance='cosine', max_per_class=None, min_cluster_samples=1):
    """
    初始化原型。use_finch=True 时用FINCH聚类生成多原型；否则每类1个均值原型。
    max_per_class: 每类原型数上限。FINCH 返回从细到粗的多层分区，
        取第一个聚类数 <= max_per_class 的分区；若最粗分区仍超过上限，
        则保留成员数最多的前 max_per_class 个簇。None = 不限制（最细分区）。
    min_cluster_samples: 成员少于此数的簇不作为原型（噪声簇过滤）；
        若过滤后该类为空，回退为该类的均值原型。
    返回统一格式: (prototypes, prototype_labels, prototype_counts)
    """
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
            # 无样本类：随机初始化1个原型
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
                # 分区选择：默认取最细分区（第0列）；若设置了每类原型上限，
                # 则在 FINCH 的层级分区中找第一个聚类数 <= 上限的分区
                col = 0
                if max_per_class is not None:
                    for j in range(c_partitions.shape[1]):
                        if len(np.unique(c_partitions[:, j])) <= max_per_class:
                            col = j
                            break
                    else:
                        col = -1   # 所有分区都超上限，后面用 top-K 截断
                if col >= 0:
                    partition_labels = c_partitions[:, col]
                else:
                    # top-K 截断：保留成员数最多的 max_per_class 个簇
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
                        continue   # top-K 被截掉的样本 / 噪声小簇不作为原型
                    cluster_feats = stacked[mask]
                    proto = cluster_feats.mean(axis=0)
                    protos_for_class.append(torch.from_numpy(proto).float())

                if len(protos_for_class) == 0:
                    # 过滤后为空：回退为该类均值原型
                    proto = torch.from_numpy(stacked.mean(axis=0)).float().unsqueeze(0)
                    all_prototypes.append(proto)
                    all_labels.extend([c])
                    print(f"  Class {c}: {len(stacked)} samples -> 全部簇被过滤, "
                          f"回退为 1 个均值原型")
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
            # 单原型模式 或 样本不足
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


# ============================ 损失函数（多原型版本） ============================

def prototypical_loss_multi(features_p, prototypes, prototype_labels, labels):
    """
    多原型 Prototypical Loss。
    每个类的logit = - min_{p in class c} ||f - p||^2 （最近原型距离）
    """
    # 计算到所有原型的平方欧氏距离 [batch, total_prototypes]
    feat_sq = torch.sum(features_p ** 2, dim=1, keepdim=True)          # [batch, 1]
    proto_sq = torch.sum(prototypes ** 2, dim=1, keepdim=True).t()      # [1, total_prototypes]
    cross = 2 * torch.matmul(features_p, prototypes.t())                # [batch, total_prototypes]
    distances = feat_sq + proto_sq - cross                            # [batch, total_prototypes]

    num_classes = int(prototype_labels.max().item()) + 1
    batch_size = features_p.shape[0]

    # 掩码无关原型为 +inf，然后按类取最小距离
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
    """归一化余弦相似度矩阵 [batch, total_prototypes]，值域 [-1, 1]。"""
    features_norm = F.normalize(features, p=2, dim=1)
    proto_norm = F.normalize(prototypes, p=2, dim=1)
    return torch.matmul(features_norm, proto_norm.t())


def _multi_positive_log_numerator(logits, prototype_labels, labels):
    """
    多正样本分子：log  mean_{p: class(p)==label}  exp(logit_p)
    同类所有原型均为正样本（对应论文 Figure 4），但按正原型数取平均，
    避免原型数多的类在分子中被额外加权；logsumexp 数值稳定。
    与类均衡分母搭配时损失保持非负。
    """
    neg_inf = torch.finfo(logits.dtype).min
    pos_mask = prototype_labels.unsqueeze(0) == labels.unsqueeze(1)  # [B, P]
    pos_logits = torch.where(pos_mask, logits, torch.full_like(logits, neg_inf))
    log_sum_pos = torch.logsumexp(pos_logits, dim=1)  # [B]
    n_pos = pos_mask.sum(dim=1).clamp(min=1).to(logits.dtype)  # [B]
    return log_sum_pos - torch.log(n_pos)


def _log_denominator(logits, prototype_labels, num_classes, class_balanced):
    """
    InfoNCE 分母（log 域）。
    class_balanced=True: 先对每个类的原型做 logsumexp 再减去 log(该类原型数)，
                         最后对所有类 logsumexp —— 每个类权重相同，
                         原型多的类不会被过度加权。
    class_balanced=False: 直接对所有原型 logsumexp（旧行为）。
    """
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
    """
    多原型 InfoNCE（数值稳定版）。
    分子：同类所有原型共同作为正样本（logsumexp）。
    分母：class_balanced 控制是否按类均衡。
    """
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
    """
    多原型 InfoNCE-α（数值稳定版）。
    cosine_shift=True（默认，对应修订后公式）:
        s ∈ [-1,1] → s01 = (s+1)/2 ∈ [0,1] → logits = s01**alpha / temperature
    cosine_shift=False（旧行为）:
        负相似度 clamp 到 0 后 pow(alpha)。
    """
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


# ============================ 动态损失组合 ============================

def compute_train_loss(log_probs, log_probs_p, features_p, centers, labels, cfg,
                       proto_labels=None):
    """根据 METHODS 动态计算训练损失（支持多原型）"""
    loss1 = F.nll_loss(log_probs_p, labels)   # DCE 头（原型余弦相似度）交叉熵

    if cfg.METHODS == -1:
        # 基线：仅普通分类头
        loss = F.nll_loss(log_probs, labels)
    elif cfg.METHODS == 0:
        loss = prototypical_loss_multi(features_p, centers, proto_labels, labels)
    elif cfg.METHODS == 1:
        loss = loss1
    elif cfg.METHODS == 2:
       loss = infoNCE_alpha_loss_multi(           
           features_p, centers, proto_labels, labels,
           cfg.TEMPERATURE, cfg.ALPHA,
           cosine_shift=getattr(cfg, 'COSINE_SHIFT', True),
           class_balanced=getattr(cfg, 'CLASS_BALANCED_INFONCE', True))
    elif cfg.METHODS == 3:
        loss_info = infoNCE_loss_multi(
            features_p, centers, proto_labels, labels, cfg.TEMPERATURE,
            class_balanced=getattr(cfg, 'CLASS_BALANCED_INFONCE', True))
        loss = cfg.L1 * loss1 + cfg.L2 * loss_info
    elif cfg.METHODS == 4:
        loss_info_alpha = infoNCE_alpha_loss_multi(
            features_p, centers, proto_labels, labels,
            cfg.TEMPERATURE, cfg.ALPHA,
            cosine_shift=getattr(cfg, 'COSINE_SHIFT', True),
            class_balanced=getattr(cfg, 'CLASS_BALANCED_INFONCE', True))
        loss = cfg.L1 * loss1 + cfg.L2 * loss_info_alpha
    else:
        raise ValueError(f"不支持的 METHODS: {cfg.METHODS}")

    return loss


def compute_eval_loss(log_probs, log_probs_p, features_p, centers, labels, cfg,
                      proto_labels=None):
    """
    验证/测试损失 —— 与 compute_train_loss 使用完全相同的 L1/L2 权重。
    （修复旧版 eval 损失丢弃 L1/L2 的不一致问题。）
    """
    return compute_train_loss(log_probs, log_probs_p, features_p, centers,
                              labels, cfg, proto_labels)


# ============================ 训练与评估 ============================

def _use_baseline_head(cfg):
    """METHODS=-1 基线：预测与概率均来自普通分类头。"""
    return getattr(cfg, 'METHODS', 2) == -1


def train_one_epoch(model, train_loader, optimizer, device, cfg):
    """训练一个 epoch"""
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
    """
    评估模型。返回 dict:
      loss, labels, preds, probs, group_codes
    probs 为所选预测头的 softmax 概率（用于 AUC / 患者级聚合）。
    """
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
    """按配置实例化模型（不再传入 centers）。"""
    ModelClass = get_model_class(cfg.MODEL_NAME)
    model_kwargs = {
        'in_channels': cfg.IN_CHANNELS,
        'num_classes': num_classes,
        'num_hidden_units': cfg.NUM_HIDDEN_UNITS,
    }
    if cfg.MODEL_NAME in ['CNN4', 'VGG16']:
        model_kwargs['input_length'] = seq_length
    # AttentionCNN1D / Transformer1D / Mamba1D 使用自适应池化，长度无关
    if cfg.MODEL_NAME == 'GoogleNet':
        model_kwargs['aux_logits'] = False
    model = ModelClass(**model_kwargs).to(device)
    # 应用 DCE 温度配置（所有模型共享 CNN4.dce_loss；需在创建优化器之前）
    if hasattr(model, 'dce') and hasattr(model.dce, 'configure_tau'):
        model.dce.configure_tau(
            tau_init=getattr(cfg, 'DCE_TAU_INIT', 1.0),
            learnable=getattr(cfg, 'DCE_TAU_LEARNABLE', True),
            tau_min=getattr(cfg, 'DCE_TAU_MIN', 0.05),
            tau_max=getattr(cfg, 'DCE_TAU_MAX', 5.0))
    return model


# ============================ 五折交叉验证（患者级） ============================

def run_5fold_cv(cfg=None, fold_callback=None):
    """
    执行患者级五折交叉验证（StratifiedGroupKFold）。
    cfg 为 None 时进入普通模式，否则使用传入配置（Ray 模式）。
    fold_callback: 可选回调，每折评估完成后调用 fold_callback(fold_idx_1based, fold_result)，
        用于 Ray trial 的提前剪枝（回调内抛出异常即可终止本次五折）。
    返回 (fold_results, best_fold_idx, num_classes)。
    """
    if cfg is None:
        cfg = Config
        cfg.SAVE_DIR = os.path.join(cfg.SAVE_DIR, cfg.DATASET, cfg.MODEL_NAME, f'method_{cfg.METHODS}')
        os.makedirs(cfg.SAVE_DIR, exist_ok=True)
    else:
        os.makedirs(cfg.SAVE_DIR, exist_ok=True)

    device = torch.device(cfg.DEVICE if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    print(f"使用模型: {cfg.MODEL_NAME}")
    print(f"损失模式: METHODS={cfg.METHODS}, L1={cfg.L1}, L2={cfg.L2}, "
          f"TEMPERATURE={cfg.TEMPERATURE}, ALPHA={cfg.ALPHA}")
    print(f"InfoNCE 开关: COSINE_SHIFT={getattr(cfg, 'COSINE_SHIFT', True)}, "
          f"CLASS_BALANCED_INFONCE={getattr(cfg, 'CLASS_BALANCED_INFONCE', True)}")
    print(f"多原型模式: {getattr(cfg, 'USE_MULTI_PROTOTYPE', True)}")
    print(f"结果保存路径: {os.path.abspath(cfg.SAVE_DIR)}")

    # 加载数据（canonical: 返回患者 id 作为 groups）
    # DATA_PATH 为预处理完成的 canonical 文件（col0=label, col1=patient_id,
    # col2+=spectra），显式声明 schema
    X, y, groups = load_data(cfg.DATA_PATH, has_patient_id=True)
    if y.min() != 0:
        y = y - y.min()
    num_classes = int(y.max().item()) + 1 if cfg.NUM_CLASSES is None else cfg.NUM_CLASSES
    seq_length = X.shape[2]
    group_codes_all, unique_groups = factorize_groups(groups)
    n_patients = len(unique_groups)
    print(f"类别数: {num_classes}, 总样本: {len(y)}, 序列长度: {seq_length}, "
          f"患者/组数: {n_patients}")

    if n_patients < 5:
        raise ValueError(f"患者/组数 ({n_patients}) 少于 5，无法进行患者级五折划分。")

    # 随机种子
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

        # ---- 硬检查：患者级无重叠（显式 raise，python -O 下同样生效）----
        train_pids = set(groups_np[train_idx].tolist())
        val_pids = set(groups_np[val_idx].tolist())
        overlap = train_pids & val_pids
        if overlap:
            raise RuntimeError(
                f"Fold {fold + 1}: 患者泄漏！train/val 共有患者: "
                f"{sorted(overlap)[:10]}")
        print(f"  患者独立性检查通过: train 患者 {len(train_pids)} 个, "
              f"val 患者 {len(val_pids)} 个, 重叠 0")

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

        # 原型初始化（多原型 或 单原型均值）；METHODS=-1 基线不需要原型
        if cfg.METHODS == -1:
            proto_counts_list = [0] * num_classes
            print("  METHODS=-1 基线：跳过原型初始化，仅使用普通分类头。")
        else:
            # FINCH 预热：先用分类头 CE 训练 backbone，使 f_theta 特征可判别，
            # 再在训练过的特征上聚类（论文公式(1) 的前提是已训练的 CNN）
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
            print(f"  原型已初始化: 总原型数={prototypes_data.shape[0]}, "
                  f"每类原型数={proto_counts_list}")

        # 优化器
        optimizer = torch.optim.Adam(model.parameters(),
                                     lr=cfg.LR, weight_decay=cfg.WEIGHT_DECAY)
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1,
                                      patience=max(1, cfg.PATIENCE // 2))

        # 早停
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

        # 最终评估（加载该折最优权重）
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

        print(f"\nFold {fold+1} Final Results:")
        print(format_metrics_report(val_metrics, title=f"Fold {fold+1} Validation",
                                    num_classes=num_classes))
        print(f"  每类原型数: {proto_counts_list}")
        cur_tau = (model.dce.get_tau() if hasattr(model, 'dce')
                   and hasattr(model.dce, 'get_tau') else None)
        if cur_tau is not None:
            print(f"  DCE tau_dce（训练后）: {cur_tau:.4f}")

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

        # Ray trial 剪枝回调：fold 1 后可抛出异常提前终止本 trial
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
            # 最优折验证集原始预测（该折最终评估的 val_out）
            best_val_preds = (np.asarray(val_out['labels']),
                              np.asarray(val_out['preds']),
                              np.asarray(val_out['probs']),
                              np.asarray(groups_np[val_idx]).astype(str))

    # 保存全局最优
    print(f"\n{'='*60}")
    print(f"全局最优折: Fold {best_fold_idx} (UAR={best_uar:.4f})")
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
    print(f"最优模型已保存: {model_path}")

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
    print(f"最优折验证集预测已保存: best_fold_val_predictions.npz "
          f"(labels/preds/probs/groups)")

    if best_centers_state is not None:
        np.savez(os.path.join(cfg.SAVE_DIR, 'best_fold_centers.npz'),
                 centers=best_centers_state.numpy(),
                 prototype_labels=best_proto_labels_state.numpy(),
                 prototype_counts=best_proto_counts_state.numpy(),
                 fold=best_fold_idx)
        print(f"原型中心已保存（含 prototype_labels / prototype_counts）")

    return fold_results, best_fold_idx, num_classes


# ============================ 结果汇总与测试 ============================

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

    print(f"\n交叉验证摘要已保存: {results_path}")


def test_best_model(num_classes, cfg=None):
    """
    对独立测试集评估一次（仅在最终确定超参数后调用；
    Ray trial 内部禁止调用本函数）。
    """
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
        print("未提供测试集路径或文件不存在，跳过最终测试。")
        return None

    device = torch.device(cfg.DEVICE if torch.cuda.is_available() else 'cpu')

    # TEST_DATA_PATH 为预处理完成的 canonical 文件
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

    # 【关键修复】多原型：先根据 checkpoint 中的原型信息初始化 dce，再加载 state_dict
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

    # 保存测试集原始预测，便于事后重算指标/绘制ROC与混淆矩阵而无需重新推理
    pred_path = os.path.join(cfg.SAVE_DIR, 'test_predictions.npz')
    np.savez(pred_path,
             labels=np.asarray(test_out['labels']),
             preds=np.asarray(test_out['preds']),
             probs=np.asarray(test_out['probs']),
             groups=np.asarray(groups_test).astype(str))
    print(f"测试集预测已保存: {pred_path} (labels/preds/probs/groups)")

    print(f"\n{report}")
    print(f"测试集结果已追加到: {results_path}")
    return test_metrics


# ============================ Ray Tune 集成 ============================

def create_cfg_from_search_config(search_config):
    """将 Config 类属性与搜索配置合并，生成独立的配置对象"""
    from types import SimpleNamespace
    cfg = SimpleNamespace()
    for attr in [a for a in dir(Config) if not a.startswith('_') and not callable(getattr(Config, a))]:
        setattr(cfg, attr, getattr(Config, attr))
    for key, value in search_config.items():
        setattr(cfg, key, value)
    return cfg


def warmup_backbone(model, train_loader, device, cfg, warmup_epochs):
    """
    FINCH 预热（修复"在随机初始化特征上聚类"的问题，与论文公式(1)一致）：
    先用普通分类头 CE 训练 backbone 若干 epoch，使 f_theta 提取到
    有类别判别性的特征，之后 initialize_prototypes 在训练过的特征上
    运行 FINCH，亚类划分才有意义。
    注意: 预热只训练 backbone；原型尚未初始化（dce 未 set_prototypes），
    预热结束后才创建主训练优化器（包含原型参数）。
    """
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
            loss = F.nll_loss(log_probs, labels)   # 仅用普通分类头预热
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
    """fold 1 验证 UAR 低于阈值时抛出，用于提前终止 Ray trial。
    trial 以 ERROR 结束 -> OptunaSearch 将其标记为 FAIL，
    TPE 采样器只用 COMPLETE 状态的 trial 拟合代理模型，
    因此被剪枝 trial 的超参数不会影响后续 trial 的采样。"""
    pass


def ray_trainable(search_config):
    """
    Ray Tune 的 trainable：每个 trial 运行一次五折 CV。
    【重要】超参搜索目标由 Config.RAY_SEARCH_OBJECTIVE 决定（默认 best_fold_val_uar，
    即五折中验证 UAR 最高的一折）；
    trial 内的测试集评估仅用于人工观察过拟合/欠拟合（临时代码，标注"后续删除"），
    不参与超参选择。
    【剪枝】Config.RAY_PRUNE_FOLD1=True 时，fold 1 验证 UAR 低于
    Config.RAY_MIN_FOLD1_UAR 立即终止本 trial（抛出 Fold1Pruned），
    被终止的 trial 不会将其超参数带入后续搜索。
    """
    import importlib
    from ray import tune

    _train_module = importlib.import_module('train_5fold_dce')

    cfg = _train_module.create_cfg_from_search_config(search_config)

    try:
        trial_dir = tune.get_trial_dir()
    except AttributeError:
        trial_dir = tune.get_context().get_trial_dir()

    cfg.SAVE_DIR = trial_dir

    print(f"\n[Trial] 开始搜索配置: {search_config}")
    print(f"[Trial] 保存路径: {cfg.SAVE_DIR}")

    # fold 1 剪枝回调：fold 1 验证 UAR 低于阈值时抛出 Fold1Pruned，
    # 此时不会执行测试集评估，直接终止本 trial 进入下一个
    prune_enable = getattr(cfg, 'RAY_PRUNE_FOLD1', False)
    min_fold1_uar = getattr(cfg, 'RAY_MIN_FOLD1_UAR', 0.60)

    def _fold1_guard(fold_idx, fold_result):
        if prune_enable and fold_idx == 1 and fold_result['uar'] < min_fold1_uar:
            raise _train_module.Fold1Pruned(
                f"fold 1 val UAR={fold_result['uar']:.4f} < {min_fold1_uar:.2f}，"
                f"提前终止本 trial（超参数: {search_config}）")

    fold_callback = _fold1_guard if prune_enable else None
    fold_results, best_fold_idx, num_classes = _train_module.run_5fold_cv(
        cfg, fold_callback=fold_callback)
    _train_module.save_cv_summary(fold_results, best_fold_idx, cfg)

    mean_val_uar = float(np.mean([r['uar'] for r in fold_results]))
    # 五折中验证 UAR 最高的一折（与"最终模型取 best fold"的选择协议一致）
    best_fold_val_uar = float(max(r['uar'] for r in fold_results))

    # 后续删除
    # ===== 新增：trial 内评估测试集（仅用于观察过拟合/欠拟合，不参与超参选择）=====
    test_metrics = _train_module.test_best_model(num_classes, cfg)
    if test_metrics is not None:
        test_uar = test_metrics['spectrum']['uar']
        test_uf1 = test_metrics['spectrum']['uf1']
        test_acc = test_metrics['spectrum']['accuracy']
    else:
        test_uar = test_uf1 = test_acc = float('nan')
    # ===================================

    return {
        "mean_val_uar": mean_val_uar,          # 五折平均（记录用，供事后核对稳健性）
        "best_fold_val_uar": best_fold_val_uar,  # 最好折（Config.RAY_SEARCH_OBJECTIVE 可选为搜索目标）
        "dce_tau_final": float(fold_results[-1].get('dce_tau') or 0.0),  # 最后一折训练后的 tau_dce
        "test_uar": test_uar,
        "test_uf1": test_uf1,
        "test_acc": test_acc,
    }


def run_ray_tune():
    """使用 Ray Tune + Optuna(TPESampler) 搜索超参数。
    搜索目标由 Config.RAY_SEARCH_OBJECTIVE 决定:
      'best_fold' -> best_fold_val_uar（五折中验证 UAR 最高的一折，
                     与"最终模型取 best fold"的选择协议一致，默认）
      'mean'      -> mean_val_uar（五折平均验证 UAR）
    两种指标在每个 trial 中都会被记录，仅是搜索目标不同。"""
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

    # 搜索空间：连续模式（loguniform/uniform，可能得到如 1.232e-4 的任意精度值，
    # 属正常现象，报告时保留 3–4 位有效数字即可）；
    # 离散模式（Config.RAY_SPACE_MODE='discrete'）从预定义的整数值网格中采样，
    # 超参数更"整齐"，便于报告与复现。
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

    # 可选：把原型特征维度纳入搜索（NUM_HIDDEN_UNITS=6 可能是性能瓶颈，
    # 5 类多原型挤在 6 维空间；建议开启一次搜索确认）
    if getattr(cfg, 'RAY_SEARCH_HIDDEN_UNITS', False):
        search_space["NUM_HIDDEN_UNITS"] = tune.choice([8, 16, 32, 64, 128])

    # 可选：搜索 tau 初值（一般不需要，tau 本身可学习；
    # 仅当 DCE_TAU_LEARNABLE=False、想搜索"固定 tau"时开启）
    if getattr(cfg, 'RAY_SEARCH_DCE_TAU', False):
        search_space["DCE_TAU_INIT"] = tune.choice(
            [0.05, 0.07, 0.1, 0.15, 0.2, 0.3, 0.5, 0.7, 1.0])

    print(f"Ray Tune 搜索空间 (METHODS={cfg.METHODS}):")
    for k, v in search_space.items():
        print(f"  {k}: {v}")
    # 搜索目标：'best_fold' = 五折中验证 UAR 最高的一折；'mean' = 五折平均
    objective = getattr(cfg, 'RAY_SEARCH_OBJECTIVE', 'best_fold')
    metric_name = {'best_fold': 'best_fold_val_uar', 'mean': 'mean_val_uar'}[objective]
    print(f"优化指标: {metric_name} "
          f"({'五折中验证 UAR 最高的一折' if objective == 'best_fold' else '五折平均验证 UAR'})")

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
        print("警告：所有 trial 均失败，未找到最佳结果！")
        print(f"{'='*60}")
        return analysis

    # 导出全部 trial 的汇总表（配置 + 搜索目标指标 + 其他记录指标）
    try:
        trials_csv = os.path.join(cfg.SAVE_DIR, 'ray_trials_summary.csv')
        df = analysis.dataframe(metric=metric_name, mode="max")
        df.to_csv(trials_csv, index=False)
        print(f"全部 trial 汇总已保存: {trials_csv} ({len(df)} 行)")
    except Exception as e:
        print(f"trial 汇总表导出失败（不影响主流程）: {e}")

    best_trial = max(completed_trials,
                     key=lambda t: t.last_result.get(metric_name, -1))
    best_config = best_trial.config
    best_search_uar = best_trial.last_result.get(metric_name, -1)

    # ============ 搜索结束后：用最优配置重跑五折，只测一次测试集 ============
    summary_dir = os.path.join(
        cfg.SAVE_DIR, cfg.DATASET, cfg.MODEL_NAME, f'method_{cfg.METHODS}'
    )
    os.makedirs(summary_dir, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"搜索完成，使用最优配置重跑五折并仅评估一次测试集。")
    print(f"最优配置: {best_config}")
    print(f"{'='*60}")

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
        if test_metrics is not None:
            f.write(f"Final test UAR (single evaluation): "
                    f"{test_metrics['spectrum']['uar']:.4f}\n")
            f.write(f"Final test UF1 (single evaluation): "
                    f"{test_metrics['spectrum']['uf1']:.4f}\n")
        f.write("\nOptimal Hyperparameters:\n")
        for k, v in best_config.items():
            f.write(f"  {k}: {v}\n")

    print(f"\nRay Tune 全流程完成！汇总文件: {os.path.abspath(summary_path)}")
    return analysis


# ============================ 入口 ============================

if __name__ == '__main__':
    if Config.USE_RAY:
        run_ray_tune()
    else:
        fold_results, best_fold_idx, num_classes = run_5fold_cv()
        save_cv_summary(fold_results, best_fold_idx, Config)
        test_best_model(num_classes)
