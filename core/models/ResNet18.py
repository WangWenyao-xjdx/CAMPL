"""
ResNet18模型（一维版本）— 多原型
"""

import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

current_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, current_dir)

# from config import Config


class dce_loss(nn.Module):
    """
    多原型 Distance-based Cross-Entropy Loss (DCE)

    【可学习温度 tau_dce】原始实现把余弦相似度（范围 [-1,1]）直接送入
    log_softmax，类别间 logit 差距很小、决策余量不足。现引入温度参数：
        logits = class_sim / tau_dce
    tau_dce = exp(log_tau)，以 log 参数化保证恒为正，并 clamp 到
    [tau_min, tau_max] 防止数值不稳定。tau_init=1.0 时与旧行为完全一致。
    可通过 configure_tau() 由 Config 配置（DCE_TAU_INIT / DCE_TAU_LEARNABLE /
    DCE_TAU_MIN / DCE_TAU_MAX）。
    """
    def __init__(self, n_classes, feat_dim, tau_init=1.0, tau_learnable=True,
                 tau_min=0.05, tau_max=5.0):
        super(dce_loss, self).__init__()
        self.n_classes = n_classes
        self.feat_dim = feat_dim
        self.prototypes = nn.Parameter(torch.empty(0, feat_dim))
        self.register_buffer('prototype_labels', torch.empty(0, dtype=torch.long))
        self.register_buffer('prototype_counts', torch.empty(0, dtype=torch.long))
        self.num_prototypes = 0
        # 温度（log 参数化；requires_grad=False 时等价于固定 tau）
        self.tau_min = float(tau_min)
        self.tau_max = float(tau_max)
        self.log_tau = nn.Parameter(torch.log(torch.tensor(float(tau_init))))
        self.log_tau.requires_grad_(bool(tau_learnable))

    def configure_tau(self, tau_init=None, learnable=None, tau_min=None, tau_max=None):
        """由外部配置温度；就地修改（不替换 Parameter 对象），需在创建优化器之前调用。"""
        if tau_min is not None:
            self.tau_min = float(tau_min)
        if tau_max is not None:
            self.tau_max = float(tau_max)
        if tau_init is not None:
            with torch.no_grad():
                self.log_tau.copy_(torch.log(
                    torch.tensor(float(tau_init), device=self.log_tau.device)))
        if learnable is not None:
            self.log_tau.requires_grad_(bool(learnable))

    def get_tau(self):
        with torch.no_grad():
            return float(self.log_tau.exp().clamp(self.tau_min, self.tau_max).item())

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        # 向后兼容：旧版 checkpoint 没有 log_tau 键，注入当前默认值（tau=1.0）
        key = prefix + 'log_tau'
        if key not in state_dict:
            state_dict[key] = self.log_tau.detach().clone()
        super()._load_from_state_dict(state_dict, prefix, local_metadata, strict,
                                      missing_keys, unexpected_keys, error_msgs)

    def set_prototypes(self, prototypes, labels, counts):
        self.num_prototypes = prototypes.shape[0]
        self.prototypes = nn.Parameter(prototypes.clone())
        self.prototype_labels = labels.clone()
        self.prototype_counts = counts.clone()
        self.n_classes = len(counts)

    def forward(self, x):
        # 【修复】未初始化时返回占位符，允许模型前向传播提取 features_p
        if self.num_prototypes == 0:
            batch_size = x.shape[0]
            return self.prototypes, torch.zeros(batch_size, self.n_classes, device=x.device)

        x_norm = F.normalize(x, p=2, dim=1)
        proto_norm = F.normalize(self.prototypes, p=2, dim=1)
        cos_sim = torch.matmul(x_norm, proto_norm.t())

        batch_size = x.shape[0]
        mask = torch.zeros(self.n_classes, self.num_prototypes, device=x.device, dtype=x.dtype)
        valid_idx = torch.arange(self.num_prototypes, device=x.device)
        mask[self.prototype_labels, valid_idx] = 1.0

        sim_expanded = cos_sim.unsqueeze(1).expand(batch_size, self.n_classes, self.num_prototypes)
        mask_expanded = mask.unsqueeze(0).expand(batch_size, -1, -1)

        sim_masked = sim_expanded.clone()
        sim_masked[mask_expanded == 0] = -float('inf')
        class_sim, _ = sim_masked.max(dim=2)

        # 可学习温度缩放：logits = class_sim / tau_dce
        tau = self.log_tau.exp().clamp(self.tau_min, self.tau_max)
        class_sim = class_sim / tau

        return self.prototypes, class_sim


class basic_block(nn.Module):
    def __init__(self, in_channels):
        super(basic_block, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm1d(in_channels)
        self.conv2 = nn.Conv1d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm1d(in_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += identity
        out = self.relu(out)
        return out


class basic_block2(nn.Module):
    def __init__(self, in_channels, out_channels, stride=2):
        super(basic_block2, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.shortcut = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride),
            nn.BatchNorm1d(out_channels)
        )

    def forward(self, x):
        identity = self.shortcut(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += identity
        out = self.relu(out)
        return out


class ResNet18_1D(nn.Module):
    def __init__(self, in_channels, num_classes, num_hidden_units=2, input_length=None):
        super(ResNet18_1D, self).__init__()

        self.conv1 = nn.Conv1d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm1d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

        self.layer1 = nn.Sequential(basic_block(64), basic_block(64))
        self.layer2 = nn.Sequential(basic_block2(64, 128, stride=2), basic_block(128))
        self.layer3 = nn.Sequential(basic_block2(128, 256, stride=2), basic_block(256))
        self.layer4 = nn.Sequential(basic_block2(256, 512, stride=2), basic_block(512))

        self.avgpool = nn.AdaptiveAvgPool1d(1)

        self.fc_p = nn.Linear(512, num_hidden_units)
        self.dce = dce_loss(num_classes, num_hidden_units)
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x_flatten = torch.flatten(x, 1)

        out_linear = self.fc(x_flatten)
        output = F.log_softmax(out_linear, dim=1)

        features_p = self.fc_p(x_flatten)
        prototypes, class_sim = self.dce(features_p)
        output_p = F.log_softmax(class_sim, dim=1)

        return output, features_p, prototypes, class_sim, output_p