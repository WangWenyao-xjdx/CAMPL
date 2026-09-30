import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import os

current_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, current_dir)

from core.config import Config


class dce_loss(nn.Module):
   
    def __init__(self, n_classes, feat_dim, tau_init=1.0, tau_learnable=True,
                 tau_min=0.05, tau_max=5.0):
        super(dce_loss, self).__init__()
        self.n_classes = n_classes
        self.feat_dim = feat_dim
        self.prototypes = nn.Parameter(torch.empty(0, feat_dim))
        self.register_buffer('prototype_labels', torch.empty(0, dtype=torch.long))
        self.register_buffer('prototype_counts', torch.empty(0, dtype=torch.long))
        self.num_prototypes = 0
        self.tau_min = float(tau_min)
        self.tau_max = float(tau_max)
        self.log_tau = nn.Parameter(torch.log(torch.tensor(float(tau_init))))
        self.log_tau.requires_grad_(bool(tau_learnable))

    def configure_tau(self, tau_init=None, learnable=None, tau_min=None, tau_max=None):
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

        tau = self.log_tau.exp().clamp(self.tau_min, self.tau_max)
        class_sim = class_sim / tau

        return self.prototypes, class_sim


class CNN4(nn.Module):
    def __init__(self, in_channels, num_classes, num_hidden_units=256, input_length=None):
        super(CNN4, self).__init__()

        self.conv1 = nn.Conv1d(in_channels, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(32)
        self.pool1 = nn.MaxPool1d(2)

        self.conv2 = nn.Conv1d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(64)
        self.pool2 = nn.MaxPool1d(2)

        self.conv3 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(128)
        self.pool3 = nn.MaxPool1d(2)

        self.conv4 = nn.Conv1d(128, 256, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm1d(256)
        self.pool4 = nn.MaxPool1d(2)

        if input_length is None:
            raise ValueError("CNN4 needs input_length")
        self._calculate_fc_input(input_length)

        self.fc_p = nn.Linear(self.fc_input_size, num_hidden_units)
        self.dce = dce_loss(n_classes=num_classes, feat_dim=num_hidden_units)
        self.fc = nn.Linear(self.fc_input_size, num_classes)

    def _calculate_fc_input(self, input_length):
        l = input_length
        for _ in range(4):
            l = l // 2
        self.fc_input_size = 256 * l

    def forward(self, x):
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        x = self.pool3(F.relu(self.bn3(self.conv3(x))))
        x = self.pool4(F.relu(self.bn4(self.conv4(x))))

        x_flatten = x.view(x.size(0), -1)

        features_main = self.fc(x_flatten)
        output = F.log_softmax(features_main, dim=1)

        features_p = self.fc_p(x_flatten)
        prototypes, class_sim = self.dce(features_p)
        output_p = F.log_softmax(class_sim, dim=1)

        return output, features_p, prototypes, class_sim, output_p