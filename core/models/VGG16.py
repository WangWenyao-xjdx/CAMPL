import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import os

current_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, current_dir)


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


class VGG16_1D(nn.Module):
    def __init__(self, in_channels, num_classes, num_hidden_units=2, input_length=None):
        super(VGG16_1D, self).__init__()

        if input_length is None:
            raise ValueError("VGG16_1D 需要提供 input_length 以计算 FC 维度")

        self.conv1_1 = nn.Conv1d(in_channels, 64, kernel_size=3, padding=1)
        self.bn1_1 = nn.BatchNorm1d(64)
        self.conv1_2 = nn.Conv1d(64, 64, kernel_size=3, padding=1)
        self.bn1_2 = nn.BatchNorm1d(64)
        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2)

        self.conv2_1 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.bn2_1 = nn.BatchNorm1d(128)
        self.conv2_2 = nn.Conv1d(128, 128, kernel_size=3, padding=1)
        self.bn2_2 = nn.BatchNorm1d(128)
        self.pool2 = nn.MaxPool1d(kernel_size=2, stride=2)

        self.conv3_1 = nn.Conv1d(128, 256, kernel_size=3, padding=1)
        self.bn3_1 = nn.BatchNorm1d(256)
        self.conv3_2 = nn.Conv1d(256, 256, kernel_size=3, padding=1)
        self.bn3_2 = nn.BatchNorm1d(256)
        self.conv3_3 = nn.Conv1d(256, 256, kernel_size=3, padding=1)
        self.bn3_3 = nn.BatchNorm1d(256)
        self.pool3 = nn.MaxPool1d(kernel_size=2, stride=2)

        self.conv4_1 = nn.Conv1d(256, 512, kernel_size=3, padding=1)
        self.bn4_1 = nn.BatchNorm1d(512)
        self.conv4_2 = nn.Conv1d(512, 512, kernel_size=3, padding=1)
        self.bn4_2 = nn.BatchNorm1d(512)
        self.conv4_3 = nn.Conv1d(512, 512, kernel_size=3, padding=1)
        self.bn4_3 = nn.BatchNorm1d(512)
        self.pool4 = nn.MaxPool1d(kernel_size=2, stride=2)

        self.conv5_1 = nn.Conv1d(512, 512, kernel_size=3, padding=1)
        self.bn5_1 = nn.BatchNorm1d(512)
        self.conv5_2 = nn.Conv1d(512, 512, kernel_size=3, padding=1)
        self.bn5_2 = nn.BatchNorm1d(512)
        self.conv5_3 = nn.Conv1d(512, 512, kernel_size=3, padding=1)
        self.bn5_3 = nn.BatchNorm1d(512)
        self.pool5 = nn.MaxPool1d(kernel_size=2, stride=2)

        self._calculate_fc_input(input_length)

        self.fc_p = nn.Linear(self.fc_input_size, num_hidden_units)
        self.dce = dce_loss(num_classes, num_hidden_units)
        self.fc = nn.Linear(self.fc_input_size, num_classes)

    def _calculate_fc_input(self, input_length):
        l = input_length
        for _ in range(5):
            l = l // 2
        self.fc_input_size = 512 * l
        print(f"FC input size: {self.fc_input_size}")

    def forward(self, x):
        x = self.pool1(F.relu(self.bn1_2(self.conv1_2(F.relu(self.bn1_1(self.conv1_1(x)))))))
        x = self.pool2(F.relu(self.bn2_2(self.conv2_2(F.relu(self.bn2_1(self.conv2_1(x)))))))
        x = self.pool3(F.relu(self.bn3_3(self.conv3_3(F.relu(self.bn3_2(self.conv3_2(F.relu(self.bn3_1(self.conv3_1(x))))))))))
        x = self.pool4(F.relu(self.bn4_3(self.conv4_3(F.relu(self.bn4_2(self.conv4_2(F.relu(self.bn4_1(self.conv4_1(x))))))))))
        x = self.pool5(F.relu(self.bn5_3(self.conv5_3(F.relu(self.bn5_2(self.conv5_2(F.relu(self.bn5_1(self.conv5_1(x))))))))))

        x_flatten = x.view(x.size(0), -1)

        out_linear = self.fc(x_flatten)
        output = F.log_softmax(out_linear, dim=1)

        features_p = self.fc_p(x_flatten)
        prototypes, class_sim = self.dce(features_p)
        output_p = F.log_softmax(class_sim, dim=1)

        return output, features_p, prototypes, class_sim, output_p

    def get_features(self, x):
        x = self.pool1(F.relu(self.bn1_2(self.conv1_2(F.relu(self.bn1_1(self.conv1_1(x)))))))
        x = self.pool2(F.relu(self.bn2_2(self.conv2_2(F.relu(self.bn2_1(self.conv2_1(x)))))))
        x = self.pool3(F.relu(self.bn3_3(self.conv3_3(F.relu(self.bn3_2(self.conv3_2(F.relu(self.bn3_1(self.conv3_1(x))))))))))
        x = self.pool4(F.relu(self.bn4_3(self.conv4_3(F.relu(self.bn4_2(self.conv4_2(F.relu(self.bn4_1(self.conv4_1(x))))))))))
        x = self.pool5(F.relu(self.bn5_3(self.conv5_3(F.relu(self.bn5_2(self.conv5_2(F.relu(self.bn5_1(self.conv5_1(x))))))))))
        return x.view(x.size(0), -1)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)