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
        # 温度（log 参数化；requires_grad=False 时等价于固定 tau）
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


class BasicConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super(BasicConv1d, self).__init__()
        self.conv = nn.Conv1d(in_channels, out_channels,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, bias=False)
        self.bn = nn.BatchNorm1d(out_channels, eps=0.001)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return F.relu(x, inplace=True)


class Inception1D(nn.Module):
    def __init__(self, in_channels, ch1x1, ch3x3red, ch3x3, ch5x5red, ch5x5, pool_proj):
        super(Inception1D, self).__init__()
        self.branch1 = BasicConv1d(in_channels, ch1x1, kernel_size=1)
        self.branch2 = nn.Sequential(
            BasicConv1d(in_channels, ch3x3red, kernel_size=1),
            BasicConv1d(ch3x3red, ch3x3, kernel_size=3, padding=1)
        )
        self.branch3 = nn.Sequential(
            BasicConv1d(in_channels, ch5x5red, kernel_size=1),
            BasicConv1d(ch5x5red, ch5x5, kernel_size=5, padding=2)
        )
        self.branch4 = nn.Sequential(
            nn.MaxPool1d(kernel_size=3, stride=1, padding=1),
            BasicConv1d(in_channels, pool_proj, kernel_size=1)
        )

    def forward(self, x):
        outputs = [self.branch1(x), self.branch2(x), self.branch3(x), self.branch4(x)]
        return torch.cat(outputs, 1)


class InceptionAux1D(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(InceptionAux1D, self).__init__()
        self.averagePool = nn.AvgPool1d(kernel_size=5, stride=3)
        self.conv = BasicConv1d(in_channels, 128, kernel_size=1)
        self.fc1 = nn.Linear(128 * 4, 1024)
        self.fc2 = nn.Linear(1024, num_classes)

    def forward(self, x):
        x = self.averagePool(x)
        x = self.conv(x)
        x = torch.flatten(x, 1)
        x = F.dropout(x, 0.5, training=self.training)
        x = F.relu(self.fc1(x), inplace=True)
        x = F.dropout(x, 0.5, training=self.training)
        x = self.fc2(x)
        return x


class GoogleNet1D(nn.Module):
    def __init__(self, in_channels, num_classes, num_hidden_units=2,
                 input_length=None, aux_logits=True):
        super(GoogleNet1D, self).__init__()
        self.aux_logits = aux_logits

        self.conv1 = BasicConv1d(in_channels, 64, kernel_size=7, stride=2, padding=3)
        self.maxpool1 = nn.MaxPool1d(3, stride=2, padding=1, ceil_mode=True)
        self.conv2 = BasicConv1d(64, 64, kernel_size=1)
        self.conv3 = BasicConv1d(64, 192, kernel_size=3, padding=1)
        self.maxpool2 = nn.MaxPool1d(3, stride=2, padding=1, ceil_mode=True)

        self.inception3a = Inception1D(192, 64, 96, 128, 16, 32, 32)
        self.inception3b = Inception1D(256, 128, 128, 192, 32, 96, 64)
        self.maxpool3 = nn.MaxPool1d(3, stride=2, padding=1, ceil_mode=True)

        self.inception4a = Inception1D(480, 192, 96, 208, 16, 48, 64)
        self.inception4b = Inception1D(512, 160, 112, 224, 24, 64, 64)
        self.inception4c = Inception1D(512, 128, 128, 256, 24, 64, 64)
        self.inception4d = Inception1D(512, 112, 144, 288, 32, 64, 64)
        self.inception4e = Inception1D(528, 256, 160, 320, 32, 128, 128)
        self.maxpool4 = nn.MaxPool1d(3, stride=2, padding=1, ceil_mode=True)

        self.inception5a = Inception1D(832, 256, 160, 320, 32, 128, 128)
        self.inception5b = Inception1D(832, 384, 192, 384, 48, 128, 128)

        if self.aux_logits:
            self.aux1 = InceptionAux1D(512, num_classes)
            self.aux2 = InceptionAux1D(528, num_classes)

        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.dropout = nn.Dropout(0.4)

        self.fc_p = nn.Linear(1024, num_hidden_units)
        self.dce = dce_loss(num_classes, num_hidden_units)
        self.fc = nn.Linear(1024, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.maxpool2(x)
        x = self.inception3a(x)
        x = self.inception3b(x)
        x = self.maxpool3(x)
        x = self.inception4a(x)
        if self.training and self.aux_logits:
            aux1 = self.aux1(x)
        x = self.inception4b(x)
        x = self.inception4c(x)
        x = self.inception4d(x)
        if self.training and self.aux_logits:
            aux2 = self.aux2(x)
        x = self.inception4e(x)
        x = self.maxpool4(x)
        x = self.inception5a(x)
        x = self.inception5b(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)

        out_linear = self.fc(x)
        output = F.log_softmax(out_linear, dim=1)

        features_p = self.fc_p(x)
        prototypes, class_sim = self.dce(features_p)
        output_p = F.log_softmax(class_sim, dim=1)

        if self.training and self.aux_logits:
            return output, features_p, prototypes, class_sim, output_p, aux2, aux1
        return output, features_p, prototypes, class_sim, output_p

    def get_features(self, x):
        x = self.conv1(x)
        x = self.maxpool1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.maxpool2(x)
        x = self.inception3a(x)
        x = self.inception3b(x)
        x = self.maxpool3(x)
        x = self.inception4a(x)
        x = self.inception4b(x)
        x = self.inception4c(x)
        x = self.inception4d(x)
        x = self.inception4e(x)
        x = self.maxpool4(x)
        x = self.inception5a(x)
        x = self.inception5b(x)
        x = self.avgpool(x)
        return torch.flatten(x, 1)