import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class MIProjection(nn.Module):
    def __init__(self, inp_dim: int, proj_dim: int, bln: bool = True):
        super().__init__()

        self.inp_dim = inp_dim
        self.proj_dim = proj_dim
        self.feat_nonlinear = nn.Sequential(
            nn.Linear(in_features=inp_dim, out_features=proj_dim),
            nn.BatchNorm1d(num_features=proj_dim),
            nn.ReLU(),
            nn.Linear(in_features=proj_dim, out_features=proj_dim),
        )

        self.feat_shortcut = nn.Linear(in_features=inp_dim, out_features=proj_dim)
        self.feature_block_ln = nn.LayerNorm(proj_dim)

        self.bln = bln
        self.reset_parameters()

    def reset_parameters(self):
        eye_mask = torch.zeros(self.proj_dim, self.inp_dim).bool()

        # 🔧 FIX 1: Calculate safe iteration range
        min_dim = min(self.proj_dim, self.inp_dim)

        # 🔧 FIX 2: Add explicit bounds checking
        for i in range(min_dim):
            # 🔧 FIX 3: Double-check bounds before assignment
            if i < eye_mask.shape[0] and i < eye_mask.shape[1]:
                eye_mask[i, i] = 1
            else:
                # 🔧 FIX 4: Safe fallback if something is still wrong
                print(f"Warning: Skipping index {i} - out of bounds for shape {eye_mask.shape}")
                break

        self.feat_shortcut.weight.data.uniform_(-0.01, 0.01)

        # 🔧 FIX 5: Safe shape checking before applying mask
        if eye_mask.shape[0] <= self.feat_shortcut.weight.data.shape[0] and \
                eye_mask.shape[1] <= self.feat_shortcut.weight.data.shape[1]:
            self.feat_shortcut.weight.data.masked_fill_(eye_mask, 1.0)
        else:
            # 🔧 FIX 6: Graceful fallback - skip identity init if shapes don't match
            pass

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.feat_nonlinear(x) + self.feat_shortcut(x)
        if self.bln:
            h = self.feature_block_ln(h)

        return h


class PriorDiscriminator(nn.Module):
    def __init__(self, inp_dim: int):
        super().__init__()

        self.l0 = nn.Linear(in_features=inp_dim, out_features=1000)
        self.l1 = nn.Linear(in_features=1000, out_features=200)
        self.l2 = nn.Linear(in_features=200, out_features=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.l0(x))
        out = F.relu(self.l1(out))
        return torch.sigmoid(self.l2(out))
