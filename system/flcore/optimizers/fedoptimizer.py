# PFLlib: Personalized Federated Learning Algorithm Library
# Copyright (C) 2021  Jianqing Zhang

# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
 
# You should have received a copy of the GNU General Public License along
# with this program; if not, write to the Free Software Foundation, Inc.,
# 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.

import torch
from torch.optim import Optimizer
from system.utils.privacy import LaplacianSmoothing


class SCAFFOLDOptimizer(Optimizer):
    def __init__(self, params, lr):
        defaults = dict(lr=lr)
        super(SCAFFOLDOptimizer, self).__init__(params, defaults)

    def step(self, privacy, dp_sigma, laplacian, ls_sigma, server_cs, client_cs, device):
        for group in self.param_groups:
            for p, sc, cc in zip(group['params'], server_cs, client_cs):
                if p.grad is None:
                    continue
                if privacy:
                    noise = torch.normal(mean=0, std=dp_sigma, size=p.data.shape).to(device)
                    torch.nn.utils.clip_grad_norm_(parameters=p.grad.data, max_norm=5.0, norm_type=2)
                    if laplacian:
                        p.data.add_(other=LaplacianSmoothing(p.grad.data + sc - cc + noise, ls_sigma, device), alpha=-group['lr'])
                    else:
                        p.data.add_(other=(p.grad.data + sc - cc + noise), alpha=-group['lr'])
                else:
                    p.data.add_(other=(p.grad.data + sc - cc), alpha=-group['lr'])

class FedLMOptimizer(Optimizer):
    def __init__(self, params, lr):
        defaults = dict(lr=lr)
        super(FedLMOptimizer, self).__init__(params, defaults)

    def step(self, privacy, dp_sigma, laplacian, ls_sigma, server_cs, client_cs, device):
        for group in self.param_groups:
            for p, sc, cc in zip(group['params'], server_cs, client_cs):
                if p.grad is None:
                    continue
                if privacy:
                    noise = torch.normal(mean=0, std=dp_sigma, size=p.data.shape).to(device)
                    torch.nn.utils.clip_grad_norm_(parameters=p.grad.data, max_norm=5.0, norm_type=2)
                    if laplacian:
                        p.data.add_(other=LaplacianSmoothing(p.grad.data + 0.9*(sc-cc) + noise, ls_sigma, device), alpha=-group['lr'])
                    else:
                        p.data.add_(other=(p.grad.data + 0.9*sc  + noise), alpha=-group['lr'])
                else:
                    p.data.add_(other=(p.grad.data +0.9*sc), alpha=-group['lr'])