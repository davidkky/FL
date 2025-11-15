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
import torch.fft
import numpy as np
from opacus import PrivacyEngine

MAX_GRAD_NORM = 1.0
DELTA = 1e-5


def initialize_dp(model, optimizer, data_loader, dp_sigma):
    privacy_engine = PrivacyEngine(accountant="rdp")
    model, optimizer, data_loader = privacy_engine.make_private(
        module=model,
        optimizer=optimizer,
        data_loader=data_loader,
        noise_multiplier=dp_sigma,
        max_grad_norm=MAX_GRAD_NORM,
    )
    return model, optimizer, data_loader, privacy_engine


def get_dp_params(privacy_engine):
    return privacy_engine.get_epsilon(delta=DELTA), DELTA


# Renyi-DP
def gaussian_mech_RDP_vec(data, sensitivity, alpha, epsilon, batch_size):
    sigma = np.sqrt((sensitivity**2 * alpha) / (2 * epsilon * batch_size))
    rdp_noise = torch.normal(0, sigma, data.shape)
    return rdp_noise


def LaplacianSmoothing(data, sigma, device):
    """ d = ifft(fft(g)/(1-sigma*fft(v))) """
    size = torch.numel(data)
    c = np.zeros(shape=(1, size))
    c[0, 0] = -2.
    c[0, 1] = 1.
    c[0, -1] = 1.
    c = torch.Tensor(c).to(device)
    c_fft = torch.view_as_real(torch.fft.fft(c))
    coeff = 1./(1.-sigma*c_fft[..., 0])
    tmp = data.view(-1, size).to(device)
    ft_tmp = torch.fft.fft(tmp)
    ft_tmp = torch.view_as_real(ft_tmp)
    tmp = torch.zeros_like(ft_tmp)
    tmp[..., 0] = ft_tmp[..., 0] * coeff
    tmp[..., 1] = ft_tmp[..., 1] * coeff
    tmp = torch.view_as_complex(tmp)
    tmp = torch.fft.ifft(tmp)
    tmp = tmp.view(data.size())
    return tmp.real