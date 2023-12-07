# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import logging
from typing import Callable, Optional

import torch
from opt_einsum import contract
from torch.optim import Optimizer

from .optimizer import (
    DPOptimizer,
    _check_processed_flag,
    _generate_noise,
    _mark_as_processed,
)


logger = logging.getLogger(__name__)


class DCSGDPOptimizer(DPOptimizer):
    """
    :class:`~opacus.optimizers.optimizer.DPOptimizer` that implements
    adaptive clipping strategy
    https://arxiv.org/pdf/1905.03871.pdf
    """

    def __init__(
        self,
        optimizer: Optimizer,
        *,
        noise_multiplier: float,
        histogram_std: float = 6.0,
        max_grad_norm: float,
        expected_batch_size: Optional[int],
        loss_reduction: str = "mean",
        generator=None,
        secure_mode: bool = False,
        batchsize_train: int = 256,
        dimension: int = 11181642,
        percentile: float = 0.3,
        stride: float = 2.0,
        bin_cnt: int = 20,
    ):
        super().__init__(
            optimizer,
            noise_multiplier=noise_multiplier,
            max_grad_norm=max_grad_norm,
            expected_batch_size=expected_batch_size,
            loss_reduction=loss_reduction,
            generator=generator,
            secure_mode=secure_mode,
        )
        # dimension
        # resnet18 11181642
        # resnet34 21289802
        # self.historgram_std = histogram_std
        self.historgram_std = histogram_std
        self.batchsize_train = batchsize_train
        self.dimension = dimension
        self.percentile = percentile
        self.timer = 0
        self.stride = 1.0
        self.bin_cnt = bin_cnt
        self.noise_multiplier = (
            self.noise_multiplier ** (-2) - (self.historgram_std) ** (-2)
        ) ** (-1 / 2)
        self.sample_size = 0
        self.unclipped_num = 0
        self.norm_stack = []

    def zero_grad(self, set_to_none: bool = False):
        """
        Clear gradients, self.sample_size and self.unclipped_num
        """
        super().zero_grad(set_to_none)

        self.sample_size = 0
        self.unclipped_num = 0

    def clip_and_accumulate(self):
        per_param_norms = [
            g.view(len(g), -1).norm(2, dim=-1) for g in self.grad_samples
        ]
        per_sample_norms = torch.stack(per_param_norms, dim=1).norm(2, dim=1)
        for norm in per_sample_norms:
            self.norm_stack.append(norm)
        # self.update_max_grad_norm()

        per_sample_clip_factor = (self.max_grad_norm / (per_sample_norms + 1e-6)).clamp(
            max=1.0
        )

        for p in self.params:
            _check_processed_flag(p.grad_sample)
            grad_sample = self._get_flat_grad_sample(p)
            grad = contract("i,i...", per_sample_clip_factor, grad_sample)

            if p.summed_grad is not None:
                p.summed_grad += grad
            else:
                p.summed_grad = grad

            _mark_as_processed(p.grad_sample)

    def add_noise(self):
        super().add_noise()


    def update_max_grad_norm(self):
        """
        Update clipping bound based on unclipped fraction
        """
        bin_cnt = self.bin_cnt
        stride = self.max_grad_norm * 2 / bin_cnt
        hist = []
        self.timer += 1
        for i in range(bin_cnt):
            hist.append(0)

        for tmp in self.norm_stack:
            if int(tmp / stride) > bin_cnt - 1:
                hist[bin_cnt - 1] += 1
            else:
                hist[int(tmp / stride)] += 1


        cnt = 0
        flag = 0
        noise_sum = 0
        for i in range(bin_cnt):
            noise = torch.normal(mean=0, std=torch.tensor(self.historgram_std / 1.0))
            hist[i] += noise
            noise_sum += hist[i]
        target = noise_sum * self.percentile
        for i in range(bin_cnt):
            cnt += hist[i]
            if cnt >= target:
                self.max_grad_norm = stride * ((i + 1 + i) / 2)
                flag = 1
                break

        if flag == 0:
            self.max_grad_norm = bin_cnt * stride - (stride / 2)
        self.norm_stack = []

    def pre_step(
        self, closure: Optional[Callable[[], float]] = None
    ) -> Optional[float]:
        pre_step_full = super().pre_step()
        if pre_step_full:
            self.update_max_grad_norm()
        return pre_step_full
