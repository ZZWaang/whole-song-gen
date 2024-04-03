"""
---
title: Latent Diffusion Models
summary: >
 Annotated PyTorch implementation/tutorial of latent diffusion models from paper
 High-Resolution Image Synthesis with Latent Diffusion Models
---

# Latent Diffusion Models

Latent diffusion models use an auto-encoder to map between image space and
latent space. The diffusion model works on the diffusion space, which makes it
a lot easier to train.
It is based on paper
[High-Resolution Image Synthesis with Latent Diffusion Models](https://papers.labml.ai/paper/2112.10752).

They use a pre-trained auto-encoder and train the diffusion U-Net on the latent
space of the pre-trained auto-encoder.

For a simpler diffusion implementation refer to our [DDPM implementation](../ddpm/index.html).
We use same notations for $\alpha_t$, $\beta_t$ schedules, etc.
"""

from typing import List, Tuple, Optional, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
from .model.autoencoder import Autoencoder
from .model.unet import UNetModel
import random


def gather(consts: torch.Tensor, t: torch.Tensor):
    """Gather consts for $t$ and reshape to feature map shape"""
    c = consts.gather(-1, t)
    return c.reshape(-1, 1, 1, 1)


class LatentDiffusion(nn.Module):
    """
    ## Latent diffusion model

    This contains following components:

    * [AutoEncoder](model/autoencoder.html)
    * [U-Net](model/unet.html) with [attention](model/unet_attention.html)
    """
    eps_model: UNetModel
    first_stage_model: Optional[Autoencoder] = None

    def __init__(
        self,
        unet_model: UNetModel,
        autoencoder: Optional[Autoencoder],
        autoreg_cond_enc,
        external_cond_enc,
        latent_scaling_factor: float,
        n_steps: int,
        linear_start: float,
        linear_end: float,
        debug_mode: Optional[bool] = False
    ):
        """
        :param unet_model: is the [U-Net](model/unet.html) that predicts noise
         $\epsilon_\text{cond}(x_t, c)$, in latent space
        :param autoencoder: is the [AutoEncoder](model/autoencoder.html)
        :param latent_scaling_factor: is the scaling factor for the latent space. The encodings of
         the autoencoder are scaled by this before feeding into the U-Net.
        :param n_steps: is the number of diffusion steps $T$.
        :param linear_start: is the start of the $\beta$ schedule.
        :param linear_end: is the end of the $\beta$ schedule.
        """
        super().__init__()
        # Wrap the [U-Net](model/unet.html) to keep the same model structure as
        # [CompVis/stable-diffusion](https://github.com/CompVis/stable-diffusion).
        self.eps_model = unet_model
        # Auto-encoder and scaling factor
        self.first_stage_model = autoencoder
        self.autoreg_cond_enc = autoreg_cond_enc
        self.external_cond_enc = external_cond_enc
        # freeze autoencoder's parameters
        if self.first_stage_model is not None:
            for param in self.first_stage_model.parameters():
                param.requires_grad = False
        self.latent_scaling_factor = latent_scaling_factor

        # Number of steps $T$
        self.n_steps = n_steps

        # $\beta$ schedule
        beta = torch.linspace(
            linear_start**0.5, linear_end**0.5, n_steps, dtype=torch.float64
        ) ** 2
        # $\alpha_t = 1 - \beta_t$
        alpha = 1. - beta
        # $\bar\alpha_t = \prod_{s=1}^t \alpha_s$
        alpha_bar = torch.cumprod(alpha, dim=0)
        self.alpha = nn.Parameter(alpha.to(torch.float32), requires_grad=False)
        self.beta = nn.Parameter(beta.to(torch.float32), requires_grad=False)
        self.alpha_bar = nn.Parameter(alpha_bar.to(torch.float32), requires_grad=False)
        self.sigma2 = self.beta

        self.debug_mode = debug_mode

    @property
    def device(self):
        """
        ### Get model device
        """
        return next(iter(self.eps_model.parameters())).device

    def autoencoder_encode(self, image: torch.Tensor):
        """
        ### Get scaled latent space representation of the image

        The encoder output is a distribution.
        We sample from that and multiply by the scaling factor.
        """
        if self.first_stage_model is not None:
            return self.latent_scaling_factor * self.first_stage_model.encode(image
                                                                             ).sample()
        else:
            return image

    def autoencoder_decode(self, z: torch.Tensor):
        """
        ### Get image from the latent representation

        We scale down by the scaling factor and then decode.
        """
        if self.first_stage_model is not None:
            return self.first_stage_model.decode(z / self.latent_scaling_factor)
        else:
            return z

    def forward(self, x: torch.Tensor, t: torch.Tensor, context: torch.Tensor):
        """
        ### Predict noise

        Predict noise given the latent representation $x_t$, time step $t$, and the
        conditioning context $c$.

        $$\epsilon_\text{cond}(x_t, c)$$
        """
        return self.eps_model(x, t, context)

    def q_xt_x0(self, x0: torch.Tensor,
                t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        #### Get $q(x_t|x_0)$ distribution
        """

        # [gather](utils.html) $\alpha_t$ and compute $\sqrt{\bar\alpha_t} x_0$
        mean = gather(self.alpha_bar, t)**0.5 * x0
        # $(1-\bar\alpha_t) \mathbf{I}$
        var = 1 - gather(self.alpha_bar, t)
        #
        return mean, var

    def q_sample(
        self, x0: torch.Tensor, t: torch.Tensor, eps: Optional[torch.Tensor] = None
    ):
        """
        #### Sample from $q(x_t|x_0)$
        """

        # $\epsilon \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$
        if eps is None:
            eps = torch.randn_like(x0)

        # get $q(x_t|x_0)$
        mean, var = self.q_xt_x0(x0, t)
        # Sample from $q(x_t|x_0)$
        return mean + (var**0.5) * eps

    def p_sample(self, xt: torch.Tensor, t: torch.Tensor):
        """
        #### Sample from $\textcolor{lightgreen}{p_\theta}(x_{t-1}|x_t)$
        """

        # $\textcolor{lightgreen}{\epsilon_\theta}(x_t, t)$
        eps_theta = self.eps_model(xt, t)
        # [gather](utils.html) $\bar\alpha_t$
        alpha_bar = gather(self.alpha_bar, t)
        # $\alpha_t$
        alpha = gather(self.alpha, t)
        # $\frac{\beta}{\sqrt{1-\bar\alpha_t}}$
        eps_coef = (1 - alpha) / (1 - alpha_bar)**.5
        # $$\frac{1}{\sqrt{\alpha_t}} \Big(x_t -
        #      \frac{\beta_t}{\sqrt{1-\bar\alpha_t}}\textcolor{lightgreen}{\epsilon_\theta}(x_t, t) \Big)$$
        mean = 1 / (alpha**0.5) * (xt - eps_coef * eps_theta)
        # $\sigma^2$
        var = gather(self.sigma2, t)

        # $\epsilon \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$
        eps = torch.randn(xt.shape, device=xt.device)
        # Sample
        return mean + (var**.5) * eps

    def loss(
        self,
        x0: torch.Tensor,
        autoreg_cond: Union[torch.Tensor, None],
        external_cond: Union[torch.Tensor, None],
        noise: Optional[torch.Tensor] = None,
    ):
        """
        #### Simplified Loss
        """
        # Get batch size
        batch_size = x0.shape[0]
        # Get random $t$ for each sample in the batch
        t = torch.randint(
            0, self.n_steps, (batch_size, ), device=x0.device, dtype=torch.long
        )
        if self.first_stage_model is not None:
            x0 = self.autoencoder_encode(x0)

        if self.debug_mode:
            print('autoreg_cond is None:', autoreg_cond is None, 'external_cond is None:', external_cond is None)

        if autoreg_cond is None:
            autoreg_cond = -torch.ones(x0.size(0), 1, self.eps_model.d_cond, device=x0.device, dtype=x0.dtype)
        else:
            autoreg_cond = self.autoreg_cond_enc(autoreg_cond)

        if external_cond is not None:
            external_cond = self.external_cond_enc(external_cond)
            if random.random() < 0.2:
                external_cond = (-torch.ones_like(external_cond)).to(self.device)

            cond = torch.cat([autoreg_cond, external_cond], 1)
        else:
            cond = autoreg_cond

        if x0.size(1) == self.eps_model.out_channels:  # generating form
            if self.debug_mode:
                print('In the mode of root level:', x0.size(), cond.size())
            if noise is None:
                noise = torch.randn_like(x0)

            xt = self.q_sample(x0, t, eps=noise)

            eps_theta = self.eps_model(xt, t, cond)

            loss = F.mse_loss(noise, eps_theta)
        else:
            if self.debug_mode:
                print('In the mode of non-root level:', x0.size(), cond.size(), autoreg_cond.size())

            if noise is None:
                noise = torch.randn_like(x0[:, 0: 2])

            front_t = self.q_sample(x0[:, 0: 2], t, eps=noise)

            background_cond = x0[:, 2:]

            xt = torch.cat([front_t, background_cond], 1)

            eps_theta = self.eps_model(xt, t, cond)

            loss = F.mse_loss(noise, eps_theta)
        if self.debug_mode:
            print('loss:', loss)
        return loss
