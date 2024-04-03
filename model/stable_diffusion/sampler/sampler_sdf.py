from typing import Optional, List, Union
import numpy as np
import torch
from labml import monit
from model.stable_diffusion.latent_diffusion import LatentDiffusion
from model.stable_diffusion.sampler import DiffusionSampler


class SDFSampler(DiffusionSampler):
    """
    ## DDPM Sampler

    This extends the [`DiffusionSampler` base class](index.html).

    DDPM samples images by repeatedly removing noise by sampling step by step from
    $p_\theta(x_{t-1} | x_t)$,

    \begin{align}

    p_\theta(x_{t-1} | x_t) &= \mathcal{N}\big(x_{t-1}; \mu_\theta(x_t, t), \tilde\beta_t \mathbf{I} \big) \\

    \mu_t(x_t, t) &= \frac{\sqrt{\bar\alpha_{t-1}}\beta_t}{1 - \bar\alpha_t}x_0
                         + \frac{\sqrt{\alpha_t}(1 - \bar\alpha_{t-1})}{1-\bar\alpha_t}x_t \\

    \tilde\beta_t &= \frac{1 - \bar\alpha_{t-1}}{1 - \bar\alpha_t} \beta_t \\

    x_0 &= \frac{1}{\sqrt{\bar\alpha_t}} x_t -  \Big(\sqrt{\frac{1}{\bar\alpha_t} - 1}\Big)\epsilon_\theta \\

    \end{align}
    """

    model: LatentDiffusion

    def __init__(
        self,
        model: LatentDiffusion,
        max_l,
        h,
        is_autocast=False,
        is_show_image=False,
        device=None,
        debug_mode=False
    ):
        """
        :param model: is the model to predict noise $\epsilon_\text{cond}(x_t, c)$
        """
        super().__init__(model)
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        # Sampling steps $1, 2, \dots, T$
        self.time_steps = np.asarray(list(range(self.n_steps)), dtype=np.int32)

        self.is_show_image = is_show_image

        self.autocast = torch.cuda.amp.autocast(enabled=is_autocast)

        self.out_channel = self.model.eps_model.out_channels
        self.max_l = max_l
        self.h = h
        self.debug_mode = debug_mode

        with torch.no_grad():
            # $\bar\alpha_t$
            alpha_bar = self.model.alpha_bar
            # $\beta_t$ schedule
            beta = self.model.beta
            #  $\bar\alpha_{t-1}$
            alpha_bar_prev = torch.cat([alpha_bar.new_tensor([1.]), alpha_bar[:-1]])

            # $\sqrt{\bar\alpha}$
            self.sqrt_alpha_bar = alpha_bar**.5
            # $\sqrt{1 - \bar\alpha}$
            self.sqrt_1m_alpha_bar = (1. - alpha_bar)**.5
            # $\frac{1}{\sqrt{\bar\alpha_t}}$
            self.sqrt_recip_alpha_bar = alpha_bar**-.5
            # $\sqrt{\frac{1}{\bar\alpha_t} - 1}$
            self.sqrt_recip_m1_alpha_bar = (1 / alpha_bar - 1)**.5

            # $\frac{1 - \bar\alpha_{t-1}}{1 - \bar\alpha_t} \beta_t$
            variance = beta * (1. - alpha_bar_prev) / (1. - alpha_bar)
            # Clamped log of $\tilde\beta_t$
            self.log_var = torch.log(torch.clamp(variance, min=1e-20))
            # $\frac{\sqrt{\bar\alpha_{t-1}}\beta_t}{1 - \bar\alpha_t}$
            self.mean_x0_coef = beta * (alpha_bar_prev**.5) / (1. - alpha_bar)
            # $\frac{\sqrt{\alpha_t}(1 - \bar\alpha_{t-1})}{1-\bar\alpha_t}$
            self.mean_xt_coef = (1. - alpha_bar_prev) * ((1 - beta)**
                                                         0.5) / (1. - alpha_bar)

    @property
    def d_cond(self):
        return self.model.eps_model.d_cond

    def get_eps(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        background_cond: Optional[torch.Tensor],
        autoreg_cond: Optional[torch.Tensor],
        external_cond: Optional[torch.Tensor],
        uncond_scale: Optional[float],
    ):
        """
        ## Get $\epsilon(x_t, c)$

        :param x: is $x_t$ of shape `[batch_size, channels, height, width]`
        :param t: is $t$ of shape `[batch_size]`
        :param background_cond: background condition
        :param autoreg_cond: autoregressive condition
        :param external_cond: external condition
        :param c: is the conditional embeddings $c$ of shape `[batch_size, emb_size]`
        :param uncond_scale: is the unconditional guidance scale $s$. This is used for
            $\epsilon_\theta(x_t, c) = s\epsilon_\text{cond}(x_t, c) + (s - 1)\epsilon_\text{cond}(x_t, c_u)$
        :param uncond_cond: is the conditional embedding for empty prompt $c_u$
        """
        # When the scale $s = 1$
        # $$\epsilon_\theta(x_t, c) = \epsilon_\text{cond}(x_t, c)$$

        batch_size = x.size(0)

        if autoreg_cond is None:
            autoreg_cond = -torch.ones(batch_size, 1, self.d_cond, device=x.device, dtype=x.dtype)
        else:
            autoreg_cond = self.model.autoreg_cond_enc(autoreg_cond)

        if hasattr(self.model, 'style_enc'):
            if external_cond is not None:
                external_cond = self.model.external_cond_enc(external_cond)
                if uncond_scale is None or uncond_scale == 1:
                    external_uncond = (-torch.ones_like(external_cond)).to(self.device)
                else:
                    external_uncond = None
                # if random.random() < 0.2:
                #     external_cond = (-torch.ones_like(external_cond)).to(self.device)
            else:
                external_cond = -torch.ones(batch_size, 4, self.d_cond, device=x.device, dtype=x.dtype)
                external_uncond = None
            cond = torch.cat([autoreg_cond, external_cond], 1)
            if external_uncond is None:
                uncond = None
            else:
                uncond = torch.cat([autoreg_cond, external_uncond], 1)
        else:
            cond = autoreg_cond
            uncond = None

        if background_cond is not None:
            x = torch.cat([x, background_cond], 1) if background_cond is not None else x

        if uncond is None:
            e_t = self.model(x, t, cond)
        else:
            e_t_cond = self.model(x, t, cond)
            e_t_uncond = self.model(x, t, uncond)
            e_t = e_t_uncond + uncond_scale * (e_t_cond - e_t_uncond)
        return e_t

    @torch.no_grad()
    def p_sample(
        self,
        x: torch.Tensor,
        background_cond: Optional[torch.Tensor],
        autoreg_cond: Optional[torch.Tensor],
        external_cond: Optional[torch.Tensor],
        t: torch.Tensor,
        step: int,
        repeat_noise: bool = False,
        temperature: float = 1.,
        uncond_scale: float = 1.,
    ):
        """
        ### Sample $x_{t-1}$ from $p_\theta(x_{t-1} | x_t)$

        :param x: is $x_t$ of shape `[batch_size, channels, height, width]`
        :param background_cond: background condition
        :param autoreg_cond: autoregressive condition
        :param external_cond: external condition
        :param t: is $t$ of shape `[batch_size]`
        :param step: is the step $t$ as an integer
        :param repeat_noise: specified whether the noise should be same for all samples in the batch
        :param temperature: is the noise temperature (random noise gets multiplied by this)
        :param uncond_scale: is the unconditional guidance scale $s$. This is used for
            $\epsilon_\theta(x_t, c) = s\epsilon_\text{cond}(x_t, c) + (s - 1)\epsilon_\text{cond}(x_t, c_u)$
        """

        # Get $\epsilon_\theta$
        with self.autocast:
            e_t = self.get_eps(x, t, background_cond, autoreg_cond, external_cond, uncond_scale=uncond_scale)

        # Get batch size
        bs = x.shape[0]

        # $\frac{1}{\sqrt{\bar\alpha_t}}$
        sqrt_recip_alpha_bar = x.new_full(
            (bs, 1, 1, 1), self.sqrt_recip_alpha_bar[step]
        )
        # $\sqrt{\frac{1}{\bar\alpha_t} - 1}$
        sqrt_recip_m1_alpha_bar = x.new_full(
            (bs, 1, 1, 1), self.sqrt_recip_m1_alpha_bar[step]
        )

        # Calculate $x_0$ with current $\epsilon_\theta$
        #
        # $$x_0 = \frac{1}{\sqrt{\bar\alpha_t}} x_t -  \Big(\sqrt{\frac{1}{\bar\alpha_t} - 1}\Big)\epsilon_\theta$$
        x0 = sqrt_recip_alpha_bar * x[:, 0: e_t.size(1)] - sqrt_recip_m1_alpha_bar * e_t

        # $\frac{\sqrt{\bar\alpha_{t-1}}\beta_t}{1 - \bar\alpha_t}$
        mean_x0_coef = x.new_full((bs, 1, 1, 1), self.mean_x0_coef[step])
        # $\frac{\sqrt{\alpha_t}(1 - \bar\alpha_{t-1})}{1-\bar\alpha_t}$
        mean_xt_coef = x.new_full((bs, 1, 1, 1), self.mean_xt_coef[step])

        # Calculate $\mu_t(x_t, t)$
        #
        # $$\mu_t(x_t, t) = \frac{\sqrt{\bar\alpha_{t-1}}\beta_t}{1 - \bar\alpha_t}x_0
        #    + \frac{\sqrt{\alpha_t}(1 - \bar\alpha_{t-1})}{1-\bar\alpha_t}x_t$$
        mean = mean_x0_coef * x0 + mean_xt_coef * x[:, 0: e_t.size(1)]
        # $\log \tilde\beta_t$
        log_var = x.new_full((bs, 1, 1, 1), self.log_var[step])

        # Do not add noise when $t = 1$ (final step sampling process).
        # Note that `step` is `0` when $t = 1$)
        if step == 0:
            noise = 0
        # If same noise is used for all samples in the batch
        elif repeat_noise:
            noise = torch.randn((1, *x0.shape[1:]), device=self.device)
        # Different noise for each sample
        else:
            noise = torch.randn(x0.shape, device=self.device)

        # Multiply noise by the temperature
        noise = noise * temperature

        # Sample from,
        #
        # $$p_\theta(x_{t-1} | x_t) = \mathcal{N}\big(x_{t-1}; \mu_\theta(x_t, t), \tilde\beta_t \mathbf{I} \big)$$
        x_prev = mean + (0.5 * log_var).exp() * noise

        #
        return x_prev, x0, e_t

    @torch.no_grad()
    def q_sample(
        self, x0: torch.Tensor, index: int, noise: Optional[torch.Tensor] = None
    ):
        """
        ### Sample from $q(x_t|x_0)$

        $$q(x_t|x_0) = \mathcal{N} \Big(x_t; \sqrt{\bar\alpha_t} x_0, (1-\bar\alpha_t) \mathbf{I} \Big)$$

        :param x0: is $x_0$ of shape `[batch_size, channels, height, width]`
        :param index: is the time step $t$ index
        :param noise: is the noise, $\epsilon$
        """

        # Random noise, if noise is not specified
        if noise is None:
            noise = torch.randn_like(x0, device=self.device)

        # Sample from $\mathcal{N} \Big(x_t; \sqrt{\bar\alpha_t} x_0, (1-\bar\alpha_t) \mathbf{I} \Big)$
        return self.sqrt_alpha_bar[index] * x0 + self.sqrt_1m_alpha_bar[index] * noise

    @torch.no_grad()
    def sample(
        self,
        shape: List[int],
        background_cond: Optional[torch.Tensor] = None,
        autoreg_cond: Optional[torch.Tensor] = None,
        external_cond: Optional[torch.Tensor] = None,
        repeat_noise: bool = False,
        temperature: float = 1.,
        uncond_scale: float = 1.,
        x_last: Optional[torch.Tensor] = None,
        t_start: int = 0,
    ):
        """
        ### Sampling Loop

        :param shape: is the shape of the generated images in the
            form `[batch_size, channels, height, width]`
        :param background_cond: background condition
        :param autoreg_cond: autoregressive condition
        :param external_cond: external condition
        :param repeat_noise: specified whether the noise should be same for all samples in the batch
        :param temperature: is the noise temperature (random noise gets multiplied by this)
        :param x_last: is $x_T$. If not provided random noise will be used.
        :param uncond_scale: is the unconditional guidance scale $s$. This is used for
            $\epsilon_\theta(x_t, c) = s\epsilon_\text{cond}(x_t, c) + (s - 1)\epsilon_\text{cond}(x_t, c_u)$
        :param t_start: t_start
        """

        # Get device and batch size
        bs = shape[0]

        # Get $x_T$
        x = x_last if x_last is not None else torch.randn(shape, device=self.device)

        # Time steps to sample at $T - t', T - t' - 1, \dots, 1$
        time_steps = np.flip(self.time_steps)[t_start:]

        # Sampling loop
        for step in monit.iterate('Sample', time_steps):
            # Time step $t$
            ts = x.new_full((bs, ), step, dtype=torch.long)

            x, pred_x0, e_t = self.p_sample(
                x,
                background_cond,
                autoreg_cond,
                external_cond,
                ts,
                step,
                repeat_noise=repeat_noise,
                temperature=temperature,
                uncond_scale=uncond_scale,
            )

            s1 = step + 1

            # if self.is_show_image:
            #     if s1 % 100 == 0 or (s1 <= 100 and s1 % 25 == 0):
            #         show_image(x, f"exp/img/x{s1}.png")

        # Return $x_0$
        # if self.is_show_image:
        #     show_image(x, f"exp/img/x0.png")

        return x

    @torch.no_grad()
    def paint(
        self,
        x: Optional[torch.Tensor] = None,
        background_cond: Optional[torch.Tensor] = None,
        autoreg_cond: Optional[torch.Tensor] = None,
        external_cond: Optional[torch.Tensor] = None,
        t_start: int = 0,
        orig: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
        orig_noise: Optional[torch.Tensor] = None,
        uncond_scale: float = 1.,
    ):
        """
        ### Painting Loop

        :param x: is $x_{S'}$ of shape `[batch_size, channels, height, width]`
        :param background_cond: background condition
        :param autoreg_cond: autoregressive condition
        :param external_cond: external condition
        :param t_start: is the sampling step to start from, $S'$
        :param orig: is the original image in latent page which we are in paining.
            If this is not provided, it'll be an image to image transformation.
        :param mask: is the mask to keep the original image.
        :param orig_noise: is fixed noise to be added to the original image.
        :param uncond_scale: is the unconditional guidance scale $s$. This is used for
            $\epsilon_\theta(x_t, c) = s\epsilon_\text{cond}(x_t, c) + (s - 1)\epsilon_\text{cond}(x_t, c_u)$
        """
        # Get  batch size
        bs = orig.size(0)

        if x is None:
            x = torch.randn(orig.shape, device=self.device)

        # Time steps to sample at $\tau_{S`}, \tau_{S' - 1}, \dots, \tau_1$
        # time_steps = np.flip(self.time_steps[: t_start])
        time_steps = np.flip(self.time_steps)[t_start:]

        for i, step in monit.enum('Paint', time_steps):
            # Index $i$ in the list $[\tau_1, \tau_2, \dots, \tau_S]$
            # index = len(time_steps) - i - 1
            # Time step $\tau_i$
            ts = x.new_full((bs, ), step, dtype=torch.long)

            # Sample $x_{\tau_{i-1}}$
            x, _, _ = self.p_sample(
                x,
                background_cond,
                autoreg_cond,
                external_cond,
                t=ts,
                step=step,
                uncond_scale=uncond_scale
            )

            # Replace the masked area with original image
            if orig is not None:
                assert mask is not None
                # Get the $q_{\sigma,\tau}(x_{\tau_i}|x_0)$ for original image in latent space
                orig_t = self.q_sample(orig, step, noise=orig_noise)
                # Replace the masked area
                x = orig_t * mask + x * (1 - mask)

            s1 = step + 1

            # if self.is_show_image:
            #     if s1 % 100 == 0 or (s1 <= 100 and s1 % 25 == 0):
            #         show_image(x, f"exp/img/x{s1}.png")

        # if self.is_show_image:
        #     show_image(x, f"exp/img/x0.png")
        return x

    def generate(self, background_cond=None, autoreg_cond=None, external_cond=None, orig_x=None,
                 mask=None, batch_size=None, uncond_scale=None):

        if batch_size is None:
            for ele in [orig_x, background_cond, autoreg_cond, external_cond]:
                if ele is not None:
                    batch_size = ele.size(0)
                    break
            else:
                ValueError("batch size cannot be determined.")
        shape = [batch_size, self.out_channel, self.max_l, self.h]

        if self.debug_mode:
            return torch.randn(shape, dtype=torch.float)

        if orig_x is None:  # sample
            return self.sample(shape, background_cond, autoreg_cond, external_cond, uncond_scale=uncond_scale)
        else:  # paint
            return self.paint(None, background_cond, autoreg_cond, external_cond, orig=orig_x, mask=mask,
                              uncond_scale=uncond_scale)
