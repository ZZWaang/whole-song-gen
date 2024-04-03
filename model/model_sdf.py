import torch
import torch.nn as nn
from .stable_diffusion.latent_diffusion import LatentDiffusion


class Diffpro_SDF(nn.Module):

    def __init__(
        self,
        ldm: LatentDiffusion,
    ):
        """
        cond_type: {chord, texture}
        cond_mode: {cond, mix, uncond}
            mix: use a special condition for unconditional learning with probability of 0.2
        use_enc: whether to use pretrained chord encoder to generate encoded condition
        """
        super(Diffpro_SDF, self).__init__()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.ldm = ldm

    @classmethod
    def load_trained(
        cls,
        ldm,
        chkpt_fpath,
    ):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = cls(ldm)
        trained_leaner = torch.load(chkpt_fpath, map_location=device)
        try:
            model.load_state_dict(trained_leaner["model"])
        except RuntimeError:
            model_dict = trained_leaner["model"]
            model_dict = {k.replace('cond_enc', 'autoreg_cond_enc'): v for k, v in model_dict.items()}
            model_dict = {k.replace('style_enc', 'external_cond_enc'): v for k, v in model_dict.items()}
            model.load_state_dict(model_dict)
        return model

    def p_sample(self, xt: torch.Tensor, t: torch.Tensor):
        return self.ldm.p_sample(xt, t)

    def q_sample(self, x0: torch.Tensor, t: torch.Tensor):
        return self.ldm.q_sample(x0, t)

    def get_loss_dict(self, batch, step):
        """
        z_y is the stuff the diffusion model needs to learn
        """
        # x = batch.float().to(self.device)

        x, autoreg_cond, external_cond = batch
        loss = self.ldm.loss(x, autoreg_cond, external_cond)
        return {"loss": loss}
