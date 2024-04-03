import torch
from torch import nn
from .pretrained_encoders import load_chd_enc8, load_ec2vae_enc2, load_txt_enc2
import os


PROJECT_PATH = os.path.realpath(os.path.join(__file__, '../../../../'))


class CtpExternalEncoder(nn.Module):

    def __init__(self, chd_enc8):
        super().__init__()
        self.enc = chd_enc8
        self.squeezer = nn.Linear(512, 128)
        self.pos_enc = nn.Embedding(4, 128)

    def forward(self, chd):
        pos = torch.arange(0, 4).to(chd.device).long()
        pos_emb = self.pos_enc(pos).unsqueeze(0)

        bs = chd.size(0)
        chd = chd.reshape(-1, 32, 36)

        self.enc.eval()
        with torch.no_grad():
            z = self.enc(chd).reshape(bs, 4, 512)
        z = self.squeezer(z)
        z += pos_emb
        return z

    @classmethod
    def create_model(cls):
        chd_enc8 = load_chd_enc8()
        return cls(chd_enc8)


class LshExternalEncoder(nn.Module):

    def __init__(self, rhy_enc2):
        super().__init__()
        self.enc = rhy_enc2
        self.squeezer = nn.Linear(128, 256)
        self.pos_enc = nn.Embedding(4, 256)

    def forward(self, mel_pr):
        pos = torch.arange(0, 4).to(mel_pr.device).long()
        pos_emb = self.pos_enc(pos).unsqueeze(0)

        bs = mel_pr.size(0)
        mel = mel_pr[:, :, 0: 130].reshape(-1, 32, 130)
        chd = mel_pr[:, :, 130:].reshape(-1, 32, 12)

        self.enc.eval()
        with torch.no_grad():
            _, z = self.enc(mel, chd)
            z = z.mean.reshape(bs, 4, 128)
        z = self.squeezer(z) + pos_emb
        return z

    @classmethod
    def create_model(cls):
        enc = load_ec2vae_enc2()
        return cls(enc)


class AccExternalEncoder(nn.Module):

    def __init__(self, txt_enc2):
        super().__init__()
        self.enc = txt_enc2
        self.pos_enc = nn.Embedding(4, 256)

    def forward(self, pr_mat):
        pos = torch.arange(0, 4).to(pr_mat.device).long()
        pos_emb = self.pos_enc(pos).unsqueeze(0)

        bs = pr_mat.size(0)
        pr_mat = pr_mat.reshape(-1, 32, 128)
        self.enc.eval()
        with torch.no_grad():
            z = self.enc(pr_mat)
            z = z.mean.reshape(bs, 4, 256)
        z = z + pos_emb
        return z

    @classmethod
    def create_model(cls):
        txt_enc2 = load_txt_enc2()
        return cls(txt_enc2)
