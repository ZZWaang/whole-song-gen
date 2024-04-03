import torch
from torch import nn


class CtpAutoregEncoder(nn.Module):

    def __init__(self, img_h=108, img_w=128):
        super().__init__()
        input_channels = 10
        mid_channels = 20
        output_channels = 2
        self.layers = nn.Sequential(
            nn.Conv2d(input_channels, mid_channels, 3, padding=1),
            nn.SiLU(),
            nn.LayerNorm([mid_channels, img_h, img_w]),
            nn.Conv2d(mid_channels, mid_channels, 3, padding=1),
            nn.SiLU(),
            nn.LayerNorm([mid_channels, img_h, img_w]),
            nn.Conv2d(mid_channels, output_channels, 3, padding=1),
        )
        self.img_h = img_h
        self.img_w = img_w
        self.squeeze = nn.Linear(1024, 128)
        self.output_channels = output_channels
        self.pos_enc = nn.Embedding(27, 128)

    def forward(self, x):
        bs = x.size(0)
        x = self.layers(x)
        x = x.reshape(bs, self.output_channels, 27, 4 * 128).permute(0, 2, 1, 3).reshape(bs, 27, -1)
        x = self.squeeze(x)

        pos = torch.arange(0, 27).to(x.device).long()
        pos_emb = self.pos_enc(pos).unsqueeze(0)
        x += pos_emb
        return x


class LshAutoregEncoder(nn.Module):

    def __init__(self, img_h=136, img_w=128):
        super().__init__()
        input_channels = 12
        mid_channels = 20
        output_channels = 2
        self.layers = nn.Sequential(
            nn.Conv2d(input_channels, mid_channels, 3, padding=1),
            nn.SiLU(),
            nn.LayerNorm([mid_channels, img_h, img_w]),
            nn.Conv2d(mid_channels, mid_channels, 3, padding=1),
            nn.SiLU(),
            nn.LayerNorm([mid_channels, img_h, img_w]),
            nn.Conv2d(mid_channels, output_channels, 3, padding=1),
        )
        self.img_h = img_h
        self.img_w = img_w
        self.squeeze = nn.Linear(2048, 256)
        self.output_channels = output_channels
        self.pos_enc = nn.Embedding(17, 256)

    def forward(self, x):
        bs = x.size(0)
        x = self.layers(x)
        x = x.reshape(bs, self.output_channels, 17, 8 * 128).permute(0, 2, 1, 3).reshape(bs, 17, -1)
        x = self.squeeze(x)

        pos = torch.arange(0, 17).to(x.device).long()
        pos_emb = self.pos_enc(pos).unsqueeze(0)
        x += pos_emb
        return x


class AccAutoregEncoder(nn.Module):

    def __init__(self, img_h=136, img_w=128):
        super().__init__()
        input_channels = 14
        mid_channels = 20
        output_channels = 2
        self.layers = nn.Sequential(
            nn.Conv2d(input_channels, mid_channels, 3, padding=1),
            nn.SiLU(),
            nn.LayerNorm([mid_channels, img_h, img_w]),
            nn.Conv2d(mid_channels, mid_channels, 3, padding=1),
            nn.SiLU(),
            nn.LayerNorm([mid_channels, img_h, img_w]),
            nn.Conv2d(mid_channels, output_channels, 3, padding=1),
        )
        self.img_h = img_h
        self.img_w = img_w
        self.squeeze = nn.Linear(2048, 256)
        self.output_channels = output_channels
        self.pos_enc = nn.Embedding(17, 256)

    def forward(self, x):
        bs = x.size(0)
        x = self.layers(x)[:, :, 0: 200]
        x = x.reshape(bs, self.output_channels, 17, 8 * 128).permute(0, 2, 1, 3).reshape(bs, 17, -1)
        x = self.squeeze(x)

        pos = torch.arange(0, 17).to(x.device).long()
        pos_emb = self.pos_enc(pos).unsqueeze(0)
        x += pos_emb
        return x
