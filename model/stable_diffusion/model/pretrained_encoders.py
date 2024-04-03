import torch
from torch import nn
from torch.distributions import Normal
import os


PROJECT_PATH = os.path.realpath(os.path.join(__file__, '../../../../'))


class Ec2VaeEncoder(nn.Module):

    roll_dims = 130
    rhythm_dims = 3
    condition_dims = 12

    def __init__(self, hidden_dims, z1_dims, z2_dims, n_step):
        super(Ec2VaeEncoder, self).__init__()

        assert n_step in [16, 32, 64, 128]

        self.gru_0 = nn.GRU(self.roll_dims + self.condition_dims,
                            hidden_dims, batch_first=True, bidirectional=True)
        self.linear_mu = nn.Linear(hidden_dims * 2, z1_dims + z2_dims)
        self.linear_var = nn.Linear(hidden_dims * 2, z1_dims + z2_dims)

        self.hidden_dims = hidden_dims
        self.z1_dims = z1_dims
        self.z2_dims = z2_dims
        self.n_step = n_step

    def encoder(self, x, condition):
        # self.gru_0.flatten_parameters()
        x = torch.cat((x, condition), -1)
        x = self.gru_0(x)[-1]
        x = x.transpose_(0, 1).contiguous()
        x = x.view(x.size(0), -1)
        mu = self.linear_mu(x)
        var = self.linear_var(x).exp_()
        distribution_1 = Normal(mu[:, :self.z1_dims], var[:, :self.z1_dims])
        distribution_2 = Normal(mu[:, self.z1_dims:], var[:, self.z1_dims:])
        return distribution_1, distribution_2

    def forward(self, x, condition):
        return self.encoder(x, condition)

    @classmethod
    def create_2bar_encoder(cls, hidden_dims=2048, zp_dims=128, zr_dims=128):
        return cls(hidden_dims, zp_dims, zr_dims, 32)

    @classmethod
    def create_4bar_encoder(cls, hidden_dims=2048, zp_dims=128, zr_dims=128):
        return cls(hidden_dims, zp_dims, zr_dims, 64)

    @classmethod
    def create_8bar_encoder(cls, hidden_dims=2048, zp_dims=128, zr_dims=128):
        return cls(hidden_dims, zp_dims, zr_dims, 128)


class PolydisChordEncoder(nn.Module):

    def __init__(self, input_dim, hidden_dim, z_dim):
        super(PolydisChordEncoder, self).__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, batch_first=True,
                          bidirectional=True)
        self.linear_mu = nn.Linear(hidden_dim * 2, z_dim)
        self.linear_var = nn.Linear(hidden_dim * 2, z_dim)
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.z_dim = z_dim

    def forward(self, x):
        x = self.gru(x)[-1]
        x = x.transpose_(0, 1).contiguous()
        x = x.view(x.size(0), -1)
        mu = self.linear_mu(x)
        # var = self.linear_var(x).exp_()
        # dist = Normal(mu, var)
        return mu

    @classmethod
    def create_encoder(cls, hidden_dim=1024, z_dim=256):
        return cls(36, hidden_dim, z_dim)


class PolydisTextureEncoder(nn.Module):

    def __init__(self, emb_size, hidden_dim, z_dim, num_channel=10):
        """input must be piano_mat: (B, 32, 128)"""
        super(PolydisTextureEncoder, self).__init__()
        self.cnn = nn.Sequential(nn.Conv2d(1, num_channel, kernel_size=(4, 12),
                                           stride=(4, 1), padding=0),
                                 nn.ReLU(),
                                 nn.MaxPool2d(kernel_size=(1, 4),
                                              stride=(1, 4)))
        self.fc1 = nn.Linear(num_channel * 29, 1000)
        self.fc2 = nn.Linear(1000, emb_size)
        self.gru = nn.GRU(emb_size, hidden_dim, batch_first=True,
                          bidirectional=True)
        self.linear_mu = nn.Linear(hidden_dim * 2, z_dim)
        self.linear_var = nn.Linear(hidden_dim * 2, z_dim)
        self.emb_size = emb_size
        self.hidden_dim = hidden_dim
        self.z_dim = z_dim

    def forward(self, pr):
        # pr: (bs, 32, 128)
        bs = pr.size(0)
        pr = pr.unsqueeze(1)
        pr = self.cnn(pr).view(bs, 8, -1)
        pr = self.fc2(self.fc1(pr))  # (bs, 8, emb_size)
        pr = self.gru(pr)[-1]
        pr = pr.transpose_(0, 1).contiguous()
        pr = pr.view(pr.size(0), -1)
        mu = self.linear_mu(pr)
        var = self.linear_var(pr).exp_()
        dist = Normal(mu, var)
        return dist

    @classmethod
    def create_encoder(cls, emb_size=256, hidden_dim=1024, num_channel=10, z_dim=256):
        return cls(emb_size, hidden_dim, z_dim, num_channel)


def load_ec2vae_enc8():
    ec2vae_enc8 = Ec2VaeEncoder.create_8bar_encoder()
    model_path = os.path.join(PROJECT_PATH, 'pretrained_models', 'ec2vae_enc_8bar.pt')
    state_dict = torch.load(model_path)
    ec2vae_enc8.load_state_dict(state_dict)
    for param in ec2vae_enc8.parameters():
        param.requires_grad = False
    return ec2vae_enc8


def load_ec2vae_enc2():
    ec2vae_enc2 = Ec2VaeEncoder.create_2bar_encoder()
    model_path = os.path.join(PROJECT_PATH, 'pretrained_models', 'ec2vae_enc_2bar.pt')
    state_dict = torch.load(model_path)
    ec2vae_enc2.load_state_dict(state_dict)
    for param in ec2vae_enc2.parameters():
        param.requires_grad = False
    return ec2vae_enc2


def load_chd_enc8():
    chd_enc = PolydisChordEncoder.create_encoder(hidden_dim=512, z_dim=512)
    model_path = os.path.join(PROJECT_PATH, 'pretrained_models', 'chd_enc8.pt')
    state_dict = torch.load(model_path)
    chd_enc.load_state_dict(state_dict)
    for param in chd_enc.parameters():
        param.requires_grad = False
    return chd_enc


def load_ec2vae_enc4():
    ec2vae_enc4 = Ec2VaeEncoder.create_4bar_encoder()
    model_path = os.path.join(PROJECT_PATH, 'pretrained_models', 'ec2vae_enc_4bar.pt')
    state_dict = torch.load(model_path)
    ec2vae_enc4.load_state_dict(state_dict)
    for param in ec2vae_enc4.parameters():
        param.requires_grad = False
    return ec2vae_enc4


def load_chd_enc2():
    chd_enc = PolydisChordEncoder.create_encoder()
    model_path = os.path.join(PROJECT_PATH, 'pretrained_models', 'polydis_chd_enc.pt')
    state_dict = torch.load(model_path)
    chd_enc.load_state_dict(state_dict)
    for param in chd_enc.parameters():
        param.requires_grad = False
    return chd_enc


def load_txt_enc2():
    txt_enc = PolydisTextureEncoder.create_encoder()
    model_path = os.path.join(PROJECT_PATH, 'pretrained_models', 'polydis_txt_enc.pt')
    state_dict = torch.load(model_path)
    txt_enc.load_state_dict(state_dict)
    for param in txt_enc.parameters():
        param.requires_grad = False
    return txt_enc
