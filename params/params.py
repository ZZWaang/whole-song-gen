from .attrdict import AttrDict


params_frm = AttrDict(
    # Training params
    batch_size=16,
    max_epoch=500,
    learning_rate=5e-5,
    max_grad_norm=10,
    fp16=True,

    # unet
    in_channels=8,
    out_channels=8,
    channels=64,
    attention_levels=[2, 3],
    n_res_blocks=2,
    channel_multipliers=[1, 2, 4, 4],
    n_heads=4,
    tf_layers=1,
    d_cond=12,

    # ldm
    linear_start=0.00085,
    linear_end=0.0120,
    n_steps=1000,
    latent_scaling_factor=0.18215
)


params_ctp = AttrDict(
    # Training params
    batch_size=16,
    max_epoch=500,
    learning_rate=5e-5,
    max_grad_norm=10,
    fp16=True,

    # unet
    in_channels=10,
    out_channels=2,
    channels=64,
    attention_levels=[2, 3],
    n_res_blocks=2,
    channel_multipliers=[1, 2, 4, 4],
    n_heads=4,
    tf_layers=1,
    d_cond=128,

    # ldm
    linear_start=0.00085,
    linear_end=0.0120,
    n_steps=1000,
    latent_scaling_factor=0.18215
)


params_lsh = AttrDict(
    # Training params
    batch_size=16,
    max_epoch=500,
    learning_rate=5e-5,
    max_grad_norm=10,
    fp16=True,

    # unet
    in_channels=12,
    out_channels=2,
    channels=64,
    attention_levels=[2, 3],
    n_res_blocks=2,
    channel_multipliers=[1, 2, 4, 4],
    n_heads=4,
    tf_layers=1,
    d_cond=256,

    # ldm
    linear_start=0.00085,
    linear_end=0.0120,
    n_steps=1000,
    latent_scaling_factor=0.18215
)


params_acc = AttrDict(
    # Training params
    batch_size=16,
    max_epoch=500,
    learning_rate=5e-5,
    max_grad_norm=10,
    fp16=True,

    # unet
    in_channels=14,
    out_channels=2,
    channels=64,
    attention_levels=[2, 3],
    n_res_blocks=2,
    channel_multipliers=[1, 2, 4, 4],
    n_heads=4,
    tf_layers=1,
    d_cond=256,

    # ldm
    linear_start=0.00085,
    linear_end=0.0120,
    n_steps=1000,
    latent_scaling_factor=0.18215
)
