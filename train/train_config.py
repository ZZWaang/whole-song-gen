from . import *
from data_utils import load_datasets, create_train_valid_dataloaders
from model import init_ldm_model, init_diff_pro_sdf


class LdmTrainConfig(TrainConfig):

    def __init__(self, params, output_dir, mode, use_autoreg_cond, use_external_cond,
                 mask_background, multi_phrase_label, random_pitch_aug, debug_mode=False) -> None:
        super().__init__(params, None, output_dir)
        self.debug_mode = debug_mode
        self.use_autoreg_cond = use_autoreg_cond
        self.use_external_cond = use_external_cond
        self.mask_background = mask_background
        self.multi_phrase_label = multi_phrase_label
        self.random_pitch_aug = random_pitch_aug

        # create model
        self.ldm_model = init_ldm_model(mode, use_autoreg_cond, use_external_cond, params, debug_mode)
        self.model = init_diff_pro_sdf(self.ldm_model, params, self.device)

        # Create dataloader
        load_first_n = 10 if self.debug_mode else None
        train_set, valid_set = load_datasets(
            mode, multi_phrase_label, random_pitch_aug, use_autoreg_cond, use_external_cond,
            mask_background, load_first_n
        )
        self.train_dl, self.val_dl = create_train_valid_dataloaders(params.batch_size, train_set, valid_set)

        # Create optimizer
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=params.learning_rate
        )
