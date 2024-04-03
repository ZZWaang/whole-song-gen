import numpy as np

SHIFT_HIGH_T = 5
SHIFT_LOW_T = -6
SHIFT_HIGH_V = 0
SHIFT_LOW_V = 0


LANGUAGE_DATASET_PARAMS = {
    'form': {'max_l': 256, 'h': 16, 'n_channel': 8, 'cur_channel': 8},
    'counterpoint': {'max_l': 128, 'h': 128, 'n_channel': 10, 'cur_channel': 2},
    'lead_sheet': {'max_l': 128, 'h': 128, 'n_channel': 12, 'cur_channel': 2},
    'accompaniment': {'max_l': 128, 'h': 128, 'n_channel': 14, 'cur_channel': 2},
}

AUTOREG_PARAMS = {
    'counterpoint': {
        'autoreg_seg_lgth': 8, 'max_n_autoreg': 3, 'n_autoreg_prob': np.array([0.1, 0.1, 0.2, 0.6]),
        'seg_pad_unit': 4, 'autoreg_max_l': 108,
    },
    'lead_sheet': {
        'autoreg_seg_lgth': 4, 'max_n_autoreg': 2, 'n_autoreg_prob': np.array([0.1, 0.2, 0.7]),
        'seg_pad_unit': 4, 'autoreg_max_l': 136
    },
    'accompaniment': {
        'autoreg_seg_lgth': 4, 'max_n_autoreg': 2, 'n_autoreg_prob': np.array([0.1, 0.2, 0.7]),
        'seg_pad_unit': 4, 'autoreg_max_l': 136
    }
}

