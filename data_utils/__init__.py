from .read_pop909_data import load_train_and_valid_data, analyze_train_and_valid_datasets
from .pytorch_datasets import create_form_datasets, create_counterpoint_datasets, create_leadsheet_datasets, \
    create_accompaniment_datasets
from .pytorch_datasets.dataloaders import create_train_valid_dataloaders
from .pytorch_datasets.const import LANGUAGE_DATASET_PARAMS, AUTOREG_PARAMS


def load_datasets(mode, multi_phrase_label, random_pitch_aug, use_autoreg_cond, use_external_cond,
                  mask_background, load_first_n=None):
    train_data, valid_data = load_train_and_valid_data(multi_phrase_label, load_first_n)

    train_analyses, valid_analyses = analyze_train_and_valid_datasets(train_data, valid_data)

    if mode == 'frm':
        train_set, valid_set = create_form_datasets(
            train_analyses, valid_analyses, multi_phrase_label, random_pitch_aug
        )
    elif mode == 'ctp':
        train_set, valid_set = create_counterpoint_datasets(
            train_analyses, valid_analyses, use_autoreg_cond, use_external_cond,
            multi_phrase_label, random_pitch_aug, mask_background
        )
    elif mode == 'lsh':
        train_set, valid_set = create_leadsheet_datasets(
            train_analyses, valid_analyses, use_autoreg_cond, use_external_cond,
            multi_phrase_label, random_pitch_aug, mask_background
        )
    elif mode == 'acc':
        train_set, valid_set = create_accompaniment_datasets(
            train_analyses, valid_analyses, use_autoreg_cond, use_external_cond,
            multi_phrase_label, random_pitch_aug, mask_background
        )
    else:
        raise NotImplementedError
    return train_set, valid_set
