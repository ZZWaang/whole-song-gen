import os
import numpy as np
from tqdm import tqdm
from data_utils.utils.read_file import read_data
from data_utils.utils.song_data_structure import McpaMusic
from data_utils.utils.song_analyzer import LanguageExtractor
from typing import List
from .train_valid_split import load_split_file


TRIPLE_METER_SONG = [
    34, 62, 102, 107, 152, 173, 176, 203, 215, 231, 254, 280, 307, 328, 369,
    584, 592, 653, 654, 662, 744, 749, 756, 770, 799, 843, 869, 872, 887
]


PROJECT_PATH = os.path.join(os.path.dirname(__file__), '..')

DATASET_PATH = os.path.join(PROJECT_PATH, 'data', 'pop909_w_structure_label')
ACC_DATASET_PATH = os.path.join(PROJECT_PATH, 'data', 'matched_pop909_acc')

LABEL_SOURCE = np.load(os.path.join(PROJECT_PATH, 'data',
                                    'pop909_w_structure_label',
                                    'label_source.npy'))

SPLIT_FILE_PATH = os.path.join(PROJECT_PATH, 'data', 'pop909_split', 'split.npz')


def read_pop909_dataset(song_ids=None, label_fns=None, desc_dataset=None):
    """If label_fn is None, use default the selected label file in LABEL_SOURCE"""

    dataset = []

    song_ids = [si for si in range(1, 910)] if song_ids is None else song_ids

    for idx, i in enumerate(tqdm(song_ids, desc=None if desc_dataset is None else f'Loading {desc_dataset}')):
        # which human label file to use
        label = LABEL_SOURCE[i - 1] if label_fns is None else label_fns[idx]

        num_beat_per_measure = 3 if i in TRIPLE_METER_SONG else 4

        song_name = str(i).zfill(3)  # e.g., '001'

        data_fn = os.path.join(DATASET_PATH, song_name)  # data folder of the song

        acc_fn = os.path.join(ACC_DATASET_PATH, song_name)

        song_data = read_data(data_fn, acc_fn, num_beat_per_measure=num_beat_per_measure, num_step_per_beat=4,
                              clean_chord_unit=num_beat_per_measure, song_name=song_name, label=label)

        dataset.append(song_data)

    return dataset


def read_pop909_dataset_with_multi_phrase_labels(song_ids=None, desc_dataset=None):
    dataset = []

    song_ids = [si for si in range(1, 910)] if song_ids is None else song_ids

    for label in [1, 2]:
        label_fns = [label] * len(song_ids)
        dataset += read_pop909_dataset(song_ids, label_fns, desc_dataset=desc_dataset + f'-label-{label}')
    return dataset


def analyze_pop909_dataset(dataset: List[McpaMusic], desc_dataset=None):
    hie_lang_dataset = []
    for song in tqdm(dataset, desc=None if desc_dataset is None else f'Analyzing {desc_dataset}'):
        lang_extractor = LanguageExtractor(song)
        hie_lang = lang_extractor.analyze_for_training()
        hie_lang_dataset.append(hie_lang)
    return hie_lang_dataset


def load_train_and_valid_data(use_multi_phrase_label=False, load_first_n=None):
    train_dataset = []
    valid_dataset = []

    train_ids, valid_ids = load_split_file(SPLIT_FILE_PATH)

    if use_multi_phrase_label:
        train_dataset = read_pop909_dataset_with_multi_phrase_labels(
            train_ids[0: load_first_n] + 1, desc_dataset='train set (multi-label)'
        )

        valid_dataset = read_pop909_dataset_with_multi_phrase_labels(
            valid_ids[0: load_first_n] + 1, desc_dataset='valid set (multi-label)'
        )
    else:
        train_dataset = read_pop909_dataset(train_ids[0: load_first_n] + 1, desc_dataset='train set')
        valid_dataset = read_pop909_dataset(valid_ids[0: load_first_n] + 1, desc_dataset='valid set')

    return train_dataset, valid_dataset


def analyze_train_and_valid_datasets(train_set, valid_set):
    return analyze_pop909_dataset(train_set, 'train set'), analyze_pop909_dataset(valid_set, 'valid set')
