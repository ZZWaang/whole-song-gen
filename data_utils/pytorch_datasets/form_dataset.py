import numpy as np
import matplotlib.pyplot as plt
from .base_class import HierarchicalDatasetBase
from .const import LANGUAGE_DATASET_PARAMS, AUTOREG_PARAMS, SHIFT_HIGH_T, SHIFT_LOW_T, SHIFT_HIGH_V, SHIFT_LOW_V


class FormDataset(HierarchicalDatasetBase):

    def __init__(self, analyses, shift_high=0, shift_low=0, max_l=256, h=12, n_channels=8,
                 multi_phrase_label=False, random_pitch_aug=True):
        super(FormDataset, self).__init__(
            analyses, shift_high, shift_low, max_l, h, n_channels,
            use_autoreg_cond=False, use_external_cond=False,
            multi_phrase_label=multi_phrase_label, random_pitch_aug=random_pitch_aug, mask_background=False)

        assert max_l >= 210, "Some pieces may be longer than the current max_l."

        form_langs = [analysis['languages']['form'] for analysis in analyses]

        self.key_rolls = [form_lang['key_roll'] for form_lang in form_langs]
        self.phrase_rolls = [form_lang['phrase_roll'][:, :, np.newaxis] for form_lang in form_langs]

        self.lengths = np.array([roll.shape[1] for roll in self.key_rolls])

        self.start_ids_per_song = [np.zeros(1, dtype=np.int64) for _ in range(len(self.lengths))]

        self.indices = self._song_id_to_indices()

    def get_data_sample(self, song_id, start_id, shift):
        self.store_key(song_id, shift)
        self.store_phrase(song_id)

        img = self.lang_to_img(song_id, start_id, end_id=start_id + self.max_l, tgt_lgth=self.max_l)

        return img, None, None

    def lang_to_img(self, song_id, start_id, end_id, tgt_lgth=None):
        key_roll = self._key[:, start_id: end_id]  # (2, L, 12)
        phrase_roll = self._phrase[:, start_id: end_id]  # (6, L, 1)

        actual_l = self._key.shape[1]

        # to output image
        if tgt_lgth is None:
            tgt_lgth = end_id - start_id
        img = np.zeros((self.n_channels, tgt_lgth, self.h), dtype=np.float32)
        img[0: 2, 0: actual_l, 0: 12] = self._key
        img[2: 8, 0: actual_l] = self._phrase

        return img

    def show(self, item, show_img=True):
        sample = self[item][0]
        titles = ['key', 'phrase0-1', 'phrase2-3', 'phrase4-5']

        if show_img:
            fig, axs = plt.subplots(4, 1, figsize=(10, 30))
            for i in range(4):
                img = sample[2 * i: 2 * i + 2]
                img = np.pad(img, pad_width=((0, 1), (0, 0), (0, 0)), mode='constant')
                img = img.transpose((2, 1, 0))
                axs[i].imshow(img, origin='lower', aspect='auto')
                axs[i].title.set_text(titles[i])
            plt.show()


def create_form_datasets(train_analyses, valid_analyses, multi_phrase_label=False, random_pitch_aug=True):

    lang_params = LANGUAGE_DATASET_PARAMS['form']

    train_dataset = FormDataset(
        train_analyses, SHIFT_HIGH_T, SHIFT_LOW_T, lang_params['max_l'], lang_params['h'], lang_params['n_channel'],
        multi_phrase_label=multi_phrase_label, random_pitch_aug=random_pitch_aug
    )
    valid_dataset = FormDataset(
        valid_analyses, SHIFT_HIGH_V, SHIFT_LOW_V, lang_params['max_l'], lang_params['h'], lang_params['n_channel'],
        multi_phrase_label=multi_phrase_label, random_pitch_aug=random_pitch_aug
    )
    return train_dataset, valid_dataset
