import matplotlib.pyplot as plt
from .base_class import *
from .const import LANGUAGE_DATASET_PARAMS, AUTOREG_PARAMS, SHIFT_HIGH_T, SHIFT_LOW_T, SHIFT_HIGH_V, SHIFT_LOW_V


def mel_to_ec2vae_pr(mel):
    # mel: (2, max_l, 128)

    pr = np.zeros((mel.shape[1], 130), dtype=np.float32)

    pr[:, 0: 128] = mel[0]

    is_sustain = mel[1].sum(-1) > 0  # (max_l,)
    is_rest = mel.sum(0).sum(-1) == 0  # (max_l, )

    pr[is_sustain, 128] = 1
    pr[is_rest, 129] = 1

    return pr


class LeadSheetDataset(HierarchicalDatasetBase):
    def __init__(self, analyses, shift_high=0, shift_low=0, max_l=128, h=128, n_channels=10,
                 autoreg_seg_lgth=8, max_n_autoreg=3, n_autoreg_prob=np.array([0.1, 0.1, 0.2, 0.6]),
                 seg_pad_unit=4, autoreg_max_l=108,
                 use_autoreg_cond=True, use_external_cond=False, multi_phrase_label=False,
                 random_pitch_aug=True, mask_background=False):

        super(LeadSheetDataset, self).__init__(
            analyses, shift_high, shift_low, max_l, h, n_channels,
            autoreg_seg_lgth, max_n_autoreg, n_autoreg_prob, seg_pad_unit, autoreg_max_l,
            use_autoreg_cond, use_external_cond, multi_phrase_label, random_pitch_aug, mask_background)

        form_langs = [analysis['languages']['form'] for analysis in analyses]
        ctpt_langs = [analysis['languages']['counterpoint'] for analysis in analyses]
        ldsht_langs = [analysis['languages']['lead_sheet'] for analysis in analyses]

        self.key_rolls = [form_lang['key_roll'] for form_lang in form_langs]
        self.expand_key_rolls()

        self.phrase_rolls = [form_lang['phrase_roll'][:, :, np.newaxis] for form_lang in form_langs]
        self.expand_phrase_rolls()

        self.red_mel_rolls = [ctpt_lang['red_mel_roll'] for ctpt_lang in ctpt_langs]
        self.expand_red_mel_rolls()

        self.red_chd_rolls = [ctpt_lang['red_chd_roll'] for ctpt_lang in ctpt_langs]
        self.expand_red_chd_rolls()

        self.mel_rolls = [ldsht_lang['mel_roll'] for ldsht_lang in ldsht_langs]
        self.chd_rolls = [ldsht_lang['chd_roll'] for ldsht_lang in ldsht_langs]
        self.expand_chd_rolls()

        self.lengths = np.array([mel.shape[1] for mel in self.mel_rolls])

        self.start_ids_per_song = [np.arange(0, lgth - self.max_l // 2, nspb * nbpm, dtype=np.int64)
                                   for lgth, nbpm, nspb in zip(self.lengths, self.nbpms, self.nspbs)]

        self.indices = self._song_id_to_indices()

    def expand_key_rolls(self):
        self.key_rolls = [expand_roll(roll, nbpm * nspb)
                          for roll, nbpm, nspb in zip(self.key_rolls, self.nbpms, self.nspbs)]

    def expand_phrase_rolls(self):
        self.phrase_rolls = [expand_roll(roll, nbpm * nspb)
                             for roll, nbpm, nspb in zip(self.phrase_rolls, self.nbpms, self.nspbs)]

    def expand_red_chd_rolls(self):
        self.red_chd_rolls = [expand_roll(roll, nspb, contain_onset=True)
                              for roll, nspb in zip(self.red_chd_rolls, self.nspbs)]

    def expand_chd_rolls(self):
        self.chd_rolls = [expand_roll(roll, nspb, contain_onset=True)
                          for roll, nspb in zip(self.chd_rolls, self.nspbs)]

    def expand_red_mel_rolls(self):
        self.red_mel_rolls = [expand_roll(roll, nspb, contain_onset=True)
                              for roll, nspb in zip(self.red_mel_rolls, self.nspbs)]

    def get_data_sample(self, song_id, start_id, shift):
        nbpm, nspb = self.nbpms[song_id], self.nspbs[song_id]

        pitch_shift = compute_pitch_shift_value(shift, self.min_mel_pitches[song_id], self.max_mel_pitches[song_id])

        self.store_key(song_id, pitch_shift)
        self.store_phrase(song_id)
        self.store_red_mel(song_id, pitch_shift)
        self.store_red_chd(song_id, pitch_shift)
        self.store_mel(song_id, pitch_shift)
        self.store_chd(song_id, pitch_shift)

        img = self.lang_to_img(song_id, start_id, end_id=start_id + self.max_l, tgt_lgth=self.max_l)

        # prepare for the autoreg condition
        if self.use_autoreg_cond:
            autoreg_cond = self.get_autoreg_cond(song_id, start_id, nbpm * nspb)
        else:
            autoreg_cond = None

        # prepare for the external condition
        if self.use_external_cond:
            external_cond = self.get_external_cond(start_id)
        else:
            external_cond = None

        # randomly mask background
        if self.mask_background and np.random.random() > 0.8:
            img[2:] = -1

        return img, autoreg_cond, external_cond

    def lang_to_img(self, song_id, start_id, end_id, tgt_lgth=None):
        key_roll = self._key[:, start_id: end_id]  # (2, L, 12)
        phrase_roll = self._phrase[:, start_id: end_id]  # (6, L, 1)
        red_mel_roll = self._red_mel[:, start_id: end_id]  # (2, L, 128)
        red_chd_roll = self._red_chd[:, start_id: end_id]  # (6, L, 12)
        mel_roll = self._mel[:, start_id: end_id]
        chd_roll = self._chd[:, start_id: end_id]

        actual_l = key_roll.shape[1]

        # to output image
        if tgt_lgth is None:
            tgt_lgth = end_id - start_id
        img = np.zeros((self.n_channels, tgt_lgth, 132), dtype=np.float32)
        img[0: 2, 0: actual_l, 0: 128] = mel_roll
        img[0: 2, 0: actual_l, 36: 48] = chd_roll[2: 4]
        img[0: 2, 0: actual_l, 24: 36] = chd_roll[4: 6]

        img[2: 4, 0: actual_l, 0: 128] = red_mel_roll
        img[2: 4, 0: actual_l, 36: 48] = red_chd_roll[2: 4]
        img[2: 4, 0: actual_l, 24: 36] = red_chd_roll[4: 6]

        img[6: 12, 0: actual_l] = phrase_roll

        img = img.reshape((self.n_channels, tgt_lgth, 11, 12))
        img[4: 6, 0: actual_l] = key_roll[:, :, np.newaxis]
        img = img.reshape((self.n_channels, tgt_lgth, 132))
        return img[:, :, 0: self.h]

    def get_external_cond(self, start_id):
        external_cond = np.zeros((self.max_l, 142), dtype=np.float32)

        chroma = self._chd[:, start_id: start_id + self.max_l][2: 4].sum(0)  # (max_l, 12)
        mel = self._mel[:, start_id: start_id + self.max_l]

        actual_l = mel.shape[1]

        external_cond[0: actual_l, 130:] = chroma
        external_cond[0: actual_l, 0: 130] = mel_to_ec2vae_pr(mel)
        return external_cond

    def show(self, item, show_img=True):
        data, autoreg, external = self[item]

        titles = ['mel+chd', 'mel+rough_chd', 'key', 'phrase0-1', 'phrase2-3', 'phrase4-5']

        if show_img:
            if self.use_external_cond:
                fig, axs = plt.subplots(6, 3, figsize=(30, 40))
            else:
                fig, axs = plt.subplots(6, 2, figsize=(20, 40))
            for i in range(6):
                img = data[2 * i: 2 * i + 2]
                img = np.pad(img, pad_width=((0, 1), (0, 0), (0, 0)), mode='constant')
                img[2][img[0] < 0] = 1
                img[img < 0] = 0
                img = img.transpose((2, 1, 0))
                axs[i, 0].imshow(img, origin='lower', aspect='auto')
                axs[i, 0].title.set_text(titles[i])

                autoreg_img = autoreg[2 * i: 2 * i + 2]
                autoreg_img = np.pad(autoreg_img, pad_width=((0, 1), (0, 0), (0, 0)), mode='constant')
                autoreg_img[2][autoreg_img[0] < 0] = 1
                autoreg_img[autoreg_img < 0] = 0
                autoreg_img = autoreg_img.transpose((2, 1, 0))

                axs[i, 1].imshow(autoreg_img, origin='lower', aspect='auto')

            if self.use_external_cond:
                axs[0, 2].imshow(external.T, origin='lower', aspect='auto')
            plt.show()


def create_leadsheet_datasets(train_analyses, valid_analyses, use_autoreg_cond=True, use_external_cond=False,
                              multi_phrase_label=False, random_pitch_aug=True, mask_background=True):

    lang_params = LANGUAGE_DATASET_PARAMS['lead_sheet']
    autoreg_params = AUTOREG_PARAMS['lead_sheet']

    train_dataset = LeadSheetDataset(
        train_analyses, SHIFT_HIGH_T, SHIFT_LOW_T, lang_params['max_l'], lang_params['h'], lang_params['n_channel'],
        autoreg_seg_lgth=autoreg_params['autoreg_seg_lgth'], max_n_autoreg=autoreg_params['max_n_autoreg'],
        n_autoreg_prob=autoreg_params['n_autoreg_prob'], seg_pad_unit=autoreg_params['seg_pad_unit'],
        autoreg_max_l=autoreg_params['autoreg_max_l'], use_autoreg_cond=use_autoreg_cond,
        use_external_cond=use_external_cond, multi_phrase_label=multi_phrase_label, random_pitch_aug=random_pitch_aug,
        mask_background=mask_background
    )
    valid_dataset = LeadSheetDataset(
        valid_analyses, SHIFT_HIGH_V, SHIFT_LOW_V, lang_params['max_l'], lang_params['h'], lang_params['n_channel'],
        autoreg_seg_lgth=autoreg_params['autoreg_seg_lgth'], max_n_autoreg=autoreg_params['max_n_autoreg'],
        n_autoreg_prob=autoreg_params['n_autoreg_prob'], seg_pad_unit=autoreg_params['seg_pad_unit'],
        autoreg_max_l=autoreg_params['autoreg_max_l'], use_autoreg_cond=use_autoreg_cond,
        use_external_cond=use_external_cond, multi_phrase_label=multi_phrase_label, random_pitch_aug=random_pitch_aug,
        mask_background=mask_background
    )
    return train_dataset, valid_dataset
