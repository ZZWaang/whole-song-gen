import numpy as np
from torch.utils.data import Dataset, DataLoader


def compute_pitch_shift_value(shift, min_pitch, max_pitch):
    # find s' = shift + 12k such that min_pitch + s' >= 48 and max_pitch + s' < 108
    min_pitch += shift
    max_pitch += shift

    min_allowed_thresh = 48
    max_allowed_thresh = 98

    down_octave = (min_allowed_thresh - min_pitch) // 12 + 1  # if it's 50 then 0
    up_octave = (max_allowed_thresh - max_pitch) // 12  # if it's 100 then -1, if it's 96 then 0

    possible_octaves = np.arange(down_octave, up_octave + 1)
    probs = np.array([5 if po == 0 else 1 for po in possible_octaves], dtype=np.float64)
    assert len(probs) >= 1
    probs /= probs.sum()
    octave = np.random.choice(possible_octaves, p=probs)

    return shift + octave * 12


def select_prev_slices(t_m, phrase_roll_m, phrase_type_score, tgt_lgth_m, n_seg):
    """
    t_m: current time step (in measure)
    phrase_roll_m: shape: (6, L) (in measure)
    phrase_type_score: (6,) importance of the 6 phrase type at current time step
    tgt_lgth_m: segment length in measure
    n_seg: number of segments to select

    returns a list of slices of length <= n_seg
    """

    slices = []

    if t_m != 0:
        measure_importance = phrase_type_score[phrase_roll_m.argmax(0)][0: t_m]  # (t_m, )

        for i in range(n_seg):
            # compute accumulate distance
            acc_measure_importance = np.array([measure_importance[i: i + tgt_lgth_m].sum()
                                               for i in range(len(measure_importance))])
            start_measure = acc_measure_importance.argmax()

            if acc_measure_importance[start_measure] < 0:  # the current segment has been previously selected.
                break

            measure_importance[start_measure: start_measure + tgt_lgth_m] = -1

            slices.append((start_measure, min(t_m, start_measure + tgt_lgth_m)))

    return slices


def expand_roll(roll, unit=4, contain_onset=False):
    # roll: (Channel, T, H) -> (Channel, T * unit, H)
    n_channel, length, height = roll.shape

    expanded_roll = roll.repeat(unit, axis=1)
    if contain_onset:
        expanded_roll = expanded_roll.reshape((n_channel, length, unit, height))
        expanded_roll[1::2, :, 1:] = np.maximum(expanded_roll[::2, :, 1:], expanded_roll[1::2, :, 1:])

        expanded_roll[::2, :, 1:] = 0
        expanded_roll = expanded_roll.reshape((n_channel, length * unit, height))
    return expanded_roll


class HierarchicalDatasetBase(Dataset):

    def __init__(self, analyses, shift_high=-6, shift_low=5, max_l=128, h=128, n_channels=None,
                 autoreg_seg_lgth=None, max_n_autoreg=None, n_autoreg_prob=None,
                 seg_pad_unit=None, autoreg_max_l=None,
                 use_autoreg_cond=False, use_external_cond=False, multi_phrase_label=False,
                 random_pitch_aug=True, mask_background=True):
        super(HierarchicalDatasetBase, self).__init__()

        # dataset setting
        self.multi_phrase_label = multi_phrase_label  # use all phrase label. Dataset size will be doubled.
        self.random_pitch_aug = random_pitch_aug  # sample a pitch shift (True) or extend the dataset 12 times (False).
        self.use_autoreg_cond = use_autoreg_cond
        self.use_external_cond = use_external_cond
        self.mask_background = mask_background

        self.shift_high = shift_high
        self.shift_low = shift_low
        self.max_l = max_l
        self.h = h
        self.n_channels = n_channels

        self.min_mel_pitches = [analysis['min_mel_pitch'] for analysis in analyses]
        self.max_mel_pitches = [analysis['max_mel_pitch'] for analysis in analyses]

        self.nbpms = [analysis['nbpm'] for analysis in analyses]
        self.nspbs = [analysis['nspb'] for analysis in analyses]
        self.song_names = [analysis['name'] for analysis in analyses]

        self.lengths = None  # an array: [num_time_step_song_0, num_time_step_song_1, num_time_step_song_2, ...]
        self.start_ids_per_song = None  # a list: [[seg0_start, seg1_start, ...], [seg0_start, seg1_start, ...], ...]
        self.indices = None  # a list of all possible (song_id, segment_id) pairs)

        self._key = None
        self._phrase = None
        self._red_mel = None
        self._red_chd = None
        self._mel = None
        self._chd = None
        self._acc = None

        self.phrase_rolls = None
        self.key_rolls = None
        self.red_mel_rolls = None
        self.red_chd_rolls = None
        self.mel_rolls = None
        self.chd_rolls = None
        self.acc_rolls = None

        if self.use_autoreg_cond:
            assert max_n_autoreg + 1 == len(n_autoreg_prob), "max_n_autoreg + 1 == len(n_autoreg_prob)."
        self.autoreg_seg_lgth = autoreg_seg_lgth
        self.max_n_autoreg = max_n_autoreg
        self.n_autoreg_prob = n_autoreg_prob
        self.seg_pad_unit = seg_pad_unit
        self.autoreg_max_l = autoreg_max_l

    def _song_id_to_indices(self):
        assert self.start_ids_per_song is not None, "The attribute start_ids_per_song must be filled first."
        return np.concatenate([
            np.stack([np.ones(len(start_ids), dtype=np.int64) * song_id, start_ids], -1)
            for song_id, start_ids in enumerate(self.start_ids_per_song)
        ], 0)

    def __len__(self):
        if self.multi_phrase_label:
            assert len(self.indices) % 2 == 0, "len(self.indices) must be even when self.random_label=True."
            num_segment = len(self.indices) // 2
        else:
            num_segment = len(self.indices)

        num_aug_pitch = 1 if self.random_pitch_aug else self.shift_high - self.shift_low + 1

        return num_segment * num_aug_pitch

    def __getitem__(self, item):
        if self.multi_phrase_label:
            if np.random.random() > 0.5:
                item = len(self) + item

        if self.random_pitch_aug:
            segment_id = item
            pitch_shift = np.random.randint(self.shift_high - self.shift_low + 1) + self.shift_low
        else:
            segment_id = item // (self.shift_high - self.shift_low + 1)
            pitch_shift = item % (self.shift_high - self.shift_low + 1) + self.shift_low

        song_id, start_id = self.indices[segment_id]

        return self.get_data_sample(song_id, start_id, pitch_shift)

    def lang_to_img(self, song_id, start_id, end_id, tgt_lgth):
        raise NotImplementedError

    def get_data_sample(self, song_id, start_id, shift):
        raise NotImplementedError

    def select_autoreg_slices(self, start_id, scale_unit):
        # compute current measure_id of start_id

        t_m = start_id // scale_unit

        # retrieve phrase_roll in measure
        phrase_roll_m = self._phrase[:, ::scale_unit, 0]

        # compute phrase type importance at start_id
        phrase_type_score = self._phrase[:, start_id: start_id + self.max_l, 0].sum(-1)  # (6, )

        # sample number of segments
        n_seg = np.random.choice(np.arange(0, self.max_n_autoreg + 1), p=self.n_autoreg_prob)

        autoreg_slices = select_prev_slices(t_m, phrase_roll_m, phrase_type_score, self.autoreg_seg_lgth, n_seg)

        return autoreg_slices

    def get_autoreg_cond(self, song_id, start_id, scale_unit):

        cond_img = -np.ones((self.n_channels, self.autoreg_max_l, self.h), dtype=np.float32)

        autoreg_slices = self.select_autoreg_slices(start_id, scale_unit)

        scale_unit_ = int(2 ** np.ceil(np.log2(scale_unit)))
        seg_lgth_unit = self.autoreg_seg_lgth * scale_unit_ + self.seg_pad_unit

        for i, slc in enumerate(autoreg_slices):
            seg_start, seg_end = slc[0] * scale_unit, slc[1] * scale_unit
            tgt_l = self.autoreg_seg_lgth * scale_unit

            autoreg_img = self.lang_to_img(song_id, seg_start, seg_end, tgt_l)

            cond_img[:, i * seg_lgth_unit: i * seg_lgth_unit + tgt_l] = autoreg_img

        return cond_img

    def store_key(self, song_id, shift):
        if self.key_rolls is not None:
            key_roll = self.key_rolls[song_id]
            self._key = np.roll(key_roll, shift=shift, axis=-1)

    def store_phrase(self, song_id):
        if self.phrase_rolls is not None:
            self._phrase = self.phrase_rolls[song_id]

    def store_red_mel(self, song_id, shift):
        if self.red_mel_rolls is not None:
            red_mel_roll = self.red_mel_rolls[song_id]
            self._red_mel = np.roll(red_mel_roll, shift=shift, axis=-1)

    def store_red_chd(self, song_id, shift):
        if self.red_chd_rolls is not None:
            red_chd_roll = self.red_chd_rolls[song_id]
            self._red_chd = np.roll(red_chd_roll, shift=shift, axis=-1)

    def store_mel(self, song_id, shift):
        if self.mel_rolls is not None:
            mel_roll = self.mel_rolls[song_id]
            self._mel = np.roll(mel_roll, shift=shift, axis=-1)

    def store_chd(self, song_id, shift):
        if self.chd_rolls is not None:
            chd_roll = self.chd_rolls[song_id]
            self._chd = np.roll(chd_roll, shift=shift, axis=-1)

    def store_acc(self, song_id, shift):
        if self.acc_rolls is not None:
            acc_roll = self.acc_rolls[song_id]
            self._acc = np.roll(acc_roll, shift=shift, axis=-1)

    def get_external_cond(self, start_id):
        raise NotImplementedError
