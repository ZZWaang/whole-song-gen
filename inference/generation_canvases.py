from data_utils import LANGUAGE_DATASET_PARAMS, AUTOREG_PARAMS
from data_utils.pytorch_datasets.base_class import select_prev_slices
import numpy as np


class GenerationCanvasBase:

    n_channels = None
    cur_channels = None
    max_l = None
    autoreg_max_l = None
    h = None
    phrase_channel = slice(-6, None)

    max_n_autoreg = None
    n_autoreg_prob = None
    autoreg_seg_lgth = None
    seg_pad_unit = None

    def __init__(self, generation, units, lengths, mask, random_n_autoreg=False):
        """generation area: """
        self.generation = generation
        self.mask = mask
        self.units = units
        self.lengths = lengths

        self.batch_size = generation.shape[0]
        self.random_n_autoreg = random_n_autoreg
        self.external_cond = None

    def __len__(self):
        return self.batch_size

    def _get_phrase(self, item):
        return self.generation[item, self.phrase_channel]

    def select_autoreg_slices(self, song_id, start_id, scale_unit):
        # compute current measure_id of start_id
        t_m = start_id // scale_unit

        # retrieve phrase_roll in measure
        phrase_roll_m = self._get_phrase(song_id)[:, ::scale_unit, 0]

        # compute phrase type importance at start_id
        phrase_type_score = self._get_phrase(song_id)[:, start_id: start_id + self.max_l, 0].sum(-1)  # (6, )

        # sample number of segments
        if self.random_n_autoreg:
            n_seg = np.random.choice(np.arange(0, self.max_n_autoreg + 1), p=self.n_autoreg_prob)
        else:
            n_seg = self.max_n_autoreg

        autoreg_slices = select_prev_slices(t_m, phrase_roll_m, phrase_type_score, self.autoreg_seg_lgth, n_seg)

        return autoreg_slices

    def get_autoreg_cond(self, song_id, start_id, scale_unit):

        cond_img = -np.ones((self.n_channels, self.autoreg_max_l, self.h), dtype=np.float32)

        autoreg_slices = self.select_autoreg_slices(song_id, start_id, scale_unit)

        scale_unit_ = int(2 ** np.ceil(np.log2(scale_unit)))

        seg_lgth_unit = self.autoreg_seg_lgth * scale_unit_ + self.seg_pad_unit

        for i, slc in enumerate(autoreg_slices):
            seg_start, seg_end = slc[0] * scale_unit, slc[1] * scale_unit

            actual_seg_l = seg_end - seg_start

            tgt_l = self.autoreg_seg_lgth * scale_unit

            autoreg_img = self.generation[song_id, :, seg_start: seg_end]

            cond_img[:, i * seg_lgth_unit: i * seg_lgth_unit + actual_seg_l] = autoreg_img
            cond_img[:, i * seg_lgth_unit + actual_seg_l: i * seg_lgth_unit + tgt_l] = 0.

        return cond_img

    def get_batch_autoreg_cond(self, start_id, sel_song_ids=None):
        if self.cur_channels == self.n_channels:
            return None
        start_id = [start_id] * self.batch_size if isinstance(start_id, int) else start_id
        cond_img = -np.ones((self.batch_size, self.n_channels, self.autoreg_max_l, self.h), dtype=np.float32)
        for i in range(len(self)):
            scale_unit = self.units[i]
            cond_img[i] = self.get_autoreg_cond(i, start_id[i], scale_unit)
        return cond_img[sel_song_ids]

    def get_batch_background_cond(self, start_id, sel_song_ids=None):
        if self.cur_channels == self.n_channels:
            return None
        if isinstance(start_id, int):
            background_cond = self.generation[:, self.cur_channels:, start_id: start_id + self.max_l]
        else:
            background_cond = [
                self.generation[i, self.cur_channels:, start_id[i]: start_id[i] + self.max_l]
                for i in range(len(self))
            ]
            background_cond = np.stack(background_cond, axis=0)

        return background_cond[sel_song_ids]

    def get_batch_mask(self, start_id, sel_song_ids=None):
        if isinstance(start_id, int):
            mask = self.mask[:, :, start_id: start_id + self.max_l]
        else:
            mask = [
                self.mask[i, :, start_id[i]: start_id[i] + self.max_l]
                for i in range(len(self))
            ]
            mask = np.stack(mask, axis=0)
        return mask[sel_song_ids]

    def get_batch_cur_level(self, start_id, sel_song_ids=None):
        if isinstance(start_id, int):
            cur_level = self.generation[:, 0: self.cur_channels, start_id: start_id + self.max_l]
        else:
            cur_level = [
                self.generation[i, 0: self.cur_channels, start_id[i]: start_id[i] + self.max_l]
                for i in range(len(self))
            ]
            cur_level = np.stack(cur_level, axis=0)

        return cur_level[sel_song_ids]

    def get_batch_external_cond(self):
        """Not Implemented."""
        if self.external_cond is None:
            return None
        else:
            raise NotImplementedError

    def check_is_generated(self, start_id, end_id):
        start_id = [start_id] * self.batch_size if isinstance(start_id, int) else start_id
        end_id = [end_id] * self.batch_size if isinstance(end_id, int) else end_id
        return np.array([(self.mask[i, :, s: e] == 1).all() for i, (s, e) in enumerate(zip(start_id, end_id))],
                        dtype=np.bool)

    def write_generation(self, new_generation, start_id, end_id, sel_song_ids=None, quantize=True):
        if quantize:
            new_generation[:, 0: self.cur_channels][new_generation[:, 0: self.cur_channels] > 0.5] = 1.
            new_generation[:, 0: self.cur_channels][new_generation[:, 0: self.cur_channels] < 0.9] = 0.

        if sel_song_ids is None:
            sel_song_ids = np.arange(self.batch_size)
        else:
            sel_song_ids = np.where(sel_song_ids)[0]

        start_id = [start_id] * len(sel_song_ids) if isinstance(start_id, int) else start_id
        end_id = [end_id] * len(sel_song_ids) if isinstance(end_id, int) else end_id
        for i, (s, e) in enumerate(zip(start_id, end_id)):
            song_id = sel_song_ids[i]
            self.generation[song_id, 0: self.cur_channels, s: e] = new_generation[i, :, 0: e - s]
            self.mask[song_id, 0: self.cur_channels, s: e] = 1

    def to_output(self):
        return [self.generation[i, :, 0: self.lengths[i]] for i in range(len(self))]


class FormCanvas(GenerationCanvasBase):

    n_channels = LANGUAGE_DATASET_PARAMS['form']['n_channel']
    cur_channels = LANGUAGE_DATASET_PARAMS['form']['cur_channel']
    max_l = LANGUAGE_DATASET_PARAMS['form']['max_l']
    h = LANGUAGE_DATASET_PARAMS['form']['h']


class CounterpointCanvas(GenerationCanvasBase):

    n_channels = LANGUAGE_DATASET_PARAMS['counterpoint']['n_channel']
    cur_channels = LANGUAGE_DATASET_PARAMS['counterpoint']['cur_channel']
    max_l = LANGUAGE_DATASET_PARAMS['counterpoint']['max_l']
    h = LANGUAGE_DATASET_PARAMS['counterpoint']['h']

    autoreg_max_l = AUTOREG_PARAMS['counterpoint']['autoreg_max_l']
    max_n_autoreg = AUTOREG_PARAMS['counterpoint']['max_n_autoreg']
    n_autoreg_prob = AUTOREG_PARAMS['counterpoint']['n_autoreg_prob']
    autoreg_seg_lgth = AUTOREG_PARAMS['counterpoint']['autoreg_seg_lgth']
    seg_pad_unit = AUTOREG_PARAMS['counterpoint']['seg_pad_unit']


class LeadSheetCanvas(GenerationCanvasBase):

    n_channels = LANGUAGE_DATASET_PARAMS['lead_sheet']['n_channel']
    cur_channels = LANGUAGE_DATASET_PARAMS['lead_sheet']['cur_channel']
    max_l = LANGUAGE_DATASET_PARAMS['lead_sheet']['max_l']
    h = LANGUAGE_DATASET_PARAMS['lead_sheet']['h']

    autoreg_max_l = AUTOREG_PARAMS['lead_sheet']['autoreg_max_l']
    max_n_autoreg = AUTOREG_PARAMS['lead_sheet']['max_n_autoreg']
    n_autoreg_prob = AUTOREG_PARAMS['lead_sheet']['n_autoreg_prob']
    autoreg_seg_lgth = AUTOREG_PARAMS['lead_sheet']['autoreg_seg_lgth']
    seg_pad_unit = AUTOREG_PARAMS['lead_sheet']['seg_pad_unit']


class AccompanimentCanvas(GenerationCanvasBase):

    n_channels = LANGUAGE_DATASET_PARAMS['accompaniment']['n_channel']
    cur_channels = LANGUAGE_DATASET_PARAMS['accompaniment']['cur_channel']
    max_l = LANGUAGE_DATASET_PARAMS['accompaniment']['max_l']
    h = LANGUAGE_DATASET_PARAMS['accompaniment']['h']

    autoreg_max_l = AUTOREG_PARAMS['accompaniment']['autoreg_max_l']
    max_n_autoreg = AUTOREG_PARAMS['accompaniment']['max_n_autoreg']
    n_autoreg_prob = AUTOREG_PARAMS['accompaniment']['n_autoreg_prob']
    autoreg_seg_lgth = AUTOREG_PARAMS['accompaniment']['autoreg_seg_lgth']
    seg_pad_unit = AUTOREG_PARAMS['accompaniment']['seg_pad_unit']
