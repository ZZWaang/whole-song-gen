from model import init_ldm_model, Diffpro_SDF, get_model_path
from model.stable_diffusion.sampler.sampler_sdf import SDFSampler
from data_utils import LANGUAGE_DATASET_PARAMS, AUTOREG_PARAMS
import numpy as np
import torch
import os
from .generation_canvases import GenerationCanvasBase, FormCanvas, CounterpointCanvas, LeadSheetCanvas, \
    AccompanimentCanvas


def compute_n_iterations(total_length, unit, hop_size=None, gen_max_l=None, max_l=None):
    """
    Assume L = n_measure * unit
        - The diffusion model generates: for i in range(k): (hop_size * i, hop_size * i + max_l)
        - In generation, we take first gen_max_l steps, i.e., (hop_size * i, hop_size * i + gen_max_l)
    So,
        - We know
            hop_size * (k - 2) + gen_max_l < L <= hop_size * (k - 1) + gen_max_l
        - That is
            k < (L - gen_max_l) / hop_size + 2
            k >= (L - gen_max_l) / hop_size + 1
    """

    if hop_size is None:
        hop_size = 60 if unit % 3 == 0 else 64
    if gen_max_l is None:
        gen_max_l = 120 if unit % 3 == 0 else 128
    if max_l is None:
        max_l = 128

    n_iteration = max(int(np.ceil((total_length - gen_max_l) / hop_size)), 0) + 1

    slices = [slice(hop_size * i, hop_size * i + max_l) for i in range(n_iteration)]

    n_padded_lgth = (n_iteration - 1) * hop_size + max_l

    return n_iteration, n_padded_lgth, slices, hop_size, gen_max_l


def expand_rolls(roll, unit=4, contain_onset=False):
    # roll: (bs, Channel, T, H) -> (Channel, T * unit, H)
    bs, n_channel, length, height = roll.shape
    expanded_roll = roll.repeat(unit, axis=2)
    if contain_onset:
        expanded_roll = expanded_roll.reshape((bs, n_channel, length, unit, height))
        expanded_roll[:, 1::2, :, 1:] = np.maximum(expanded_roll[:, ::2, :, 1:], expanded_roll[:, 1::2, :, 1:])

        expanded_roll[:, ::2, :, 1:] = 0
        expanded_roll = expanded_roll.reshape((bs, n_channel, length * unit, height))
    return expanded_roll


class GenOpBase:

    mode = None
    data_params = None

    def __init__(self, params, model_path, device, use_autoreg_cond=False, use_external_cond=False, debug_mode=False,
                 is_autocast_fp16=True):
        self.params = params
        self.device = device
        self.use_autoreg_cond = use_autoreg_cond
        self.use_external_cond = use_external_cond
        self.debug_mode = debug_mode
        self.is_autocast_fp16 = is_autocast_fp16

        self.sampler = self.load_sampler(model_path, False)
        # self._consistency_check(model_path)

    def _consistency_check(self, model_path):
        train_args = model_path.split(os.sep)[-4]
        args0, args1, _, _ = train_args.split('-')
        assert args0 == self.mode
        if self.use_autoreg_cond:
            assert 'a' in args1
        if self.use_external_cond:
            assert 'e' in args1

    def load_sampler(self, model_path, show_image):
        # model ready
        ldm_model = init_ldm_model(self.mode, self.use_autoreg_cond, self.use_external_cond, self.params, self.debug_mode)

        model = Diffpro_SDF.load_trained(ldm_model, model_path).to(self.device)

        sampler = SDFSampler(model.ldm, self.data_params['max_l'],
                             self.data_params['h'], is_autocast=self.is_autocast_fp16, device=self.device,
                             debug_mode=self.debug_mode)

        return sampler

    def predict(self, background_cond, autoreg_cond, external_cond, orig_x, mask, uncond_scale=None,
                n_sample=None):
        if background_cond is not None:
            background_cond = torch.from_numpy(background_cond).float().to(self.device)

        if autoreg_cond is not None:
            autoreg_cond = torch.from_numpy(autoreg_cond).float().to(self.device)

        if external_cond is not None:
            external_cond = torch.from_numpy(external_cond).float().to(self.device)

        if mask is not None:
            mask = torch.from_numpy(mask).float().to(self.device)

        if orig_x is not None:
            orig_x = torch.from_numpy(orig_x).float().to(self.device)

        if n_sample is not None:
            batch_size = n_sample

        self.sampler.model.eval()

        output_x = self.sampler.generate(background_cond, autoreg_cond, external_cond, orig_x, mask,
                                         n_sample, uncond_scale)

        output_x = torch.clamp(output_x, min=0, max=1)

        output_x = output_x.cpu().numpy()

        return output_x

    def generation(self, generation_canvas: GenerationCanvasBase, slices, gen_max_l, quantize=True,
                   n_sample=None):
        for slc in slices:
            start_id, end_id = slc.start, slc.stop
            generated_end_id = start_id + gen_max_l

            # determine song to generate
            is_generated = generation_canvas.check_is_generated(start_id, generated_end_id)

            if is_generated.all():
                continue

            sel_song_ids = np.logical_not(is_generated)

            background_cond = generation_canvas.get_batch_background_cond(start_id, sel_song_ids)
            autoreg_cond = generation_canvas.get_batch_autoreg_cond(start_id, sel_song_ids)
            mask = generation_canvas.get_batch_mask(start_id, sel_song_ids)

            if mask.any():
                orig_x = generation_canvas.get_batch_cur_level(start_id, sel_song_ids)
            else:
                orig_x, mask = None, None

            external_cond = generation_canvas.get_batch_external_cond()

            if background_cond is None and autoreg_cond is None and external_cond is None and mask is None:
                batch_size = n_sample
            else:
                batch_size = None
            output_x = self.predict(background_cond, autoreg_cond, external_cond, orig_x, mask, uncond_scale=None,
                                    n_sample=batch_size)

            generation_canvas.write_generation(output_x, start_id, start_id + gen_max_l, sel_song_ids,
                                               quantize=quantize)
        songs = generation_canvas.to_output()

        return songs


class FormGenOp(GenOpBase):

    mode = 'frm'
    data_params = LANGUAGE_DATASET_PARAMS['form']

    def create_canvas(self, n_sample, prompt=None):
        n_channel, cur_channel, max_l, h = \
            [self.data_params[key] for key in ['n_channel', 'cur_channel', 'max_l', 'h']]
        total_length = max_l

        slices = [slice(0, max_l)]

        gen_max_l = max_l

        generation = np.ones((n_sample, n_channel, max_l, h)) * -1

        units = [1] * n_sample
        lengths = [max_l] * n_sample

        mask = np.zeros((n_sample, cur_channel, max_l, h), dtype=np.int64)

        if prompt is not None:
            prompt_l = prompt.shape[2]
            generation[:, 0: cur_channel, 0: prompt_l] = prompt
            mask[:, :, 0: prompt_l] = 1

        return FormCanvas(generation, units, lengths, mask, False), slices, gen_max_l


class CounterpointGenOp(GenOpBase):

    mode = 'ctp'
    data_params = LANGUAGE_DATASET_PARAMS['counterpoint']

    def expand_background(self, background, nbpm):
        """form: (1/bs, 8, L, 14) -> (1/bs, 8, L * nbpm, 128)"""
        background = background[:, :, :, 0: 12]
        background = np.tile(background.repeat(nbpm, axis=-2), reps=(1, 1, 1, 11))
        background = background[:, :, :, 0: self.data_params['h']]
        return background

    def create_canvas(self, background_cond, n_sample, nbpm=4, prompt=None, random_n_autoreg=False):
        total_length = background_cond.shape[-2]

        n_channel, cur_channel, max_l, h = \
            [self.data_params[key] for key in ['n_channel', 'cur_channel', 'max_l', 'h']]

        # create canvas
        n_iteration, n_padded_lgth, slices, hop_size, gen_max_l = \
            compute_n_iterations(total_length, nbpm, max_l=max_l)

        generation = np.ones((n_sample, n_channel, n_padded_lgth, h)) * -1
        generation[:, cur_channel:, 0: total_length] = background_cond

        units = [nbpm] * n_sample
        lengths = [total_length] * n_sample

        mask = np.zeros((n_sample, cur_channel, n_padded_lgth, h), dtype=np.int64)

        if prompt is not None:
            prompt_l = prompt.shape[2]
            generation[:, 0: cur_channel, 0: prompt_l] = prompt
            mask[:, :, 0: prompt_l] = 1

        return CounterpointCanvas(generation, units, lengths, mask, random_n_autoreg), slices, gen_max_l


class LeadSheetGenOp(GenOpBase):

    mode = 'lsh'
    data_params = LANGUAGE_DATASET_PARAMS['lead_sheet']

    def expand_background(self, background, nspb):
        """form: (1/bs, 10, L, 128) -> (1/bs, 10, L * nspb, 128)"""
        frm = background[:, 2:]
        ctp = background[:, 0: 2]

        frm = expand_rolls(frm, unit=nspb, contain_onset=False)
        ctp = expand_rolls(ctp, unit=nspb, contain_onset=True)

        return np.concatenate([ctp, frm], axis=1)

    def create_canvas(self, background_cond, n_sample=1, nbpm=4, nspb=4, prompt=None, random_n_autoreg=False):
        if background_cond.shape[0] != 1:
            n_sample = background_cond.shape[0]
        total_length = background_cond.shape[-2]

        n_channel, cur_channel, max_l, h = \
            [self.data_params[key] for key in ['n_channel', 'cur_channel', 'max_l', 'h']]

        # create canvas
        n_iteration, n_padded_lgth, slices, hop_size, gen_max_l = \
            compute_n_iterations(total_length, nbpm * nspb, max_l=max_l)

        generation = np.ones((n_sample, n_channel, n_padded_lgth, h)) * -1
        generation[:, cur_channel:, 0: total_length] = background_cond

        units = [nbpm * nspb] * n_sample
        lengths = [total_length] * n_sample

        mask = np.zeros((n_sample, cur_channel, n_padded_lgth, h), dtype=np.int64)

        if prompt is not None:
            prompt_l = prompt.shape[2]
            generation[:, 0: cur_channel, 0: prompt_l] = prompt
            mask[:, :, 0: prompt_l] = 1

        return LeadSheetCanvas(generation, units, lengths, mask, random_n_autoreg), slices, gen_max_l


class AccompanimentGenOp(GenOpBase):

    mode = 'acc'
    data_params = LANGUAGE_DATASET_PARAMS['accompaniment']

    def create_canvas(self, background_cond, n_sample=1, nbpm=4, nspb=4, prompt=None, random_n_autoreg=False):
        if background_cond.shape[0] != 1:
            n_sample = background_cond.shape[0]
        total_length = background_cond.shape[-2]

        n_channel, cur_channel, max_l, h = \
            [self.data_params[key] for key in ['n_channel', 'cur_channel', 'max_l', 'h']]

        # create canvas
        n_iteration, n_padded_lgth, slices, hop_size, gen_max_l = \
            compute_n_iterations(total_length, nbpm * nspb, max_l=max_l)

        generation = np.ones((n_sample, n_channel, n_padded_lgth, h)) * -1
        generation[:, cur_channel:, 0: total_length] = background_cond

        units = [nbpm * nspb] * n_sample
        lengths = [total_length] * n_sample

        mask = np.zeros((n_sample, cur_channel, n_padded_lgth, h), dtype=np.int64)

        if prompt is not None:
            prompt_l = prompt.shape[2]
            generation[:, 0: cur_channel, 0: prompt_l] = prompt
            mask[:, :, 0: prompt_l] = 1

        return AccompanimentCanvas(generation, units, lengths, mask, random_n_autoreg), slices, gen_max_l
