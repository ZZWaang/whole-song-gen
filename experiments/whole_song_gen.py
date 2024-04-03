# from data_utils import load_datasets
import os
from datetime import datetime
from inference.generation_operations import FormGenOp, CounterpointGenOp, LeadSheetGenOp, AccompanimentGenOp
from inference.utils import quantize_generated_form_batch, specify_form
import numpy as np
from params import params_frm, params_ctp, params_lsh, params_acc
from model import get_model_path
import torch
from data_utils.midi_output import piano_roll_to_note_mat, note_mat_to_notes, create_pm_object


class WholeSongGeneration:

    def __init__(
            self,
            frm_op: FormGenOp,
            ctp_op: CounterpointGenOp,
            lsh_op: LeadSheetGenOp,
            acc_op: AccompanimentGenOp,
            desc: str = None,
            random_n_autoreg: bool = False,
    ):
        self.frm_op = frm_op
        self.ctp_op = ctp_op
        self.lsh_op = lsh_op
        self.acc_op = acc_op
        self.random_n_autoreg = random_n_autoreg
        self.desc = desc
        print(f"Description of the experiment is: {self.desc}")

    def form_generation(self):
        print("Form generation...")
        frm_canvas, slices, gen_max_l = self.frm_op.create_canvas(n_sample=1, prompt=None)
        frm = self.frm_op.generation(frm_canvas, slices, gen_max_l, quantize=False, n_sample=1)
        frm, lengths, phrase_labels = quantize_generated_form_batch(frm)
        print(f"Length of the song: {lengths[0]}, phrase_label:\n{phrase_labels[0]}")
        return frm[:, :, 0: lengths[0]], lengths[0], phrase_labels[0]

    def counterpoint_generation(self, background_cond, n_sample, nbpm):
        print("Counterpoint generation...")
        background_cond = self.ctp_op.expand_background(background_cond, nbpm)
        ctp_canvas, slices, gen_max_l = \
            self.ctp_op.create_canvas(background_cond, n_sample, nbpm, None, self.random_n_autoreg)
        print(f"Number of iterations: {len(slices)}")
        ctp = self.ctp_op.generation(ctp_canvas, slices, gen_max_l)
        ctp = np.stack(ctp, 0)
        return ctp

    def leadsheet_generation(self, background_cond, n_sample=1, nbpm=4, nspb=4):
        print("Lead Sheet generation...")
        background_cond = self.lsh_op.expand_background(background_cond, nspb)
        lsh_canvas, slices, gen_max_l = \
            self.lsh_op.create_canvas(background_cond, n_sample, nbpm, nspb, None, self.random_n_autoreg)
        print(f"Number of iterations: {len(slices)}")
        lsh = self.lsh_op.generation(lsh_canvas, slices, gen_max_l)
        lsh = np.stack(lsh, 0)
        return lsh

    def accompaniment_generation(self, background_cond, n_sample=1, nbpm=4, nspb=4):
        print("Accompaniment generation...")
        acc_canvas, slices, gen_max_l = \
            self.acc_op.create_canvas(background_cond, n_sample, nbpm, nspb, None, self.random_n_autoreg)
        print(f"Number of iterations: {len(slices)}")
        lsh = self.acc_op.generation(acc_canvas, slices, gen_max_l)
        return lsh

    def main(self, n_sample, nbpm=4, nspb=4, phrase_string=None, key=0, is_major=True, demo_dir=None, bpm=90):
        if phrase_string is None:
            frm, _, phrase_string = self.form_generation()
        else:
            frm = np.expand_dims(specify_form(phrase_string, key, is_major), 0)

        ctp = self.counterpoint_generation(frm, n_sample, nbpm)
        lsh = self.leadsheet_generation(ctp, 1, nbpm, nspb)
        acc = self.accompaniment_generation(lsh, 1, nbpm, nspb)
        self.output(acc, phrase_string, key, is_major, demo_dir, bpm)

    def output(self, hie_langs, phrase_string, key, is_major, demo_dir, bpm=90):
        cur_time_str = f"{datetime.now().strftime('%m-%d_%H%M%S')}"
        exp_name = f"whole-song-gen-{cur_time_str}"
        exp_path = os.path.join(demo_dir, exp_name)

        os.makedirs(exp_path, exist_ok=True)

        # write description
        with open(os.path.join(exp_path, 'description.txt'), 'w') as file:
            file.write(self.desc)

        # write phrase_string
        if key is None:
            key = 'key: Key is generated (not specified). Visualization is not implemented.'
            is_major = 'is_major: Key is generated. Visualization is not implemented.'
        else:
            key = f"key: {key}"
            is_major = f"is_major: {is_major}"

        form = '\n'.join([phrase_string, key, is_major])

        with open(os.path.join(exp_path, 'form.txt'), 'w') as file:
            file.write(form)

        # write midi
        for i in range(len(hie_langs)):
            acc = hie_langs[i][0: 2]
            lsh = hie_langs[i][2: 4]
            cpt = hie_langs[i][4: 6]

            # output
            nmat_red_mel, nmat_red_chd = piano_roll_to_note_mat(cpt, True, seperate_chord=True)
            notes_red_mel = note_mat_to_notes(nmat_red_mel, bpm, unit=0.25)
            notes_red_chd = note_mat_to_notes(nmat_red_chd, bpm, unit=0.25)

            nmat_mel, nmat_chd = piano_roll_to_note_mat(lsh, True, seperate_chord=True)
            notes_mel = note_mat_to_notes(nmat_mel, bpm, unit=0.25)
            notes_chd = note_mat_to_notes(nmat_chd, bpm, unit=0.25)

            notes_acc = note_mat_to_notes(piano_roll_to_note_mat(acc, False), bpm, unit=0.25)

            midi = create_pm_object(bpm, preset=5,
                                    notes_list=[notes_mel, notes_red_chd, notes_mel, notes_chd, notes_acc])
            midi.write(os.path.join(exp_path, f'generation-{i}.mid'))

    @classmethod
    def init_pipeline(cls, frm_model_folder, ctp_model_folder, lsh_model_folder, acc_model_folder,
                      frm_model_id='best', ctp_model_id='best', lsh_model_id='best', acc_model_id='best',
                      use_autoreg_cond=True, use_external_cond=False,
                      debug_mode=False, is_autocast_fp16=True, random_n_autoreg=False, device=None):
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(frm_model_id, ctp_model_id, lsh_model_id, acc_model_id)
        frm_model_path, frm_model_id, frm_desc = get_model_path(frm_model_folder, frm_model_id)
        ctp_model_path, ctp_model_id, ctp_desc = get_model_path(ctp_model_folder, ctp_model_id)
        lsh_model_path, lsh_model_id, lsh_desc = get_model_path(lsh_model_folder, lsh_model_id)
        acc_model_path, acc_model_id, acc_desc = get_model_path(acc_model_folder, acc_model_id)

        frm_op = FormGenOp(params_frm, frm_model_path, device, False, False, debug_mode, is_autocast_fp16)
        ctp_op = CounterpointGenOp(params_ctp, ctp_model_path, device, use_autoreg_cond, use_external_cond,
                                   debug_mode, is_autocast_fp16)
        lsh_op = LeadSheetGenOp(params_lsh, lsh_model_path, device, use_autoreg_cond, use_external_cond,
                                debug_mode, is_autocast_fp16)
        acc_op = AccompanimentGenOp(params_acc, acc_model_path, device, use_autoreg_cond, use_external_cond,
                                    debug_mode, is_autocast_fp16)

        desc = f'm0-{frm_desc}-{frm_model_id}\nm1-{ctp_desc}-{ctp_model_id}\nm2-{lsh_desc}-{lsh_model_id}\n' \
               f'm3-{acc_desc}-{acc_model_id}'

        return cls(frm_op, ctp_op, lsh_op, acc_op, desc, random_n_autoreg)

