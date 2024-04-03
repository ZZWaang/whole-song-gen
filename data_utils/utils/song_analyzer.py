import numpy as np
from ..tonal_reduction_algo.main import TrAlgo
from .format_converter import note_matrix_to_piano_roll, chord_mat_to_chord_roll
from .key_analysis import get_key_roll
from .phrase_analysis import phrase_to_phrase_roll
from .song_data_structure import McpaMusic
from .chord_reduction import get_chord_reduction


class LanguageExtractor:

    def __init__(self, song: McpaMusic):
        self.song = song

        self._mel_roll = None
        self._chd_roll = None
        self._song_dict = None

    def _phrase_to_beat(self, phrase):
        start_measure = self.song.phrase_starts[phrase]
        end_measure = self.song.phrase_lengths[phrase] + start_measure
        start_beat = start_measure * self.song.num_beat_per_measure
        end_beat = end_measure * self.song.num_beat_per_measure
        return start_beat, end_beat

    def _create_a_phrase_level_dict(self, phrase_id):
        start_measure = self.song.phrase_starts[phrase_id]
        phrase_length = self.song.phrase_lengths[phrase_id]
        end_measure = start_measure + phrase_length
        phrase_dict = {
            'phrase_name': self.song.phrase_names[phrase_id],
            'phrase_type': self.song.phrase_types[phrase_id],
            'phrase_length': self.song.phrase_lengths[phrase_id],
            'start_measure': start_measure,
            'end_measure': end_measure,
            'length': phrase_length,
            'mel_slice': None,
            'chd_slice': None,
        }
        return phrase_dict

    def _create_song_level_dict(self, melody, chord):
        self._song_dict = {
            'song_name': self.song.song_name,
            'total_phrase': self.song.num_phrases,
            'total_measure': self.song.total_measure,
            'total_beat': self.song.total_beat,
            'total_step': self.song.total_step,
            'phrases': [self._create_a_phrase_level_dict(phrase_id)
                        for phrase_id in range(self.song.num_phrases)]
        }
        self._fill_phrase_level_slices(melody, chord)

    def _fill_phrase_level_mel_slices(self, melody):
        n_note = melody.shape[0]

        onset_beats = melody[:, 0] // self.song.num_step_per_beat

        current_ind = 0
        for phrase_id, phrase in enumerate(self._song_dict['phrases']):
            start_beat, end_beat = self._phrase_to_beat(phrase_id)
            for i in range(current_ind, n_note):
                if onset_beats[i] >= end_beat:
                    phrase[f'mel_slice'] = slice(current_ind, i)
                    current_ind = i
                    break
            else:
                phrase[f'mel_slice'] = slice(current_ind, n_note)
                current_ind = n_note

    def _fill_phrase_level_chd_slices(self, chord):
        n_chord = chord.shape[0]
        current_ind = 0
        for phrase_id, phrase in enumerate(self._song_dict['phrases']):
            start_beat, end_beat = self._phrase_to_beat(phrase_id)
            for i in range(current_ind, n_chord):
                if chord[i, 0] >= end_beat:
                    phrase['chd_slice'] = slice(current_ind, i)
                    current_ind = i
                    break
            else:
                phrase['chd_slice'] = slice(current_ind, n_chord)
                current_ind = n_chord

    def _fill_phrase_level_slices(self, melody, chord):
        self._fill_phrase_level_mel_slices(melody)
        self._fill_phrase_level_chd_slices(chord)

    def extract_form(self):
        """Extract lang0: Form (key and phrase)"""

        key_roll = get_key_roll(self._mel_roll, self._chd_roll,
                                self.song.phrase_starts, self.song.phrase_lengths, self.song.total_measure,
                                self.song.num_beat_per_measure, self.song.num_step_per_beat)

        phrase_roll = phrase_to_phrase_roll(self.song.phrase_starts, self.song.phrase_lengths,
                                            self.song.phrase_types, self.song.total_measure)

        return {'key_roll': key_roll, 'phrase_roll': phrase_roll}

    def get_melody_reduction(self, num_reduction=1, melody=None, chord=None):
        melody = self.song.melody if melody is None else melody
        chord = self.song.chord if chord is None else chord

        self._create_song_level_dict(melody, chord)

        tr_algo = TrAlgo()

        nbpm = self.song.num_beat_per_measure
        nspb = self.song.num_step_per_beat

        reductions = [[] for _ in range(num_reduction)]

        for phrase in self._song_dict['phrases']:
            mel_slice = phrase['mel_slice']
            chd_slice = phrase['chd_slice']

            note_mat = melody[mel_slice].copy()
            chord_mat = chord[chd_slice].copy()

            start_measure = phrase['start_measure']

            _, _, reduction_mats = \
                tr_algo.run(note_mat, chord_mat, start_measure, nbpm, nspb, num_path=1, plot_graph=False)

            for i in range(num_reduction):
                red_mat = reduction_mats[i]
                red_mat[:, 0] = red_mat[:, 0] // self.song.num_step_per_beat
                red_mat[:, 2] = red_mat[:, 2] // self.song.num_step_per_beat
                reductions[i].append(red_mat)

        reductions = [np.concatenate(reductions[i], 0) for i in range(num_reduction)]

        return reductions

    def extract_counterpoint(self):
        rough_chord = get_chord_reduction(self.song.chord, self.song.clean_chord_unit)
        red_chd_roll = chord_mat_to_chord_roll(rough_chord, self.song.total_beat)

        reduction = self.get_melody_reduction(num_reduction=1, melody=self.song.melody, chord=rough_chord)[0]

        red_mel_roll = note_matrix_to_piano_roll(reduction, self.song.total_beat)

        return {'red_mel_roll': red_mel_roll, 'red_chd_roll': red_chd_roll}

    def extract_lead_sheet(self):
        mel_roll = note_matrix_to_piano_roll(self.song.melody, self.song.total_step)
        chd_roll = chord_mat_to_chord_roll(self.song.chord, self.song.total_beat)

        self._mel_roll = mel_roll
        self._chd_roll = chd_roll

        return {'mel_roll': mel_roll, 'chd_roll': chd_roll}

    def extract_accompaniment(self):
        acc_roll = note_matrix_to_piano_roll(self.song.acc, self.song.total_step)
        return {'acc_roll': acc_roll}

    def extract_all_hie_langs(self):
        accompaniment = self.extract_accompaniment()
        lead_sheet = self.extract_lead_sheet()
        counterpoint = self.extract_counterpoint()
        form = self.extract_form()

        return {'form': form, 'counterpoint': counterpoint, 'lead_sheet': lead_sheet, 'accompaniment': accompaniment}

    def analyze_for_training(self):
        # extract min and max melody pitch. In image representation, the lowest melody pitch should be higher than midi
        # pitch 48 after pitch augmentation.

        min_mel_pitch, max_mel_pitch = self.song.melody[:, 1].min(), self.song.melody[:, 1].max()

        languages = self.extract_all_hie_langs()

        return {'name': self.song.song_name, 'nbpm': self.song.num_beat_per_measure,
                'nspb': self.song.num_step_per_beat,
                'min_mel_pitch': min_mel_pitch, 'max_mel_pitch': max_mel_pitch,
                'languages': languages}

