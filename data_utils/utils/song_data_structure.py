import numpy as np


class McpaMusic:

    """
    MCPA Music contains Melody, Chords, Phrase, and Accompaniment annotations.
    """

    def __init__(self, melody, chord, acc, phrase_label,
                 num_beat_per_measure=4, num_step_per_beat=4,
                 song_name=None, clean_chord_unit=4):
        self.song_name = song_name

        # structural attributes
        self.num_beat_per_measure = num_beat_per_measure
        self.num_step_per_beat = num_step_per_beat

        # phrase attributes
        self.phrase_names = \
            np.array([pl['name'] for pl in phrase_label])
        self.phrase_types = \
            np.array([pl['type'] for pl in phrase_label])
        self.phrase_starts = \
            np.array([pl['start'] for pl in phrase_label])
        self.phrase_lengths = \
            np.array([pl['lgth'] for pl in phrase_label])
        self.num_phrases = len(phrase_label)

        # melody, chord and accompaniment
        self.melody = melody
        self.chord = chord
        self.acc = acc

        # determine piece length from phrase label, melody, and chord input
        self.total_measure = self.compute_total_measure()
        self.total_beat = self.total_measure * self.num_beat_per_measure
        self.total_step = self.total_beat * self.num_step_per_beat

        # ensuring chord having a maximum duration of self.clearn_chord_unit
        self.clean_chord_unit = clean_chord_unit
        self.clean_chord()

        # pad chord (pad = last chord) to match self.total_beat
        self.regularize_chord()

        # pad phrase with label 'z' to match self.total_measure
        self.regularize_phrases()

    def compute_total_measure(self):
        # propose candidates from phrase, chord, melody and acc
        last_step_mel = (self.melody[:, 0] + self.melody[:, 2]).max()
        if self.acc is None:
            last_step = last_step_mel
        else:
            last_step_acc = (self.acc[:, 0] + self.acc[:, 2]).max()
            last_step = max(last_step_mel, last_step_acc)

        num_measure0 = int(np.ceil(last_step / self.num_step_per_beat / self.num_beat_per_measure))

        last_beat = (self.chord[:, 0] + self.chord[:, -1]).max()
        num_measure1 = int(np.ceil(last_beat / self.num_beat_per_measure))

        num_measure2 = sum(self.phrase_lengths)
        return max(num_measure0, num_measure1, num_measure2)

    def regularize_chord(self):
        chord = self.chord
        end_time = (self.chord[:, 0] + self.chord[:, -1]).max()
        fill_n_beat = self.total_beat - end_time
        if fill_n_beat == 0:
            return

        pad_durs = [self.clean_chord_unit] * (fill_n_beat // self.clean_chord_unit)
        if fill_n_beat - sum(pad_durs) > 0:
            pad_durs = [fill_n_beat - sum(pad_durs)] + pad_durs
        for d in pad_durs:
            stack_chord = chord[-1].copy()
            stack_chord[0] = chord[-1, 0] + chord[-1, -1]
            stack_chord[-1] = d

            chord = np.concatenate([chord, stack_chord[np.newaxis, :]], 0)
        self.chord = chord

    def regularize_phrases(self):
        original_phrase_length = sum(self.phrase_lengths)
        if self.total_measure == original_phrase_length:
            return

        extra_phrase_length = self.total_measure - original_phrase_length
        extra_phrase_name = 'z' + str(extra_phrase_length)

        self.phrase_names = np.append(self.phrase_names, extra_phrase_name)
        self.phrase_types = np.append(self.phrase_types, 'z')
        self.phrase_lengths = np.append(self.phrase_lengths,
                                        extra_phrase_length)
        self.phrase_starts = np.append(self.phrase_starts,
                                       original_phrase_length)

    def clean_chord(self):
        chord = self.chord
        unit = self.clean_chord_unit

        new_chords = []
        n_chord = len(chord)
        for i in range(n_chord):
            chord_start = chord[i, 0]
            chord_dur = chord[i, -1]

            cum_dur = 0
            s = chord_start
            while cum_dur < chord_dur:
                d = min(unit - s % unit, chord_dur - cum_dur)
                c = chord[i].copy()
                c[0] = s
                c[-1] = d
                new_chords.append(c)

                s = s + d
                cum_dur += d

        new_chords = np.stack(new_chords, 0)
        self.chord = new_chords
