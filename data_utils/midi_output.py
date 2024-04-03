import numpy as np
import pretty_midi as pm


def default_quantization(v):
    return 1 if v > 0.5 else 0


def piano_roll_to_note_mat(piano_roll: np.ndarray, raise_chord: bool,
                           quantization_func=None, seperate_chord=False):
    """
    piano_roll: (2, L, 128), onset and sustain channel.
    raise_chord: whether pitch below 48 (mel-chd boundary) will be raised an octave
    """
    def convert_p(p_, note_list, raise_pitch=False):
        edit_note_flag = False
        for t in range(n_step):
            onset_state = quantization_func(piano_roll[0, t, p_])
            sustain_state = quantization_func(piano_roll[1, t, p_])

            is_onset = bool(onset_state)
            is_sustain = bool(sustain_state) and not is_onset

            pitch = p_ + 12 if raise_pitch else p_

            if is_onset:
                edit_note_flag = True
                note_list.append([t, pitch, 1])
            elif is_sustain:
                if edit_note_flag:
                    note_list[-1][-1] += 1
            else:
                edit_note_flag = False
        return note_list

    quantization_func = default_quantization if quantization_func is None else quantization_func
    assert len(piano_roll.shape) == 3 and piano_roll.shape[0] == 2 and piano_roll.shape[2] == 128

    n_step = piano_roll.shape[1]

    notes = []
    chord_notes = []

    for p in range(128):
        if p < 48:
            convert_p(p, chord_notes if seperate_chord else notes, True if raise_chord else False)
        else:
            convert_p(p, notes, False)

    if seperate_chord:
        return notes, chord_notes
    else:
        return notes


def note_mat_to_notes(note_mat, bpm, unit, shift_beat=0., shift_sec=0., vel=100):
    """Default use shift beat"""

    beat_alpha = 60 / bpm
    step_alpha = unit * beat_alpha

    notes = []

    shift_sec = shift_sec if shift_beat is None else shift_beat * beat_alpha

    for note in note_mat:
        onset, pitch, dur = note
        start = onset * step_alpha + shift_sec
        end = (onset + dur) * step_alpha + shift_sec

        notes.append(pm.Note(vel, int(pitch), start, end))

    return notes


def create_pm_object(bpm, preset=0, instrument_names=None, notes_list=None):
    midi = pm.PrettyMIDI(initial_tempo=bpm)

    presets = {
        1: ['red_mel+red_chd'],
        2: ['red_mel+red_chd', 'mel+chd'],
        3: ['red_mel+red_chd', 'mel+chd', 'acc'],
        5: ['red_mel', 'red_chd', 'mel', 'chd', 'acc']
    }

    if instrument_names is None:
        instrument_names = presets[preset]

    midi.instruments = [pm.Instrument(0, name=name) for name in instrument_names]

    if notes_list is not None:
        assert len(notes_list) == len(midi.instruments)
        for i in range(len(midi.instruments)):
            midi.instruments[i].notes += notes_list[i]

    return midi
