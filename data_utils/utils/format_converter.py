import numpy as np
from sklearn.linear_model import LinearRegression
import pretty_midi as pm


def mono_note_matrix_to_pr_array(note_mat, total_length):
    total_length = total_length if total_length is not None else max(note_mat[:, 0] + note_mat[:, 2])

    pr_array = np.ones(total_length, dtype=np.int64) * -1
    for note in note_mat:
        onset, pitch, duration = note
        pr_array[onset] = pitch
        pr_array[onset + 1: onset + duration] = pitch + 128
    return pr_array


def note_matrix_to_piano_roll(note_mat, total_length=None):
    total_length = total_length if total_length is not None else max(note_mat[:, 0] + note_mat[:, 2])

    piano_roll = np.zeros((2, total_length, 128), dtype=np.int64)
    for note in note_mat:
        onset, pitch, duration = note
        piano_roll[0, onset, pitch] = 1
        piano_roll[1, onset + 1: onset + duration, pitch] = 1
    return piano_roll


def pr_array_to_piano_roll(pr_array):
    total_length = pr_array.shape[0]

    piano_roll = np.zeros((2, total_length, 128), dtype=np.int64)

    is_onset = np.logical_and(pr_array >= 0, pr_array < 128)
    is_sustain = pr_array >= 128

    piano_roll[0, is_onset, pr_array[is_onset]] = 1
    piano_roll[1, is_sustain, pr_array[is_sustain] - 128] = 1

    return piano_roll


def pr_array_to_pr_contour(pr_array):
    pr_contour = pr_array.copy()

    onsets = np.where(np.logical_and(pr_array >= 0, pr_array < 128))[0]
    if len(onsets) == 0:
        pr_contour[:] = 60
        return pr_contour

    first_onset = onsets[0]
    first_pitch = pr_array[first_onset]

    pr_contour[0: first_onset] = first_pitch
    for i in range(first_onset, len(pr_contour)):
        pitch = pr_contour[i]
        if pitch >= 128:
            pr_contour[i] = pitch - 128
        elif pitch == -1:
            pr_contour[i] = pr_contour[i - 1]
    return pr_contour


def extract_pitch_contour(pr_contour, nspb, stride=2):
    # see pivot point high low point.
    def direction_via_regression(contour, length=None):

        t = np.linspace(0, 1, length)[:, np.newaxis]
        reg = LinearRegression().fit(t[0: len(contour)], contour)
        a = reg.coef_[0]

        if a > 5:
            contour_type = 4
        elif 1 < a <= 5:
            contour_type = 3
        elif -1 <= a <= 1:
            contour_type = 2
        elif -5 <= a < -1:
            contour_type = 1
        else:
            contour_type = 0
        return contour_type

    contour = []
    for i in range(0, len(pr_contour), int(nspb * stride)):
        segment = pr_contour[i: int(i + nspb * stride * 2)]
        direction = direction_via_regression(segment, int(nspb * stride * 2))
        contour.append(direction)
    return np.array(contour, dtype=np.int64)


def pr_array_to_rhythm(pr_array):
    rhythm_array = np.zeros_like(pr_array)

    sustain = pr_array >= 128
    rest = pr_array < 0

    rhythm_array[sustain] = 1
    rhythm_array[rest] = 2

    return rhythm_array


def extract_rhythm_intensity(rhythm_array, nspb, stride=2,
                             quantization_bin=4):
    def compute_intensity(rhy_segment):
        n_step = len(rhy_segment)

        onset = np.count_nonzero(rhy_segment == 0) / n_step
        rest = np.count_nonzero(rhy_segment == 2) / n_step

        # quantization
        onset_val = int(np.ceil(onset * (quantization_bin - 1)))
        rest_val = int(np.ceil(rest * (quantization_bin - 1)))

        return [onset_val, rest_val]

    intensity_array = []
    for i in range(0, len(rhythm_array), int(nspb * stride)):
        segment = rhythm_array[i: int(i + nspb * stride * 2)]
        intensity = compute_intensity(segment)
        intensity_array.append(intensity)
    return np.array(intensity_array, dtype=np.int64)


def chord_mat_to_chord_roll(chord, total_beat):
    chord_roll = np.zeros((6, total_beat, 12), dtype=np.int64)

    for c in chord:
        start_beat = c[0]
        chord_content = c[1: 15]
        dur_beat = c[15]

        root = chord_content[0]
        bass = (chord_content[-1] + root) % 12
        chroma = chord_content[1: 13]
        # print(start_beat, total_beat)
        chord_roll[0, start_beat, root] = 1
        chord_roll[1, start_beat + 1: start_beat + dur_beat, root] = 1

        chord_roll[2, start_beat, :] = chroma
        chord_roll[3, start_beat + 1: start_beat + dur_beat, :] = chroma

        chord_roll[4, start_beat, bass] = 1
        chord_roll[5, start_beat + 1: start_beat + dur_beat, bass] = 1

    return chord_roll


def chord_to_chord_roll(chord, total_beat):
    chord_roll = np.zeros((2, total_beat, 36), dtype=np.int64)

    for c in chord:
        start_beat = c[0]
        chord_content = c[1: 15]
        dur_beat = c[15]

        root = chord_content[0]
        bass = (chord_content[-1] + root) % 12
        chroma = chord_content[1: 13]

        chord_roll[0, start_beat, root] = 1
        chord_roll[0, start_beat, 12: 24] = chroma
        chord_roll[0, start_beat, 24 + bass] = 1
        chord_roll[1, start_beat + 1: start_beat + dur_beat, root] = 1
        chord_roll[1, start_beat + 1: start_beat + dur_beat, 12: 24] = chroma
        chord_roll[1, start_beat + 1: start_beat + dur_beat, 24 + bass] = 1

    return chord_roll


def note_mat_to_notes(note_mat, bpm, factor=4, shift=0.):
    alpha = 60 / bpm / factor
    notes = []
    for note in note_mat:
        onset, pitch, dur = note

        notes.append(pm.Note(100, int(pitch), onset * alpha + shift, (onset + dur) * alpha + shift))
    return notes


def chord_roll_to_chord_stack(chord_roll, nbpm, pad=True):
    # (6, T, 12) -> (6 * nbpm, T // nbpm, 12)
    n_channel, lgth, h = chord_roll.shape
    assert lgth % nbpm == 0
    lgth_ = lgth // nbpm
    chord_roll = chord_roll.copy().reshape((n_channel, lgth_, nbpm, h))
    chord_roll = chord_roll.transpose((0, 2, 1, 3))

    if pad and nbpm == 3:
        chord_roll = np.pad(chord_roll, pad_width=((0, 0), (0, 1), (0, 0), (0, 0)),
                            mode='constant')
    chord_roll = chord_roll.reshape((n_channel // 2, 2, 4, lgth_, h)).sum(1)
    chord_roll = chord_roll.reshape((n_channel // 2 * 4, lgth_, h))

    return chord_roll


def reduction_roll_to_reduction_stack(reduction, nbpm, pad=True):
    # (2, T, 128) -> (nbpm, T // nbpm, 12)
    n_channel, lgth, h = reduction.shape
    assert lgth % nbpm == 0
    assert h == 128

    lgth_ = lgth // nbpm

    reduction = np.pad(reduction.copy(), pad_width=((0, 0), (0, 0), (0, 4)),
                       mode='constant')

    reduction = reduction.reshape((n_channel, lgth_, nbpm, 11, 12)).sum(-2)
    reduction = reduction.transpose((0, 2, 1, 3))
    if pad and nbpm == 3:
        reduction = np.pad(reduction, pad_width=((0, 0), (0, 1), (0, 0), (0, 0)),
                           mode='constant')
    reduction = reduction.sum(0).reshape((4, lgth_, 12))
    return reduction
