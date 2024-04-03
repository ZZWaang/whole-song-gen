import warnings
import numpy as np


def remove_offset(note_mat, chord_mat, start_measure, measure_to_step_fn,
                  measure_to_beat_fn):

    start_step = measure_to_step_fn(start_measure)
    start_beat = measure_to_beat_fn(start_measure)

    note_mat_ = note_mat.copy()
    note_mat_[:, 0] -= start_step

    chord_mat_ = chord_mat.copy()
    chord_mat_[:, 0] -= start_beat

    return note_mat_, chord_mat_


def chord_id_analysis(note_mat, chord_mat, step_to_beat_fn):
    """compute note to chord pointer"""
    chord_starts = chord_mat[:, 0]

    chord_ends = chord_mat[:, -1] + chord_starts  # chord duration in beat + chord start

    note_onsets = note_mat[:, 0]

    onset_beats = step_to_beat_fn(note_onsets)

    # output: (note_ids, chord_ids) e.g., (0, 1, 2, 3, ...), (0, 0, 1, 1, ...)
    chord_ids = np.where(np.logical_and(
        chord_starts <= onset_beats[:, np.newaxis],
        chord_ends > onset_beats[:, np.newaxis]))

    if not (chord_ids[0] == np.arange(0, len(note_onsets))).all():
        raise ValueError("Input melody onsets cannot point to input chords.")

    return chord_ids[1]


def chord_tone_analysis(note_mat, chord_mat, chord_ids):
    # output col 1: normal chord tone, col 2: anticipation
    n_note = note_mat.shape[0]

    chords = chord_mat[chord_ids]
    pitches = note_mat[:, 1].astype(np.int64)

    # find notes of regular chord tone: pitch in chord chroma
    is_reg_chord_tone = (chords[np.arange(0, n_note), 2 + pitches % 12] == 1).astype(np.int64)

    # find notes of anticipation
    # condition 1: next chord exists
    next_c_exist_rows = chord_ids < chord_mat.shape[0] - 1

    # condition 2:  last note in the current chord
    last_note_in_chord_rows = chord_ids[0: -1] < chord_ids[1:]
    last_note_in_chord_rows = np.append(last_note_in_chord_rows, True)

    # anticipation_condition_rows = np.where(np.logical_and(next_c_exist_rows,
    #                                                  last_note_in_chord_rows))[0]
    anticipation_condition_rows = np.logical_and(next_c_exist_rows, last_note_in_chord_rows)

    # anticipation: is the chord tone of the next chord (and not a regular chord tone)
    is_anticiptation = np.zeros(n_note, dtype=np.int64)

    is_anticiptation[anticipation_condition_rows] = \
        chord_mat[chord_ids[anticipation_condition_rows] + 1,
                  2 + pitches[anticipation_condition_rows] % 12] == 1

    is_anticiptation = \
        np.logical_and(np.logical_not(is_reg_chord_tone), is_anticiptation)

    # chord tones are regular chord tones or anticipation
    is_chord_tone = np.logical_or(is_reg_chord_tone, is_anticiptation)

    # tonal chord ids are the actual chord that a note is in harmonic with
    tonal_chord_ids = chord_ids.copy()
    tonal_chord_ids[is_anticiptation] += 1

    return is_chord_tone, is_anticiptation, tonal_chord_ids


def compute_tonal_note_onsets(onsets, chord_mat, chord_ids, is_anticipation, beat_to_step_fn):
    tonal_onset = onsets.copy()
    tonal_onset[is_anticipation] = beat_to_step_fn(chord_mat[chord_ids[is_anticipation] + 1, 0])
    return tonal_onset


def compute_bar_ids(onsets, tonal_onsets, step_to_measure_fn):
    bar_ids = step_to_measure_fn(onsets)
    tonal_bar_ids = step_to_measure_fn(tonal_onsets)
    return bar_ids, tonal_bar_ids


def preprocess_data(note_mat, chord_mat, start_measure,
                    measure_to_step_fn, measure_to_beat_fn,
                    step_to_beat_fn,
                    step_to_measure_fn, beat_to_measure_fn,
                    beat_to_step_fn):

    # setting phrase start to 0
    note_mat, chord_mat = remove_offset(note_mat, chord_mat, start_measure,
                                        measure_to_step_fn, measure_to_beat_fn)

    onsets, pitches, durations = note_mat.T

    # harmony analysis
    chord_ids = chord_id_analysis(note_mat, chord_mat, step_to_beat_fn)

    is_chord_tone, is_anticipation, tonal_chord_ids = \
        chord_tone_analysis(note_mat, chord_mat, chord_ids)

    # compute tonal onsets and bar ids
    tonal_onsets = compute_tonal_note_onsets(onsets, chord_mat, chord_ids, is_anticipation, beat_to_step_fn)
    bar_id, tonal_bar_id = compute_bar_ids(onsets, tonal_onsets, step_to_measure_fn)

    output_note_mat = \
        np.stack([onsets, pitches, durations, bar_id, tonal_bar_id,
                  chord_ids, tonal_chord_ids, is_chord_tone, tonal_onsets], -1)
    return output_note_mat, chord_mat
