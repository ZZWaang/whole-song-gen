import numpy as np


def key_estimation(mel_roll, chord_roll,
                   phrase_starts, phrase_lengths, num_beat_per_measure, num_step_per_beat):
    # chord_roll = self.chord_to_compact_pianoroll()
    key_template = np.array([1., 0., 1., 0., 1., 1., 0., 1., 0., 1., 0., 1.])
    key_templates = \
        np.stack([np.roll(key_template, step) for step in range(12)], 0)

    scales = []
    tonics = []
    for i in range(len(phrase_lengths)):
        start_measure = phrase_starts[i]
        lgth = phrase_lengths[i]

        start_beat = start_measure * num_beat_per_measure
        end_beat = (start_beat + lgth) * num_beat_per_measure

        start_step = start_beat * num_step_per_beat
        end_step = end_beat * num_step_per_beat

        chroma_hist = chord_roll[2: 4, start_beat: end_beat].sum(0).sum(0)

        if not (chroma_hist == 0).all():
            chroma_hist = chroma_hist / chroma_hist.sum()

        score = (chroma_hist[np.newaxis, :] @ key_templates.T)[0]
        max_val = score.max()
        cand_key = np.where(np.abs(score - max_val) < 1e-4)[0]
        if len(scales) > 0 and scales[-1] in cand_key:
            scale = scales[-1]
        else:
            scale = cand_key[0]
        scales.append(scale)

        mel_hist = mel_roll[:, start_step: end_step].sum(0).sum(0)
        major_score, minor_score = mel_hist[scale], mel_hist[(scale + 9) % 12]
        tonic = scale if major_score >= minor_score else (scale + 9) % 12
        tonics.append(tonic)

    scales = np.array(scales)
    tonics = np.array(tonics)
    keys = np.stack([tonics, scales], 0)
    return keys


def key_to_key_roll(keys, total_measure, phrase_lengths, phrase_starts):
    key_template = np.array([1., 0., 1., 0., 1., 1., 0., 1., 0., 1., 0., 1.])
    key_templates = \
        np.stack([np.roll(key_template, step) for step in range(12)], 0)

    key_roll = np.zeros((2, total_measure, 12), dtype=np.int64)

    for i in range(len(phrase_lengths)):
        start_measure = phrase_starts[i]
        lgth = phrase_lengths[i]
        key_roll[0, start_measure: start_measure + lgth, keys[0, i]] = 1
        key_roll[1, start_measure: start_measure + lgth] = key_templates[keys[1, i]]

    return key_roll


def get_key_roll(mel_roll, chord_roll, phrase_starts, phrase_lengths, total_measure,
                 num_beat_per_measure, num_step_per_beat):
    keys = key_estimation(mel_roll, chord_roll, phrase_starts, phrase_lengths, num_beat_per_measure, num_step_per_beat)
    key_roll = key_to_key_roll(keys, total_measure, phrase_lengths, phrase_starts)
    return key_roll
