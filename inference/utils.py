import numpy as np


def quantize_generated_form(form, song_end_thresh=3, phrase_start_thresh=0.6):
    # form: (6, 256, 16)
    cleaned_form = np.zeros_like(form)

    key = form[0: 2, :, 0: 12]  # (2, 256, 12)

    # determine song end
    song_end = np.where(key.sum(-1).sum(0) < song_end_thresh)[0]
    song_end = key.shape[1] if len(song_end) == 0 else song_end[0]

    # quantize key
    key[key > 0.5] = 1.
    key[key < 0.95] = 0.
    key[:, song_end:] = 0.

    # quantize phrase to discrete representation
    phrase_roll = form[2:].mean(-1)  # (6, 256)
    phrase_value = phrase_roll.sum(0)  # (256,)

    phrases = []
    phrase_starts = []
    phrase_types = []
    phrase_lengths = []

    for t in range(song_end):
        if t == 0 or (phrase_value[t] > phrase_value[t - 1] and phrase_value[t] > phrase_start_thresh):
            phrase_starts.append(t)
            phrase_types.append(phrase_roll[:, t].argmax())
            phrase_lengths.append(1)
        else:
            phrase_lengths[-1] += 1
    phrases.append((phrase_starts, phrase_types, phrase_lengths))
    phrase_label = phrase_type_to_string(phrase_starts, phrase_types, phrase_lengths)

    # create new continuous phrase roll
    new_phrase_roll = np.zeros_like(phrase_roll)
    for ps, pt, pl in zip(phrase_starts, phrase_types, phrase_lengths):
        new_phrase_roll[int(pt), int(ps): int(ps) + int(pl)] = np.linspace(1, 0, int(pl), endpoint=False)

    cleaned_form[0: 2, :, 0: 12] = key
    cleaned_form[2:] = new_phrase_roll[:, :, np.newaxis]

    return cleaned_form, song_end, phrase_label


def phrase_type_to_string(phrase_starts, phrase_types, phrase_lengths):
    phrase_type_mapping = ['A', 'B', 'X', 'i', 'o', 'b']
    phrase_label = ''
    for ps, pt, pl in zip(phrase_starts, phrase_types, phrase_lengths):
        phrase_type = phrase_type_mapping[int(pt)]
        phrase_label += f"{int(ps)}: {phrase_type}{int(pl)}\n"
    return phrase_label


def quantize_generated_form_batch(forms, song_end_thresh=3, phrase_start_thresh=0.6):

    cleaned_forms = np.zeros_like(forms)
    phrase_labels = []
    n_measures = []

    for i, form in enumerate(forms):
        cleaned_form, song_end, phrase_label = quantize_generated_form(form, song_end_thresh, phrase_start_thresh)
        cleaned_forms[i] = cleaned_form
        phrase_labels.append(phrase_label)
        n_measures.append(song_end)
    return cleaned_forms, n_measures, phrase_labels


def phrase_config_from_string(phrase_annotation):
    index = 0
    phrase_configuration = []
    while index < len(phrase_annotation):
        label = phrase_annotation[index]
        index += 1
        n_bars = ''
        while index < len(phrase_annotation) and phrase_annotation[index].isdigit():
            n_bars += phrase_annotation[index]
            index += 1
        phrase_configuration.append((label, int(n_bars)))
    return phrase_configuration


def phrase_type_to_phrase_type_id(p_type):
    if p_type == 'A':
        return 0
    elif p_type == 'B':
        return 1
    elif p_type.isupper():
        return 2
    elif p_type == 'i':
        return 3
    elif p_type in ['o', 'z']:
        return 4
    else:
        return 5


def phrase_string_to_roll(phrase_string):
    phrase_config = phrase_config_from_string(phrase_string)
    total_measure = sum([p[1] for p in phrase_config])

    phrase_mat = np.zeros((6, total_measure, 16))
    cur_measure = 0
    for phrase_type, phrase_length in phrase_config:
        phrase_value = np.linspace(1, 0, phrase_length, endpoint=False)
        phrase_type_id = phrase_type_to_phrase_type_id(phrase_type)
        phrase_mat[phrase_type_id, cur_measure: cur_measure + phrase_length, :] = phrase_value[:, np.newaxis]
        cur_measure += phrase_length
    return phrase_mat


def specify_form(phrase_string, key, is_major=True):
    major_template = np.array([1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1])
    phrase_mat = phrase_string_to_roll(phrase_string)
    form = np.zeros((8, phrase_mat.shape[1], 16))
    form[0, :, key] = 1.
    shift = key if is_major else key - 3
    form[1, :, 0: 12] = np.roll(major_template, shift=key)
    form[2:, :, 0: 16] = phrase_mat
    return form
