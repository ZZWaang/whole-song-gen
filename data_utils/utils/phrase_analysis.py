import numpy as np


def phrase_to_phrase_roll(phrase_starts, phrase_lengths, phrase_types,
                          total_measure=None):
    def phrase_type_to_index(p_type):
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

    total_measure = phrase_lengths.sum() if total_measure is None else total_measure

    measures = np.zeros((6, total_measure), dtype=np.float32)
    for start, length, ptype in zip(phrase_starts, phrase_lengths, phrase_types):
        phrase_index = phrase_type_to_index(ptype)
        measures[phrase_index, start: start + length] = \
            np.linspace(1., 0., length, endpoint=False)
    return measures