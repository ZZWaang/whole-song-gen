import numpy as np


def _parse_chord(c):
    """Returns onset, root, chroma, bass, duration."""
    return c[0], c[1], c[2: 14], c[14], c[15]


def _chroma1_in_chroma_2(chroma1, chroma2):
    return (chroma2[chroma1 != 0] == 1).all()


def _share_chroma(chroma1, chroma2):
    return np.count_nonzero(np.logical_and(chroma1, chroma2)) >= 3


def get_chord_reduction(chord_mat, time_unit=4):
    """
    This function merge chords together. 9Two chords will be merged if:
        1) They are within the same time_unit (usually 2 - 4 beats)
        2) They have the same root or bass
        3) One chord chroma includes the other or two chord chromas share more than three notes
    """

    red_chord_mat = []

    i = 0
    while i < len(chord_mat):
        c_i = chord_mat[i]
        onset_i, root_i, chroma_i, bass_i, duration_i = _parse_chord(chord_mat[i])

        new_root, new_chroma, new_bass, acc_duration = root_i, chroma_i, bass_i, duration_i

        j = i + 1
        while acc_duration < time_unit and j < len(chord_mat):
            onset_j, root_j, chroma_j, bass_j, duration_j = _parse_chord(chord_mat[j])

            if onset_j // time_unit != onset_i // time_unit:  # not in the same time_unit
                break

            if root_i == root_j or bass_i == bass_j:
                if _chroma1_in_chroma_2(chroma_i, chroma_j): # chroma i in chroma j, use chord_j
                    new_root, new_chroma, new_bass = root_j, chroma_j, bass_j
                    acc_duration += duration_j
                    j += 1

                elif _chroma1_in_chroma_2(chroma_j, chroma_i): # chroma j in chroma i, use chord_i
                    acc_duration += duration_j
                    j += 1

                elif _share_chroma(chroma_i, chroma_j): # share more than three chord tone, use chord_i
                    acc_duration += duration_j
                    j += 1
                else:
                    break
            else:
                break

        red_chord_mat.append(
            np.concatenate([np.array([onset_i, new_root]), new_chroma, np.array([new_bass, acc_duration])]))
        i = j

    red_chord_mat = np.stack(red_chord_mat, 0).astype(np.int64)

    return red_chord_mat
