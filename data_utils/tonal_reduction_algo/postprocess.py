import numpy as np


def compute_chord_density(path, note_mat, chords):
    chord_density = [0 for _ in range(len(chords))]
    chord_bin = [[] for _ in range(len(chords))]
    chord_max_density = [chord[-1] for chord in chords]

    for pid, p in enumerate(path):
        note = note_mat[p]

        # use tonal chord id as the chord id if possible
        chord_id = int(note[6]) \
            if int(note[6]) < len(chord_density) else int(note[5])

        chord_density[chord_id] += 1
        chord_bin[chord_id].append(pid)

    is_overflow = [cd > cmd for cd, cmd in
                   zip(chord_density, chord_max_density)]

    return chord_density, chord_max_density, chord_bin, is_overflow


def path_to_chord_bins(path, note_mat, chord):
    if path is None:
        return None, None
    path, reduction_rate = path['path'], path['reduction_rate']

    chord_density, chord_max_density, chord_bin, is_overflow = \
        compute_chord_density(path, note_mat, chord)

    modify_list = [[] for _ in range(len(chord_density))]
    modify_chord_density = chord_density.copy()

    prev_cd = None
    for i in range(len(chord_density) - 1, -1, -1):
        cd, cmd, cb = chord_density[i], chord_max_density[i], chord_bin[i]

        current_cd = cd
        # check prolongation
        if not is_overflow[i]:
            prev_cd = current_cd
            continue

        # check prolongation
        for j, pid in enumerate(cb):
            p = path[pid]
            if pid != 0:
                prev_p = path[pid - 1]
                if note_mat[p, 2] == note_mat[prev_p, 2]:
                    modify_list[i].append((j, 'r'))
                    current_cd -= 1
                if current_cd <= cmd:
                    break

        if current_cd > cmd:
            for j in range(len(cb) - 1, -1, -1):

                if prev_cd is None:
                    break
                if j in [m[0] for m in modify_list[i]]:
                    continue

                pid = cb[j]
                p = path[pid]
                note = note_mat[p]

                # check note is chord tone of chord[i + 1]
                if chord[i + 1, int(note[2] % 12) + 1] == 1 and prev_cd <= chord_max_density[i + 1] - 1:
                    modify_list[i].append((j, 'm'))
                    current_cd -= 1
                    modify_chord_density[i + 1] += 1
                    prev_cd += 1
                break

        if current_cd > cmd:
            # in this case drop note
            for j in range(len(cb) - 1, -1, -1):
                if j not in [m[0] for m in modify_list[i]]:
                    modify_list[i].append((j, 'd'))
                    current_cd -= 1
                if current_cd <= cmd:
                    break

        prev_cd = current_cd
        modify_chord_density[i] = current_cd

    new_chord_bin = [[] for _ in range(len(chord_density))]
    num_prolonged, num_moved, num_dropped = 0, 0, 0
    num_note = 0

    for i in range(len(chord_density)):
        cb = chord_bin[i]
        modify = modify_list[i]
        num_note += 1
        for j, pid in enumerate(cb):
            if j in [m[0] for m in modify]:
                status = modify[[m[0] for m in modify].index(j)][1]
                if status == 'r':
                    num_prolonged += 1
                elif status == 'd':
                    num_dropped += 1
                else:
                    num_moved += 1
                    new_chord_bin[i + 1].append(note_mat[path[pid]])
            else:
                new_chord_bin[i].append(note_mat[path[pid]])
    postprocess_reduction_rate = 1 - num_dropped / num_note
    # compute duration
    report = {'num_prolonged': num_prolonged,
              'num_moved': num_moved,
              'num_dropped': num_dropped,
              'red_rate_0': reduction_rate,
              'red_rate_1': postprocess_reduction_rate,
              'red_rate_final': postprocess_reduction_rate * reduction_rate}
    return new_chord_bin, report


def chord_bins_to_reduction_mat(chord_mat, path_bin, nspb):
    if path_bin is None:
        return np.zeros((0, 3), dtype=np.int64)

    notes = []
    # note_bin = []
    for i, chord in enumerate(chord_mat):
        if len(path_bin[i]) == 0:
            continue
        chord_start = chord[0]
        chord_end = chord[-1] + chord_start
        n_note = len(path_bin[i])
        len_chord = chord_end - chord_start

        if len_chord == 1:
            assert n_note <= 1
            durs = [1.]
        elif len_chord == 2:
            assert n_note <= 2
            durs = [2.] if n_note == 1 else [1., 1.]
        elif len_chord == 3:
            assert n_note <= 3
            if n_note == 1:
                durs = [3.]
            elif n_note == 2:
                durs = [2., 1.]
            else:
                durs = [1., 1., 1.]
        elif len_chord == 4:
            if n_note == 1:
                durs = [4.]
            elif n_note == 2:
                durs = [2., 2.]
            elif n_note == 3:
                durs = [2., 1., 1.]
            elif n_note == 4:
                durs = [1., 1., 1., 1.]
            else:
                raise AssertionError
        else:
            assert n_note <= len_chord
            durs = [len_chord - n_note + 1.] + [1.] * (n_note - 1)

        cur_start = chord_start * nspb

        for note_id, note in enumerate(path_bin[i]):
            ict = chord_mat[i, int(note[1] % 12) + 1] == 1
            dur = durs[note_id] * nspb
            notes.append((cur_start, note[1], dur, ict, i))
            cur_start += dur

    new_notes = [list(notes[0])[0: 3]]
    for i in range(1, len(notes)):
        if notes[i][1] == notes[i - 1][1] and (
                (not notes[i - 1][3] and notes[i][3]) or notes[i - 1][4] ==
                notes[i][4]):
            new_notes[-1][2] = notes[i][0] - new_notes[-1][0] + notes[i][2]
        else:
            new_notes[-1][2] = notes[i][0] - new_notes[-1][0]
            new_notes.append(list(notes[i])[0: 3])
    new_notes = np.array(new_notes).astype(np.int64)
    return new_notes
