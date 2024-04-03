from .shortest_path_algo import find_tonal_shortest_paths
from .preprocess import preprocess_data
from .postprocess import path_to_chord_bins, chord_bins_to_reduction_mat
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np


class TrAlgo:

    def __init__(self, distance_factor=1.6, onset_factor=1.0, chord_factor=1.0, pitch_factor=1.0,
                 duration_factor=1.0):

        self.distance_factor = distance_factor
        self.onset_factor = onset_factor
        self.chord_factor = chord_factor
        self.pitch_factor = pitch_factor
        self.duration_factor = duration_factor

        self._note_mat = None
        self._chord_mat = None

        self._num_beat_per_measure = None
        self._num_step_per_beat = None

        self._start_measure = None

        self._reduction_mats = None

        self._report = None

    def preprocess_data(self, note_mat, chord_mat,
                        start_measure, num_beat_per_measure, num_step_per_beat):
        """
        Analyze the phrase and add music attributes to note mat. The columns becomes:
        [onsets, pitches, durations, bar_id, tonal_bar_id, chord_ids, tonal_chord_ids, is_chord_tone, tonal_onsets]
        """
        def measure_to_step_fn(measure):
            return measure * num_beat_per_measure * num_step_per_beat

        def measure_to_beat_fn(measure):
            return measure * num_beat_per_measure

        def step_to_beat_fn(step):
            return step // num_step_per_beat

        def step_to_measure_fn(step):
            return step // (num_step_per_beat * num_beat_per_measure)

        def beat_to_measure_fn(beat):
            return beat // num_beat_per_measure

        def beat_to_step_fn(beat):
            return beat * num_step_per_beat

        note_mat, chord_mat = preprocess_data(
            note_mat, chord_mat, start_measure,
            measure_to_step_fn, measure_to_beat_fn, step_to_beat_fn,
            step_to_measure_fn, beat_to_measure_fn, beat_to_step_fn
        )

        self.fill_data(note_mat, chord_mat, start_measure, num_beat_per_measure,
                       num_step_per_beat)

    def algo(self, num_path=1, plot_graph=False):
        # find the top-k shortest paths and compute distance
        paths, G = find_tonal_shortest_paths(
            self._note_mat, self._num_beat_per_measure,
            self._num_step_per_beat, num_path,
            self.distance_factor, self.onset_factor,
            self.chord_factor, self.pitch_factor, self.duration_factor)

        if plot_graph:
            print('The current version can only print one shortest path.')
            self.plot_graph(G, paths[0])

        return paths

    def postprocess_paths(self, paths):
        # use fixed rhythm template to compose melody reduction
        chord_bins, reduction_report = \
            zip(*[path_to_chord_bins(path, self._note_mat, self._chord_mat) for path in paths])

        reduction_mats = [chord_bins_to_reduction_mat(self._chord_mat, cb, self._num_step_per_beat)
                          for cb in chord_bins]

        self._reduction_mats = reduction_mats
        self._report = reduction_report

    def output(self, start_measure=None):
        start_measure = start_measure if start_measure is not None else \
            self._start_measure
        start_beat = start_measure * self._num_beat_per_measure
        start_step = start_beat * self._num_step_per_beat

        note_mat = self._note_mat.copy()
        note_mat[:, 0] += start_step

        chord_mat = self._chord_mat.copy()
        chord_mat[:, 0] += start_beat

        reduction_mats = self._reduction_mats.copy()
        for red_mat in reduction_mats:
            red_mat[:, 0] += start_step
        return note_mat, chord_mat, reduction_mats

    def run(self, note_mat, chord_mat, start_measure, num_beat_per_measure,
            num_step_per_beat, num_path=1, plot_graph=False):

        # analyze input melody phrase and add music attributes to note_mat. Translate phrase start to zero.
        self.preprocess_data(note_mat, chord_mat, start_measure,
                             num_beat_per_measure, num_step_per_beat)

        # run shortest-path algorithm.
        paths = self.algo(num_path, plot_graph=plot_graph)

        # use fixed rhythm template to compose reduced melody.
        self.postprocess_paths(paths)

        # Translate phrase to actual phrase start.
        note_mat, chord_mat, reduction_mats = self.output()

        # clear stored data
        self.clear_data()

        return note_mat, chord_mat, reduction_mats

    def plot_graph(self, G, path):
        fig = plt.figure(figsize=(8, 6))
        if len(G.nodes) == 0:
            return
        cmap = plt.cm.Greys
        pos_dict = {n: (self._note_mat[i, 0], self._note_mat[i, 1]) for i, n in enumerate(G.nodes)}
        edges, weights = zip(*nx.get_edge_attributes(G, 'weight').items())
        weights = tuple(-w for w in weights)
        ax = plt.subplot()
        nx.draw_networkx(G, node_size=100, node_color='black',
                         pos=pos_dict, edge_color=weights, edge_cmap=cmap)
        ax.set_axis_on()
        ax.tick_params(left=True, bottom=True, labelleft=True, labelbottom=True)
        ax.set_xticks(np.arange(0, self._note_mat[:, 0].max(), self._num_step_per_beat))

        new_G = nx.DiGraph()
        for i in range(len(self._note_mat)):
            G.add_node(i, data=self._note_mat[i])

        path = path['path']
        for i in range(len(path) - 1):
            new_G.add_edge(path[i], path[i + 1])
        nx.draw_networkx_edges(new_G, pos_dict, edge_color='red')

        ax.grid()
        plt.show()

    def fill_data(self, note_mat, chord_mat, start_measure,
                  num_beat_per_measure, num_step_per_beat):
        self._note_mat = note_mat
        self._chord_mat = chord_mat
        self._start_measure = start_measure
        self._num_beat_per_measure = num_beat_per_measure
        self._num_step_per_beat = num_step_per_beat

    def clear_data(self):
        self._note_mat = None
        self._chord_mat = None
        self._num_beat_per_measure = None
        self._num_step_per_beat = None
        self._start_measure = None
