import math

import Storage
import graphviz as gviz


class Transducer:
    def __init__(self, state_count, alphabet_map):
        self.state_count = state_count
        self.alphabet_map = alphabet_map
        self.transitions = Storage.SparseStorage(state_count, alphabet_map.get_num_csym())
        self.statistics = Storage.Statistics()

    def get_alphabet_map(self):
        return self.alphabet_map

    def get_state_count(self):
        return self.state_count

    def add_transition(self, origin, symbol, target):
        self.statistics.log_transition()
        self.transitions.add_transition(origin, symbol, target)

    def get_successor(self, origin, symbol):
        return self.transitions.get_successor(origin, symbol)

    def dot_string(self, filename):
        g = gviz.Digraph('G', filename=f'{filename}')

        for source in range(1, self.state_count+1):
            for x in self.alphabet_map.sigma:
                for y in self.alphabet_map.sigma:
                    target = self.get_successor(source, self.alphabet_map.combine_symbols(x, y))
                    if target != -1:
                        g.node(str(source), str(source))
                        g.edge(str(source), str(target), x + "\n" + y)
        g.view()
