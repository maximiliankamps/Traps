import math

import Storage
import graphviz as gviz


class NFATransducer:
    def __init__(self, state_count, alphabet_map):
        self.state_count = state_count
        self.states = []
        self.alphabet_map = alphabet_map
        self.transitions = Storage.SimpleStorageNFA(state_count, alphabet_map.get_num_symbols_in_sigma_x_sigma())
        self.statistics = Storage.Statistics()

    def get_alphabet_map(self):
        return self.alphabet_map

    def get_state_count(self):
        return self.state_count

    def add_transition(self, origin, symbol_index, target):
        if origin not in self.states:
            self.states.append(origin)
        self.statistics.log_transition()
        self.transitions.add_transition(origin, symbol_index, target)

    def get_successor(self, origin, symbol_index):
        return self.transitions.get_successor(origin, symbol_index)

    def dot_string(self, filename, column_hashing):
        g = gviz.Digraph('G', filename=f'{filename}')

        for source in self.states:
            for x in self.alphabet_map.sigma:
                for y in self.alphabet_map.sigma:
                    target = self.get_successor(source, self.alphabet_map.combine_symbols(x, y))
                    if target is not None:
                        for target_node in target:
                            if column_hashing is not None:
                                g.node(column_hashing.get_column_str(source), column_hashing.get_column_str(source))
                                g.edge(column_hashing.get_column_str(source), column_hashing.get_column_str(target_node),
                                       x + "\n" + y)
                            else:
                                g.node(str(source), str(source))
                                g.edge(str(source), str(target_node), x + "\n" + y)

        g.view()


class Transducer:
    def __init__(self, state_count, alphabet_map):
        self.state_count = state_count
        self.alphabet_map = alphabet_map
        self.transitions = Storage.SparseStorage(state_count, alphabet_map.get_num_symbols_in_sigma_x_sigma())
        self.statistics = Storage.Statistics()

    def get_alphabet_map(self):
        return self.alphabet_map

    def get_state_count(self):
        return self.state_count

    def add_transition(self, origin, symbol_index, target):
        self.statistics.log_transition()
        self.transitions.add_transition(origin, symbol_index, target)

    def get_successor(self, origin, symbol_index):
        return self.transitions.get_successor(origin, symbol_index)

    def dot_string(self, filename, as_bin):
        g = gviz.Digraph('G', filename=f'{filename}')

        for source in range(0, self.state_count):
            for x in self.alphabet_map.sigma:
                for y in self.alphabet_map.sigma:
                    target = self.get_successor(source, self.alphabet_map.combine_symbols(x, y))
                    if target != -1:
                        if as_bin:
                            g.node(bin(source), bin(source))
                            g.edge(bin(source), bin(target), x + "\n" + y)
                        else:
                            g.node(str(source), str(source))
                            g.edge(str(source), str(target), x + "\n" + y)

        g.view()
