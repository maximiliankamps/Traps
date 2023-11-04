import Algorithms
import Storage
import graphviz as gviz
from abc import ABC, abstractmethod
from itertools import *

import Util


class AbstractTransducer(ABC):
    @abstractmethod
    def __init__(self, alphabet_map):
        pass

    @abstractmethod
    def add_final_state(self, state):
        pass

    @abstractmethod
    def is_final_state(self, state):
        pass

    @abstractmethod
    def get_final_states(self):
        pass


    @abstractmethod
    def get_alphabet_map(self):
        pass

    @abstractmethod
    def add_transition(self, origin, symbol, target):
        pass

    @abstractmethod
    def get_successor(self, origin, symbol_index):
        pass


class NFATransducer(AbstractTransducer):

    def __init__(self, alphabet_map):
        self.state_count = 0
        self.initial_states = []
        self.final_states = []
        self.alphabet_map = alphabet_map
        self.transitions = Storage.SimpleStorageNFA(0, alphabet_map.get_num_symbols_in_sigma_x_sigma())
        self.statistics = Storage.Statistics()

    def set_state_count(self, state_count):
        self.state_count = state_count
    def get_state_count(self):
        return self.state_count

    def get_initial_states(self):
        return self.initial_states

    def add_initial_state(self, initial_state):
        self.initial_states.append(initial_state)

    def init_initial_states(self, initial_state_list):
        self.initial_states = initial_state_list


    def is_final_state(self, state):
        return state in self.final_states

    def add_final_state(self, state):
        self.final_states.append(state)

    def get_final_states(self):
        return self.final_states

    def get_alphabet_map(self):
        return self.alphabet_map

    def add_transition(self, origin, symbol_index, target):
        self.statistics.log_transition()
        self.transitions.add_transition(origin, symbol_index, target)

    def get_successor(self, origin, symbol_index):
        return self.transitions.get_successor(origin, symbol_index)

    def state_iterator(self):
        return self.transitions.state_iterator()

    def to_dot(self, filename, column_hashing):
        g = gviz.Digraph('G', filename="Pictures/" + f'{filename}')

        for source in range(0, 100000):
            for x in self.alphabet_map.sigma:
                for y in self.alphabet_map.sigma:
                    target = self.get_successor(source, self.alphabet_map.combine_symbols(x, y))
                    if target is not None:
                        for target_node in target:

                            if column_hashing is not None:
                                g.node(column_hashing.get_column_str(source), column_hashing.get_column_str(source),
                                       shape="circle")
                                g.edge(column_hashing.get_column_str(source),
                                       column_hashing.get_column_str(target_node),
                                       x + "\n" + y)
                            else:
                                g.node(str(source), str(source), shape="circle")
                                g.edge(str(source), str(target_node), x + "\n" + y)

        g.view()

    def join(self, nfa):
        alph_map = self.get_alphabet_map()
        T_new = NFATransducer(self.alphabet_map)
        W = list(product(self.get_initial_states(), nfa.get_initial_states()))
        Q = []
        c_hash = Storage.ColumnHashing(True)

        # Add the initial states to T_new
        T_new.init_initial_states(list(map(lambda x: Algorithms.hash_state(x, 1), W)))

        while W:
            (q1, q2) = W.pop(0)
            q1_q2_hash = Algorithms.hash_state([q1, q2], 1)
            c_hash.store_column(q1_q2_hash, [q1, q2])

            if self.is_final_state(q1) and nfa.is_final_state(q2) and q1_q2_hash not in T_new.get_final_states():
                T_new.add_final_state(q1_q2_hash)

            Q.append(q1_q2_hash)
            for a_c in alph_map.sigma_x_sigma_iterator():
                for c_b in alph_map.sigma_x_sigma_iterator():
                    q1_target_list = self.get_successor(q1, a_c)
                    q2_target_list = nfa.get_successor(q2, c_b)

                    if q1_target_list is not None and q2_target_list is not None:
                        for (q1_target, q2_target) in product(q1_target_list, q2_target_list):
                            q1_q2_target_hash = Algorithms.hash_state([q1_target, q2_target], 1)
                            a_b = alph_map.combine_x_and_y(alph_map.get_x(a_c), alph_map.get_y(c_b))
                            c_hash.store_column(q1_q2_target_hash, [q1_target, q2_target])

                            if alph_map.get_y(a_c) == alph_map.get_x(c_b):
                                if q1_q2_target_hash in Util.optional_list(T_new.get_successor(q1_q2_hash, a_b)):
                                    continue
                                if q1_q2_target_hash not in Q:
                                    W.append((q1_target, q2_target))
                                T_new.add_transition(q1_q2_hash, a_b, q1_q2_target_hash)
        return T_new


class Transducer(ABC):
    def __init__(self, state_count, alphabet_map):
        self.state_count = state_count
        self.initial_state = -1
        self.final_states = []
        self.alphabet_map = alphabet_map
        self.transitions = Storage.SparseStorage(state_count, alphabet_map.get_num_symbols_in_sigma_x_sigma())
        self.statistics = Storage.Statistics()

    def get_initial_state(self):
        return self.initial_state

    def set_initial_state(self, initial_state):
        self.initial_state = initial_state

    def is_final_state(self, state):
        return state in self.final_states

    def add_final_state(self, state):
        self.final_states.append(state)

    def get_final_states(self):
        return self.final_states

    def get_alphabet_map(self):
        return self.alphabet_map

    def get_state_count(self):
        return self.state_count

    def add_transition(self, origin, symbol_index, target):
        self.statistics.log_transition()
        self.transitions.add_transition(origin, symbol_index, target)

    def get_successor(self, origin, symbol_index):
        return self.transitions.get_successor(origin, symbol_index)

    def to_dot(self, filename, as_bin):
        g = gviz.Digraph('G', filename="Pictures/" + f'{filename}')

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
