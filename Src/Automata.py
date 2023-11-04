import Algorithms
import Storage
import graphviz as gviz
from abc import ABC, abstractmethod
from itertools import *
import json
import re

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

    def add_initial_state_list(self, initial_state_list):
        self.initial_states.extend(initial_state_list)

    def is_final_state(self, state):
        return state in self.final_states

    def add_final_state(self, state):
        self.final_states.append(state)

    def add_final_state_list(self, state_list):
        self.final_states.extend(state_list)

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

        for source in range(0, 100):
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
        T_new.add_initial_state_list(list(map(lambda x: Algorithms.hash_state(x, 1), W)))

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


class RTS:
    def __init__(self, filename):
        self.I = None
        self.T = None
        self.B_dict = None
        self.rts_from_json(filename)

    def get_I(self):
        return self.I

    def get_T(self):
        return self.T

    def get_B(self, property_name):
        return self.B_dict[property_name]

    def rts_from_json(self, filename):
        file = open(f'benchmark/{filename}')
        rts_dict = json.load(file)
        alphabet_map = Storage.AlphabetMap(rts_dict["alphabet"])

        initial_dict = rts_dict["initial"]
        transducer_dict = rts_dict["transducer"]
        properties_dict = rts_dict["properties"]

        self.I = self.built_id_transducer(initial_dict, alphabet_map)
        self.T = self.build_transducer(transducer_dict, alphabet_map)

        self.B_dict = {name: self.built_id_transducer(properties_dict[name], alphabet_map) for name in properties_dict}

    def built_id_transducer(self, nfa_dict, alph_map):
        id_transducer = NFATransducer(alph_map)
        id_transducer.add_initial_state(int(nfa_dict["initialState"][1:]))
        id_transducer.add_final_state_list(list(map(lambda q: int(q[1:]), nfa_dict["acceptingStates"])))
        for t in nfa_dict["transitions"]:
            letter = t["letter"]
            symbol = alph_map.combine_symbols(letter, letter)
            id_transducer.add_transition(int(t["origin"][1:]), symbol, int(t["target"][1:]))
        return id_transducer

    def build_transducer(self, trans_dict, alph_map):
        transducer = NFATransducer(alph_map)
        transducer.set_state_count(len(trans_dict["states"]))
        transducer.add_initial_state(int(trans_dict["initialState"][1:]))
        transducer.add_final_state_list(list(map(lambda q: int(q[1:]), trans_dict["acceptingStates"])))
        for t in trans_dict["transitions"]:
            (x, y) = tuple(self.parse_transition_regex(t["letter"]))
            symbol = alph_map.combine_symbols(x, y)
            transducer.add_transition(int(t["origin"][1:]), symbol, int(t["target"][1:]))
        return transducer

    def parse_transition_regex(self, regex):
        return regex.split(",")
