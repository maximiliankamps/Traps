import Storage
import graphviz as gviz
from abc import ABC, abstractmethod
from itertools import *
import json
import re

import Util


def hash_state(column_list, byte_length):
    """Combines all states in column_list in to a string and returns a hash of byte_length"""
    state_str = ""
    for state in column_list:
        state_str += str(state + 1)  # Important!!! + 1 to generate unique hash for columns with q0 states
    return int(state_str)
    # return int.from_bytes(hexlify(shake.read(byte_length)), 'big')


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
        self.partial_sigma_origin = set()  # contains all used origin symbols
        self.partial_sigma_target = set()  # contains all used target symbols
        self.transitions = Storage.SimpleStorageNFA()
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
        if state not in self.final_states:
            self.final_states.append(state)

    def add_final_state_list(self, state_list):
        self.final_states.extend(state_list)

    def get_final_states(self):
        return self.final_states

    def get_alphabet_map(self):
        return self.alphabet_map

    def add_transition(self, origin, symbol_index, target):
        self.partial_sigma_origin.add(self.alphabet_map.get_x(symbol_index))
        self.partial_sigma_target.add(self.alphabet_map.get_y(symbol_index))
        self.transitions.add_transition(origin, symbol_index, target)

    def get_transitions(self, origin):
        yield from self.transitions.transition_iterator(origin)

    def get_successor(self, origin, symbol_index):
        return self.transitions.get_successor(origin, symbol_index)

    def state_iterator(self):
        return self.transitions.state_iterator()

    def copy_inverted(self):
        """Create a copy of the transducer and invert all transitions and final and initial states"""
        copy = NFATransducer(self.alphabet_map)
        copy.initial_states = self.final_states
        copy.final_states = self.initial_states
        for q in self.state_iterator():
            for (symbol, p) in self.transitions.transition_iterator(q):
                copy.add_transition(p, symbol, q)
        return copy

    def copy_with_restricted_trans(self, origin_symbols, target_symbols):
        """
        Create a copy of the transducer and remove all transitions where:
        :param origin_symbols: x not in origin symbol or
        :param target_symbols: y not in target symbol
        :return:
        """
        copy = NFATransducer(self.alphabet_map)
        copy.initial_states = self.initial_states
        for q in self.state_iterator():
            for (symbol, p) in self.transitions.transition_iterator(q):
                if self.alphabet_map.get_x(symbol) in origin_symbols and self.alphabet_map.get_y(
                        symbol) in target_symbols:
                    copy.add_transition(q, symbol, p)
                    if p in self.final_states:
                        copy.add_final_state(p)
        return copy

    def to_dot(self, filename, column_hashing):
        g = gviz.Digraph('G', filename="Pictures/" + f'{filename}')

        for source in self.state_iterator():
            for (symbol, target) in self.transitions.transition_iterator(source):
                x = self.alphabet_map.int_to_symbol(self.alphabet_map.get_x(symbol))
                y = self.alphabet_map.int_to_symbol(self.alphabet_map.get_y(symbol))
                if target is not None:
                    if column_hashing is not None:
                        g.node(column_hashing.get_column_str(source), column_hashing.get_column_str(source),
                               shape="circle")
                        g.edge(column_hashing.get_column_str(source),
                               column_hashing.get_column_str(target),
                               x + "\n" + y)
                    else:
                        g.node(str(source), str(source), shape="circle")
                        g.edge(str(source), str(target), x + "\n" + y)

        g.view()

    def join(self, nfa):
        alph_map = self.get_alphabet_map()
        T_new = NFATransducer(self.alphabet_map)
        W = list(product(self.get_initial_states(), nfa.get_initial_states()))
        Q = []
        c_hash = Storage.ColumnHashing(True)

        # Add the initial states to T_new
        T_new.add_initial_state_list(list(map(lambda x: hash_state(x, 1), W)))

        while W:
            (q1, q2) = W.pop(0)
            q1_q2_hash = hash_state([q1, q2], 1)
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
                            q1_q2_target_hash = hash_state([q1_target, q2_target], 1)
                            a_b = alph_map.combine_x_and_y(alph_map.get_x(a_c), alph_map.get_y(c_b))
                            c_hash.store_column(q1_q2_target_hash, [q1_target, q2_target])

                            if alph_map.get_y(a_c) == alph_map.get_x(c_b):
                                if q1_q2_target_hash in Util.optional_list(T_new.get_successor(q1_q2_hash, a_b)):
                                    continue
                                if q1_q2_target_hash not in Q:
                                    W.append((q1_target, q2_target))
                                T_new.add_transition(q1_q2_hash, a_b, q1_q2_target_hash)
        return T_new

    def nfa_to_dfa(self):
        result = NFATransducer(self.alphabet_map)
        work_queue = [self.initial_states.copy()]
        visited = [self.initial_states.copy()]
        result.initial_states = list(map(lambda x: hash_state([x], 0), self.initial_states.copy()))

        while len(work_queue) != 0:
            q_list = work_queue.pop(0)

            new_q = hash_state(q_list, 0)
            if any(map(lambda x: x in self.final_states, q_list)):
                result.add_final_state(new_q)

            for t in self.alphabet_map.sigma_x_sigma_iterator():
                p_gen = filter(lambda x: x is not None, map(lambda q: self.get_successor(q, t), q_list))
                p_list = list(set(chain.from_iterable(p_gen)))
                if p_list:
                    new_p = hash_state(p_list, 0)
                    if p_list not in visited:
                        work_queue.append(p_list)
                        visited.append(p_list)
                    result.add_transition(new_q, t, new_p)
        return result

    def all_transitions(self):
        result = []
        for o in self.state_iterator():
            result.extend(
                list(map(lambda x: (o, self.alphabet_map.get_x(x[0]), x[1]), self.transitions.transition_iterator(o))))
        return result

    def project_origin(self):
        result = NFATransducer(self.alphabet_map)
        result.initial_states = self.initial_states.copy()
        result.final_states = self.final_states.copy()
        for q in self.state_iterator():
            for (symbol, p) in self.get_transitions(q):
                origin = self.alphabet_map.get_x(symbol)
                result.add_transition(q, self.alphabet_map.combine_x_and_y(origin, origin), p)
        return result

    def get_deadlock_transducer(self):
        D = self.project_origin().nfa_to_dfa()
        D.final_states = list(set(D.state_iterator()).difference(set(D.final_states.copy())))
        return D

    def minimize(self):
        return 0


def parse_transition_regex(regex, alph_map, id):
    """
    :param regex: The regex for transition from .json file
    :param alph_map: Maps string transitions symbols to bit encoding
    :param id: If set to true returns id transitions for I and B
    :return: A list of transition symbols
    """
    m = (map(lambda s: s[0] + "," + s[1],  # create a list of sigma,sigma
             (product(alph_map.sigma, alph_map.sigma), zip(alph_map.sigma, alph_map.sigma))[id]))
    r = re.compile(((regex, f'{regex},{regex}')[id]))
    return list(map(lambda z: alph_map.combine_symbols(z[0], z[1]),  # map x y -> int
                    (map(lambda y: y.split(","),  # remove the ','
                         filter(lambda x: r.match(x), m)))))  # match all x,y that satisfy the pattern


def parse_transition_regex_dfa(trans_dict, alph_map):
    """
    :param trans_dict: The regex for transition from .json file
    :param alph_map: Maps string transitions symbols to bit encoding
    :return: A list of transitions (q, x, p)
    """
    transitions = []
    for t in trans_dict:
        r = re.compile(t["letter"])
        q = int(t["origin"][1:])
        p = int(t["target"][1:])
        transitions.extend(
            list(map(lambda y: (q, alph_map.symbol_to_int(y), p), filter(lambda x: r.match(x), alph_map.sigma))))
    return transitions


class RTS:
    def __init__(self, filename):
        self.IxB_dict = None
        self.B_dict = None
        self.I = None
        self.T = None
        self.alphabet_map = None
        self.rts_from_json(filename)

    def get_I(self):
        return self.I

    def get_T(self):
        return self.T

    def get_B(self, property_name):
        return self.B_dict[property_name]

    def get_IxB(self, property_name):
        return self.IxB_dict[property_name]

    def rts_from_json(self, filename):
        file = open(f'benchmark/{filename}')
        rts_dict = json.load(file)
        alphabet_map = Storage.AlphabetMap(rts_dict["alphabet"])
        self.alphabet_map = alphabet_map

        initial_dict = rts_dict["initial"]
        transducer_dict = rts_dict["transducer"]
        properties_dict = rts_dict["properties"]
        deadlock_threshold = rts_dict["deadlockThreshold"]

        self.T = self.build_transducer(transducer_dict, alphabet_map, False)
        self.I = self.build_transducer(initial_dict, alphabet_map, True)

        self.B_dict = {name: self.build_transducer(properties_dict[name], alphabet_map, True) for name in
                       properties_dict}

        self.IxB_dict = {name: self.build_IxB_transducer(initial_dict, properties_dict[name]) for name in
                         properties_dict}

        self.IxB_dict["deadlock"] = self.build_SxD_transducer(deadlock_threshold)

        self.alphabet_map = alphabet_map

    def pair_transducers(self, q0, p0, t1, t2, f1, f2):
        result = NFATransducer(self.alphabet_map)
        result.add_initial_state(hash_state([q0, p0], 0))

        Q = [(q0, p0)]
        W = []

        while len(Q) != 0:
            (q1, q2) = Q.pop(0)
            W.append((q1, q2))

            if q1 in f1 and q2 in f2:
                result.add_final_state(hash_state([q1, q2], 0))

            for (q1_, x, p1) in t1:
                for (q2_, y, p2) in t2:
                    if q1 == q1_ and q2 == q2_:
                        q1_q2_hash = hash_state([q1_, q2_], 0)
                        p1p2hash = hash_state([p1, p2], 0)
                        symbol = self.alphabet_map.combine_x_and_y(x, y)
                        if result.get_successor(q1_q2_hash, symbol) is None or p1p2hash not in result.get_successor(
                                q1_q2_hash, symbol):
                            result.add_transition(q1_q2_hash, symbol, p1p2hash)
                        if (p1, p2) not in W:
                            Q.append((p1, p2))
        return result

    def build_IxB_transducer(self, I_dict, B_dict):
        t1 = parse_transition_regex_dfa(I_dict["transitions"], self.alphabet_map)
        f1 = list(map(lambda q: int(q[1:]), I_dict["acceptingStates"]))

        t2 = parse_transition_regex_dfa(B_dict["transitions"], self.alphabet_map)
        f2 = list(map(lambda q: int(q[1:]), B_dict["acceptingStates"]))

        q0 = int(I_dict["initialState"][1:])
        p0 = int(B_dict["initialState"][1:])

        return self.pair_transducers(q0, p0, t1, t2, f1, f2)

    def build_SxD_transducer(self, deadlock_threshold):
        D = self.T.get_deadlock_transducer()
        t1 = D.all_transitions()
        f1 = D.final_states

        t2 = []
        for i in range(0, deadlock_threshold):
            for x in self.alphabet_map.sigma_iterator():
                t2.append((i, x, i+1))
                if i == deadlock_threshold-1: # self loops at last state
                    t2.append((i+1, x, i+1))
        f2 = [deadlock_threshold-1]
        print(t2)

        q0 = D.get_initial_states()[0]
        p0 = 0

        return self.pair_transducers(q0, p0, t1, t2, f1, f2)


    def built_id_transducer(self, nfa_dict, alph_map):
        id_transducer = NFATransducer(alph_map)
        id_transducer.add_initial_state(int(nfa_dict["initialState"][1:]))
        id_transducer.add_final_state_list(list(map(lambda q: int(q[1:]), nfa_dict["acceptingStates"])))
        for t in nfa_dict["transitions"]:
            letter = t["letter"]
            symbol = alph_map.combine_symbols(letter, letter)
            id_transducer.add_transition(int(t["origin"][1:]), symbol, int(t["target"][1:]))
        return id_transducer

    def build_transducer(self, trans_dict, alph_map, id):
        transducer = NFATransducer(alph_map)
        transducer.set_state_count(len(trans_dict["states"]))
        transducer.add_initial_state(int(trans_dict["initialState"][1:]))
        transducer.add_final_state_list(list(map(lambda q: int(q[1:]), trans_dict["acceptingStates"])))
        for t in trans_dict["transitions"]:
            for symbol in parse_transition_regex(t["letter"], alph_map, id):
                transducer.add_transition(int(t["origin"][1:]), symbol, int(t["target"][1:]))
        return transducer
