"""Different one shot implementations"""
import itertools

from Automata import NFATransducer, hash_state
import numpy as np
from Util import *
from itertools import chain, product, permutations
from abc import ABC, abstractmethod
from itertools import *
import math
from pyeda.inter import *
import bitarray

import multiprocessing as mp


def multi_disprove_oneshot(IxB, T):
    init_states = T.get_final_states()
    init_permutations = chain.from_iterable(permutations(init_states, r) for r in range(1, len(init_states) + 1))

    one_shot_instances = {}
    for init in init_permutations:
        one_shot_instances[init] = OneshotSmart(IxB, T.copy_inverted())

    processes = []
    manager = mp.Manager()
    return_code = manager.dict()
    run = manager.Event()
    run.set()  # We should keep running.
    for init_state in one_shot_instances.keys():
        process = mp.Process(
            one_shot_instances[init_state].one_shot_dfs_process(run, (IxB.get_initial_states()[0], init_state), return_code)
        )
        processes.append(process)
        process.start()

    for process in processes:
        process.join()

    print(return_code)


class OneshotSmart:
    """OneShot implementation similar to the one in dodo"""

    def __init__(self, IxB, T):
        self.IxB = IxB
        self.T = T
        self.alphabet_map = T.get_alphabet_map()
        self.step_cache = self.StepGameCache()
        self.i = 0  # keeps count of the number of explored states
        self.trans = 0

    class StepGameCache:
        def __init__(self):
            self.cache = {}
            self.cache_hits = 0

        def add_entry(self, c, game_state, v, d_current, d_winning):
            self.cache[(tuple(c), game_state.l, game_state.I, v, tuple(d_current))] = d_winning

        def get_entry(self, c, game_state, v, d_current):
            look_up = self.cache.get((tuple(c), game_state.l, game_state.I, v, tuple(d_current)))
            if look_up is not None:
                self.cache_hits += 1
            return look_up

        def print(self):
            for key in self.cache:
                print(f'{key} -> {self.cache[key]}')

    def min_sigma_disprove_oneshot(self):
        """
        Restrict the alphabet of T by the partial target alphabet of I and B.
        One_shot result is only valid when it is False!
        :return: False if property was disproved
        """
        self.T = self.T.copy_with_restricted_trans(self.IxB.partial_sigma_origin, self.IxB.partial_sigma_target)
        value = self.one_shot_dfs_standard()
        if not value:
            print("Property could not be established!")
        return value

    def one_shot_dfs_process(self, run, initial_tuple, return_dict):
        """
        A one shot instance for multi disproving
        :param run: Signal if an instance has already found a counter example
        :param initial_tuple: the initial tuple of this instance
        :param return_dict: A dictionary for the return value of one shot
        :return:
        """
        print(f'{initial_tuple} started')
        while run.is_set():
            result = self.one_shot_dfs(initial_tuple)
            return_dict[str(initial_tuple)] = result
            if result is not None and len(result):
                run.clear()

    def one_shot_dfs_standard(self):
        """Explore reduced seperator transducer from initial state pair in dfs"""
        return self.one_shot_dfs((self.IxB.get_initial_states()[0], [self.T.get_initial_states()[0]]))

    def one_shot_dfs(self, initial_tuple):
        """Explore reduced seperator transducer from initial_tuple in dfs and return the first counterexample"""
        (ib0, c0) = initial_tuple
        visited_states = {(ib0, tuple(c0))}
        for a in self.one_shot_dfs_helper(ib0, c0, visited_states):
            return a
        return None

    def one_shot_dfs_helper(self, ib, c, visited_states):
        # iterate over all transitions of the state ixb
        for (ib_trans, ib_succ) in self.IxB.get_transitions(ib):
            u, v = self.alphabet_map.get_y(ib_trans), self.alphabet_map.get_x(ib_trans)
            gs = Triple(0, refine_seperator(self.alphabet_map.get_bit_map_sigma(), u), 0)

            # iterate over all reachable (ib ∩ c) -> (ib_successor ∩ d)
            for d in self.step_game_gen_buffered_dfs(c, [], v, gs, []):
                self.trans += 1
                if (ib_succ, tuple(d)) not in visited_states:
                    visited_states.add((ib_succ, tuple(d)))
                    self.i += 1
                    #print(self.i)
                    if self.IxB.is_final_state(ib_succ) and len(
                            list((filter(lambda q: (not self.T.is_final_state(q)), d)))) == 0:
                        #print(ib_succ, d)
                        yield ib_succ, d
                    yield from self.one_shot_dfs_helper(ib_succ, d, visited_states)

    def print_oneshot_result(self, result_bool):
        print("# states: " + str(self.i))
        print("# cache hits: " + str(self.step_cache.cache_hits))
        print("# transitions: " + str(self.trans))
        if result_bool is None:
            print("Result: ✓")
        else:
            print("Result: x")

    # TODO: implement one_shot_dfs with optimal cashing -> pick next state for which cashing entries exist
    def one_shot_bfs(self):
        """Explores the IxB ∩ (reduced seperator transducer) in a breath first search"""
        # Pairing of the initial states of ixb ∩ (reduced seperator transducer)
        (ib0, c0) = (self.IxB.get_initial_states()[0], [self.T.get_initial_states()[0]])
        work_set = [(ib0, c0)]
        visited_states = {(ib0, tuple(c0))}
        trans = 0
        while len(work_set) != 0:
            (ib, c) = work_set.pop(0)
            # iterate over all transitions of the state ixb
            for (ib_trans, ib_succ) in self.IxB.get_transitions(ib):

                u, v = self.alphabet_map.get_y(ib_trans), self.alphabet_map.get_x(ib_trans)
                gs = Triple(0, refine_seperator(self.alphabet_map.get_bit_map_sigma(), u), 0)

                # iterate over all reachable (ib ∩ c) -> (ib_successor ∩ d)
                for d in self.step_game_gen_buffered_dfs(c, [], v, gs, []):
                    trans += 1
                    if (ib_succ, tuple(d)) not in visited_states:
                        visited_states.add((ib_succ, tuple(d)))
                        work_set.append((ib_succ, d))
                        self.i += 1
                        if self.IxB.is_final_state(ib_succ) and len(
                                list((filter(lambda q: (not self.T.is_final_state(q)), d)))) == 0:
                            return ib_succ, d
        return None

    def step_game_gen_simple_dfs(self, c1, c2, v, gs, visited):
        """
        :param c1: List of the from-column
        :param c2: List of the to-column
        :param v: The symbol to be removed from the seperator
        :param gs: The game state <l, I, r>
        :param visited: A list keeping track of all winning states d
        :return: Lazily return states d
        """
        next_marked = []  # store if the next step gs_, c_ has been explored already
        if c2 in visited:  # Return if c2 has been visited
            return

        if len(c1) == gs.l and symbol_not_in_seperator(gs.I, v):  # Return c2 if step game is won
            visited.append(c2)
            yield c2, gs.I

        for (q, trans_gen) in map(lambda origin: (origin, self.T.get_transitions(origin)), c1[:gs.l + 1]):
            for (qp_t,
                 p) in trans_gen:  # TODO: can this part be parallelized? -> use one thread per (qp_t, p) pair until q has no more successors
                x, y = self.alphabet_map.get_x(qp_t), self.alphabet_map.get_y(qp_t)
                if symbol_not_in_seperator(gs.I, y):
                    if p not in c2:
                        c2_ = c2 + [p]
                        if c2_ in visited:
                            continue
                    else:
                        c2_ = c2
                    gs_ = Triple(gs.l + (1, 0)[q in c1[:gs.l]],
                                 refine_seperator(gs.I, x),
                                 gs.r + (1, 0)[p in c2])
                    if not gs.equal(gs_) and (gs_.l, gs_.I, c2_) not in next_marked:
                        next_marked.append((gs_.l, gs_.I, c2_))
                        yield from self.step_game_gen_buffered_dfs(c1, c2_, v, gs_, visited)

    def step_game_gen_buffered_dfs(self, c1, c2, v, gs, visited):
        """
        Uses the same buffer as the one_shot implementation of dodo, returns states d in a depth first search
        :param c1: List of the from-column
        :param c2: List of the to-column
        :param v: The symbol to be removed from the seperator
        :param gs: The game state <l, I, r>
        :param visited: A list keeping track of all winning states d
        :return: Lazily return states d
        """
        #print(f'{c2} + {gs}')
        next_marked = []  # store if the next step gs_, c_ has been explored already
        if c2 in visited:  # Return if c2 has been visited
            return
        cache_hit = self.step_cache.get_entry(c1, gs, v, c2)  # Check if this partially played game is in cache
        if cache_hit is not None:
            for hit in cache_hit:
                yield hit
            return

        if len(c1) == gs.l and symbol_not_in_seperator(gs.I, v):  # Return c2 if step game is won
            visited.append(c2)
            #print("winning")
            yield c2

        for (q, trans_gen) in map(lambda origin: (origin, self.T.get_transitions(origin)), c1[:gs.l + 1]):
            for (qp_t, p) in trans_gen:
                x, y = self.alphabet_map.get_x(qp_t), self.alphabet_map.get_y(qp_t)
                if symbol_not_in_seperator(gs.I, y):
                    if p not in c2:
                        c2_ = c2 + [p]
                        if c2_ in visited:
                            continue
                    else:
                        c2_ = c2
                    gs_ = Triple(gs.l + (1, 0)[q in c1[:gs.l]],
                                 refine_seperator(gs.I, x),
                                 gs.r + (1, 0)[p in c2])
                    #if (gs_.l, gs_.I, c2_) in next_marked:
                        #print(f'marked: {q}, {self.alphabet_map.transition_to_str(qp_t)}, {p} + {str(gs_)}')
                    if not gs.equal(gs_) and (gs_.l, gs_.I, c2_) not in next_marked:
                        next_marked.append((gs_.l, gs_.I, c2_))
                        #print(f'step: {q}, {self.alphabet_map.transition_to_str(qp_t)}, {p}')
                        yield from self.step_game_gen_buffered_dfs(c1, c2_, v, gs_, visited)
        self.step_cache.add_entry(c1, gs, v, c2, visited)  # Add Game to cache

    def step_game_gen_buffered_dfs_2(self, c1, c2, v, gs, winning):
        """
        Uses the same buffer as the one_shot implementation of dodo, returns states d in a depth first search
        :param c1: List of the from-column
        :param c2: List of the to-column
        :param v: The symbol to be removed from the seperator
        :param gs: The game state <l, I, r>
        :param winning: A list keeping track of all winning states d
        :param visited:
        :return: Lazily return states d
        """
        #print(f'{c2} + {gs}')
        next_marked = []  # store if the next step gs_, c_ has been explored already
        if c2 in map(lambda d_gs: d_gs[0], winning):  # Return if c2 has been visited
            return
        cache_hit = self.step_cache.get_entry(c1, gs, v, c2)  # Check if this partially played game is in cache
        if cache_hit is not None:
            for hit in cache_hit:
                yield hit[0]
            return

        if len(c1) == gs.l and symbol_not_in_seperator(gs.I, v):  # Return c2 if step game is won
            winning.append((c2, gs))
            #print("winning")
            yield c2

        cache_hit = self.step_cache.get_entry(c1[:gs.l], gs, v, c2)  # Check if this partially played game is in cache
        d_new_list = []
        if cache_hit is not None:
            for hit in cache_hit:
                d_new = hit[0]
                gs_ = hit[1]
                if len(d_new) > len(c2) and (gs_.l, gs_.I, d_new) not in next_marked:
                    next_marked.append((gs_.l, gs_.I, d_new))
                    d_new_list.append(d_new)
                    yield from self.step_game_gen_buffered_dfs_2(c1, d_new, v, gs_, winning)
        for (q, trans_gen) in map(lambda origin: (origin, self.T.get_transitions(origin)), c1[:gs.l + 1]):
            for (qp_t, p) in trans_gen:
                if p in d_new_list:
                    continue
                x, y = self.alphabet_map.get_x(qp_t), self.alphabet_map.get_y(qp_t)
                if symbol_not_in_seperator(gs.I, y):
                    if p not in c2:
                        c2_ = c2 + [p]
                        if c2_ in winning:
                            continue
                    else:
                        c2_ = c2
                    gs_ = Triple(gs.l + (1, 0)[q in c1[:gs.l]],
                                 refine_seperator(gs.I, x),
                                 gs.r + (1, 0)[p in c2])
                    #if (gs_.l, gs_.I, c2_) in next_marked:
                        #print(f'marked: {q}, {self.alphabet_map.transition_to_str(qp_t)}, {p} + {str(gs_)}')
                    if not gs.equal(gs_) and (gs_.l, gs_.I, c2_) not in next_marked:
                        next_marked.append((gs_.l, gs_.I, c2_))
                        #print(f'step: {q}, {self.alphabet_map.transition_to_str(qp_t)}, {p}')
                        yield from self.step_game_gen_buffered_dfs_2(c1, c2_, v, gs_, winning)
        self.step_cache.add_entry(c1, gs, v, c2, winning)  # Add Game to cache


    def step_game_gen_buffered_bfs(self, c1, c2, v, gs, visited):
        """
        Uses the same buffer as the one_shot implementation of dodo, returns states d in a breath first search
        :param c1: List of the from-column
        :param c2: List of the to-column
        :param v: The symbol to be removed from the seperator
        :param gs: The game state <l, I, r>
        :param visited: A list keeping track of all winning states d
        :return: Lazily return states d
        """
        print(f'{c1} + {c2} + {v} + {str(gs)} + {visited}')
        next_marked = []  # store if the next step gs_, c_ has been explored already
        if c2 in visited:  # Return if c2 has been visited
            return
        cache_hit = self.step_cache.get_entry(c1, gs, v, c2)  # Check if this partially played game is in cache
        if cache_hit is not None:
            for hit in cache_hit:
                yield hit
            return

        if len(c1) == gs.l and symbol_not_in_seperator(gs.I, v):  # Return c2 if step game is won
            visited.append(c2)
            yield c2

        for (q, trans_gen) in map(lambda origin: (origin, self.T.get_transitions(origin)), c1[:gs.l + 1]):
            for (qp_t, p) in trans_gen:
                x, y = self.alphabet_map.get_x(qp_t), self.alphabet_map.get_y(qp_t)
                if symbol_not_in_seperator(gs.I, y):
                    if p not in c2:
                        c2_ = c2 + [p]
                        if c2_ in visited:
                            continue
                    else:
                        c2_ = c2
                    gs_ = Triple(gs.l + (1, 0)[q in c1[:gs.l]],
                                 refine_seperator(gs.I, x),
                                 gs.r + (1, 0)[p in c2])
                    if not gs.equal(gs_) and (gs_.l, gs_.I, c2_) not in next_marked:
                        next_marked.append((gs_.l, gs_.I, c2_))
                        yield from self.step_game_gen_buffered_dfs(c1, c2_, v, gs_, visited)
        self.step_cache.add_entry(c1, gs, v, c2, visited)  # Add Game to cache


class OneShotSimple:
    """Naive OneShot implementation"""

    def powerset_permutations(self, iterable, length):
        s = list(iterable)
        return list(chain.from_iterable(permutations(s, r) for r in length))

    def transition_iterator(self, T):
        """Returns a lazy iterator over all permutations c2, (u, S)"""
        state_range = range(0, T.get_state_count() + 1)
        column_len_range = range(1, T.get_state_count() + 1)
        u_range = range(0, T.get_alphabet_map().get_sigma_size())
        S_range = range(0, int(math.pow(2, T.get_alphabet_map().get_sigma_size())) - 1)
        return product(self.powerset_permutations(state_range, column_len_range), product(u_range, S_range))

    def initial_state_permutations(self, T):
        return self.powerset_permutations(T.get_initial_states(), range(1, len(T.get_initial_states()) + 1))

    class AbstractStepGameMemo(ABC):

        @abstractmethod
        def __init__(self):
            pass

        @abstractmethod
        def add_node(self, c1, c2, u, S, I, miss):
            pass

        @abstractmethod
        def check_step(self, c1, c2, u, S, initial_I):
            pass

        @abstractmethod
        def print_statistics(self):
            pass

    class StepGameMemoTree(AbstractStepGameMemo):
        """Memorizes Steps in the Step Game as a tree"""

        class Node:
            def __init__(self, I_tmp, miss):
                self.next_nodes = {}
                self.miss = miss
                self.I_tmp = I_tmp

            def add_next(self, b1, b2, node):
                self.next_nodes[(b1, b2)] = node

            def get_next(self, b1, b2):
                return self.next_nodes.get((b1, b2))

            def get_I_tmp(self):
                return self.I_tmp

            def set_I_tmp(self, I_tmp):
                self.I_tmp = I_tmp

            def set_miss(self, miss):
                self.miss = miss

            def is_miss(self):
                return self.miss

            def to_string(self, b1, b2, layer):
                string_acc = "b: " + str((b1, b2)) + " | I_tmp: " + bin(self.get_I_tmp()) + " | miss: " + str(
                    self.is_miss())
                for (b1_next, b2_next) in self.next_nodes:
                    string_acc += "\n" + ("*" * layer) + self.next_nodes[(b1_next, b2_next)].to_string(b1_next, b2_next,
                                                                                                       layer + 1)
                return string_acc

        def __init__(self, node_size):
            self.node_size = node_size
            self.root_nodes = {}
            self.total_checks = 0
            self.total_excluded = 0
            self.total_advanced = 0

        def add_node(self, c1, c2, u, S, I, miss):
            """
            :param c1: List of states
            :param c2: List of states
            :param u: Symbol
            :param S: Seperator bit map
            :param I: Game State Seperator bit map
            :param miss: Was the step rejected by step game
            :return:
            """
            if self.node_size == 0:  # Table is full
                return
            self.node_size -= 1

            current = self.root_nodes.get((c1[0], c2[0], u, S))
            if current is None:
                current = self.Node(False, -1)
                self.root_nodes[(c1[0], c2[0], u, S)] = current

            for I_index, (b1, b2) in enumerate(zip_longest(c1[1:], c2[1:])):
                next_node = current.get_next(b1, b2)
                if next_node is None:
                    next_node = self.Node(False, -1)
                    current.add_next(b1, b2, next_node)
                current = next_node
            current.set_I_tmp(I)
            current.set_miss(miss)

        def check_step(self, c1, c2, u, S, initial_I):
            self.total_checks = self.total_checks + 1
            """
            :param c1: List of states
            :param c2: List of states
            :param u: Symbol
            :param S: Seperator bit map
            :param initial_I: The I with which the Game State was initialized
            :return: The precomputed Game State or None if the Step Game was lost
            """
            game_state = Triple(0, 0, 0)
            current = self.root_nodes.get((c1[0], c2[0], u, S))

            for (b1, b2) in zip_longest(c1[1:], c2[1:]):
                if current is None:
                    break
                if current.is_miss():
                    self.total_excluded = self.total_excluded + 1
                    return None
                game_state.update(b1, b2, current.get_I_tmp())
                current = current.get_next(b1, b2)
            if game_state.l != 0 or game_state.r != 0:
                self.total_advanced += 1
            game_state.xor_I(initial_I)  # Remove all symbols that have been removed in pre-computation
            return game_state

        def print_statistics(self):
            print("Chache size left: " + str(self.node_size))
            print("-total checked: " + str(self.total_checks))
            print("-total excluded: " + str(self.total_excluded))
            print("-total advanced: " + str(self.total_advanced))

        def __str__(self):
            string_acc = ""
            for (b1, b2, u, S) in self.root_nodes.keys():
                current = self.root_nodes[(b1, b2, u, S)]
                string_acc += "u: " + str(u) + " | S: " + bin(S) + "\n" + current.to_string(b1, b2, 1) + "\n"
            return string_acc

    class StepGameMemoBDD(AbstractStepGameMemo):
        """Memorizes Steps in the Step Game as BDD"""

        def __init__(self, u_bits_num, state_bit_num):
            self.u_bits_num = u_bits_num
            self.S_bits_num = int(math.pow(2, self.u_bits_num))
            self.state_bit_num = state_bit_num
            self.column_len = int(math.pow(2, self.state_bit_num))
            self.encoding_bit_num = self.u_bits_num + self.S_bits_num + 2 * (self.state_bit_num + self.column_len)
            self.bdd_vars = list(map(lambda b: bddvar(f"b{b}", b), [*range(0, self.encoding_bit_num)]))
            self.total_checks = 0
            self.total_excluded = 0
            self.f = None

        def zero_padding(self, bin_num, length):
            """pads bin_num with 0s"""
            return ("0" * (length - len(bin_num))) + bin_num

        def encode(self, c1, c2, u, S):  # TODO: Needs to become alot faster
            """encodes c1, c2, u, S as a bitarray bin(u) + bin(S) + bin(c1) + bin(c2)"""
            u_bin = self.zero_padding(format(u, "b"), self.u_bits_num)
            S_bin = self.zero_padding(format(S, "b"), self.S_bits_num)
            c1_bin = ""
            c2_bin = ""
            if c1 is not None:
                c1_bin = "".join(list(map(lambda b: self.zero_padding(format(b, "b"), self.state_bit_num), c1)))
            if c2 is not None:
                c2_bin = "".join(list(map(lambda b: self.zero_padding(format(b, "b"), self.state_bit_num), c2)))

            return bitarray.bitarray(u_bin + S_bin + c1_bin + c2_bin)

        def add_node(self, c1, c2, u, S, I, miss):
            """uses the encoding to set bbd_vars to (u & S & c1 & c2) | f_tmp"""
            if not miss:
                return

            encoding = self.encode(c1, c2, u, S)

            tmp = bddvar("tmp") | ~bddvar("tmp")
            for i, bit in enumerate(encoding):
                if bit:
                    tmp = tmp & self.bdd_vars[i]
                else:
                    tmp = tmp & ~self.bdd_vars[i]
            if self.f is None:
                self.f = tmp
            else:
                self.f = self.f | tmp

        def check_step(self, c1, c2, u, S, initial_I):
            if self.f is None:
                return Triple(0, initial_I, 0)
            self.total_checks += 1
            encoding = self.encode(c1, c2, u, S)

            point = {self.bdd_vars[i]: bit for (i, bit) in enumerate(encoding)}

            check = self.f.restrict(point)
            if check:
                self.total_excluded += 1
                return None
            else:
                return Triple(0, initial_I, 0)

        def print_statistics(self):
            print("u bit size: " + str(self.u_bits_num))
            print("S bit size: " + str(self.S_bits_num))
            print("state bit size: " + str(self.state_bit_num))
            print("max column len: " + str(self.column_len))
            print("max encoding length: " + str(self.encoding_bit_num))
            print("---------------------")
            print("total checked: " + str(self.total_checks))
            print("total excluded: " + str(self.total_excluded))

    def one_shot_bfs(self, I, T, B):
        """
        Computes I ∘ (reduced seperator transducer) ∘ B
        :param I: An id transducer recognizing the initial states
        :param T: The transducer T
        :param B: An id transducer recognizing the bad states
        :return: Returns True if the property B could be verified
        """
        step_memo = self.StepGameMemoTree(10000000)
        alph_m = T.get_alphabet_map()
        sst = NFATransducer(alph_m)
        work_queue = self.initial_state_permutations(T)
        sst.add_initial_state_list(list(map(lambda x: hash_state(list(x), 1), self.initial_state_permutations(T))))
        visited_queue = self.initial_state_permutations(T)
        i = 0

        while len(work_queue) != 0:
            c1 = work_queue.pop(0)

            if len((I.join(sst)).join(B).get_final_states()) != 0:  # TODO: Optimize this check
                print("Result: x")
                return False

            c1_hash = hash_state(c1, 1)
            # Add final states to the sigma x sigma transducer
            if set(c1).issubset(set(T.get_final_states())):
                if len((I.join(sst)).join(B).get_final_states()) != 0:
                    step_memo.print_statistics()
                    print("# states: " + str(i))
                    print("Result: x")
                    return False

            for c2, (u, S) in self.transition_iterator(T):
                if self.step_game(c1, u, S, c2, T, False, step_memo):
                    print(str(c1) + " " + str(c2))
                    # Add c2 to the work queue
                    if c2 not in visited_queue:
                        i += 1
                        work_queue.append(c2)
                        visited_queue.append(c2)
                    # Hash the states
                    c2_hash = hash_state(c2, 1)
                    # Add transitions for c1, c2
                    for y in bit_map_seperator_to_inv_list(S, alph_m.get_sigma_size()):
                        sst.add_transition(c1_hash, alph_m.combine_x_and_y(y, u), c2_hash)
        step_memo.print_statistics()
        print("# states: " + str(i))
        print("Result: ✓")
        return True

    def step_game(self, c1, u, S, c2, T, logging, step_memo):
        """
        :param c1:          An array of the states in the from-column
        :param u:           The int encoding of u
        :param S:           A bit map encoding the seperator
        :param c2:          An array of the states in the to-column
        :param T:           The transducer T
        :param logging:     Print game state progress to the console
        :param step_memo:   A buffer to exclude columns c1, c2 that can't be winning based on previous step games
        :return:            Returns True if the transition (c1, [u,S], c2) is part of the reduced seperator transducer
        """
        alphabet_map = T.get_alphabet_map()
        game_state = step_memo.check_step(c1, c2, u, S, refine_seperator(alphabet_map.get_bit_map_sigma(), u))
        if game_state is None:
            return False

        game_state_tmp = Triple(-1, 0, -1)
        n = len(c1)
        m = len(c2)

        cur_c1 = slice_column(c1, game_state.l)
        cur_c2 = slice_column(c2, game_state.r)

        while not game_state_tmp.equal(game_state):
            game_state_tmp.copy(game_state)
            for q, x_y in product(c1, alphabet_map.sigma_x_sigma_iterator()):
                p = T.get_successor(q, x_y)
                if p is not None:
                    p = p[0]  # TODO: This is a temporary fix, access the first successor (there could be more)

                    # Verifies the 3 conditions to see if <q,[x,y],p> is part of winning strategy
                    q_in_c1 = q in cur_c1 or (False, q == c1[np.clip(0, n - 1, game_state.l)])[game_state.l < n]
                    p_in_c2 = p in cur_c2 or (False, p == c2[np.clip(0, m - 1, game_state.r)])[game_state.r < m]
                    y_not_in_I = symbol_not_in_seperator(game_state.I, alphabet_map.get_y(x_y))

                    if q_in_c1 and p_in_c2 and y_not_in_I:
                        # Update the game state and current columns
                        game_state.l += (1, 0)[q in cur_c1]
                        game_state.r += (1, 0)[p in cur_c2]
                        game_state.I = refine_seperator(game_state.I, alphabet_map.get_x(x_y))
                        cur_c1 = slice_column(c1, game_state.l)
                        cur_c2 = slice_column(c2, game_state.r)

                        # Game won
                        if game_state.l == n and game_state.r == m and game_state.I == S:
                            if logging:
                                self.log_step(n, m, S, c1, c2, cur_c1, cur_c2, q, x_y, p, game_state, alphabet_map)
                            return True

        if game_state.l < n and game_state.r < m:
            step_memo.add_node(c1, c2, u, S, -1, True)
        return False

    def log_step(self, n, m, S, c1, c2, cur_c1, cur_c2, q, x_y, p, game_state, alphabet_map):
        print("n: " + str(n) + "\nm: " + str(m) + "\nS: " + bin(S) + "\nc1 " + str(c1) + " | cur_c1: " + str(cur_c1))
        print("c2: " + str(c2) + " | cur_c2: " + str(cur_c2))
        print("cur_trans: <" + str(q) + ", " + str(alphabet_map.transition_to_str(x_y)) + ", " + str(p) + ">")
        print("game_state: " + str(game_state))
