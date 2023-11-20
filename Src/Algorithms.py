import math

from Automata import NFATransducer, hash_state
from enum import Enum
import Storage
import numpy as np
from Util import Triple, strS
from itertools import chain, product, permutations
from Optimizations import StepGameMemo, StepGameMemo2


def powerset_permutations(iterable, length):
    s = list(iterable)
    return list(chain.from_iterable(permutations(s, r) for r in length))


def transition_iterator(T):
    """Returns a lazy iterator over all permutations c2, (u, S)"""
    state_range = range(0, T.get_state_count() + 1)
    column_len_range = range(1, T.get_state_count() + 1)
    u_range = range(0, T.get_alphabet_map().get_sigma_size())
    S_range = range(0, int(math.pow(2, T.get_alphabet_map().get_sigma_size())) - 1)
    return product(powerset_permutations(state_range, column_len_range), product(u_range, S_range))


def initial_state_permutations(T):
    return powerset_permutations(T.get_initial_states(), range(1, len(T.get_initial_states()) + 1))


def verify(I, T, B):
    return 0


class StepGameBuffer:
    def __init__(self):
        self.c_ltb = {}
        self.cache_hit = 0

    def add_entry(self, c, symbol, d_I_list):
        if self.c_ltb.get((tuple(c), symbol)) is None:
            self.c_ltb[(tuple(c), symbol)] = d_I_list

    def get_entry(self, c, symbol):
        i = self.c_ltb.get((tuple(c), symbol))
        if i is not None:
            self.cache_hit += len(self.c_ltb.get((tuple(c), symbol)))
        return i
    """
    def add_entry(self, c, symbols, d):
        c = tuple(c)
        transition_ltb = (self.c_ltb.get(c), {})[self.c_ltb.get(c) is None]
        self.c_ltb[c] = transition_ltb

        for s in symbols:
            if transition_ltb.get(s) is None:
                transition_ltb[s] = [d]
            else:
                transition_ltb[s].append(d)

    def get_entry(self, c, symbol):
        transition_ltb = self.c_ltb.get(tuple(c))
        if transition_ltb is None:
            return None
        if transition_ltb.get(symbol) is not None:
            print("hit")
        return transition_ltb.get(symbol)
    """


if __name__ == '__main__':
    step = StepGameBuffer()

    step.add_entry([1,2], [('n', 't'), ('n', 'n')], [1, 2])
    print(step.get_entry([1,2], ('n', 't')))
    print(step.get_entry([1,2], ('n', 'n')))
    print(step.get_entry([1,2], ('t', 'n')))
    print(step.get_entry([1,2,3], ('t', 'n')))


class ONESHOT:
    def __init__(self, IxB, T):
        self.IxB = IxB
        self.T = T
        self.alphabet_map = T.get_alphabet_map()
        self.sst = NFATransducer(self.alphabet_map)
        self.step_buffer = StepGameBuffer()

    def one_shot_bfs(self):
        (ib0, c0) = (self.IxB.get_initial_states()[0], [self.T.get_initial_states()[0]])
        Q = [(ib0, c0)]
        W = [(ib0, c0)]
        C = []
        i = 0
        ctr = 0
        while len(Q) != 0:
            (ib, c) = Q.pop(0)
            # iterate over all transitions of the state ib
            for (ib_trans, ib_succ) in self.IxB.get_transitions(ib):
                u, v = self.alphabet_map.get_y(ib_trans), self.alphabet_map.get_x(ib_trans)
                gs = Triple(0, refine_seperator(self.alphabet_map.get_bit_map_sigma(), u), 0)
                # iterate over all reachable (ib, c) -> (ib_succ, d)

                if (tuple(c), ib_trans) not in C:
                    C.append((tuple(c), ib_trans))
                ctr += 1

                d_I_list = []
                hits = self.step_buffer.get_entry(c, ib_trans)
                d_I_itr = (self.step_game_gen(c, [], v, gs, [], [], ib_trans), hits)[hits is not None]

                for (d, I) in d_I_itr:
                    d_I_list.append((d, I))
                    if (ib_succ, d) not in W:
                        W.append((ib_succ, d))
                        Q.append((ib_succ, d))
                        i += 1
                        print(i)

                        if self.IxB.is_final_state(ib_succ) and len(
                                list((filter(lambda q: (not self.T.is_final_state(q)), d)))) == 0:
                            print(f'{ib_succ}, {d}')
                            print("Result: x")
                            return 0
                self.step_buffer.add_entry(c, ib_trans, d_I_list)
        print("Result: ✓")
        print("Total unique pairs: " + str(len(C)))
        print("Total c-trans pairs checked: " + str(ctr))
        print("Total cache hits: " + str(self.step_buffer.cache_hit))
        return 0

    def step_game_gen(self, c1, c2, v, gs, visited, marked, ib_trans):
        if len(c1) == gs.l and symbol_not_in_seperator(gs.I, v):
            visited.append(c2)
            yield c2, gs.get_I()

        d_I_hits = self.step_buffer.get_entry(c1[:gs.l], ib_trans)
        if d_I_hits is not None and c1[:gs.l] not in marked:
            for (d, I) in d_I_hits:
                if d not in visited:
                    marked.append(c1[:gs.l])
                    yield from self.step_game_gen(c1, d, v, Triple(gs.l, I, len(d)), visited, marked, ib_trans)
        else:
            for (q, trans_gen) in (map(lambda origin: (origin, self.T.get_transitions(origin)), c1[:gs.l + 1])):
                for (qp_t, p) in trans_gen:
                    if c2 in visited:
                        break
                    x, y = self.alphabet_map.get_x(qp_t), self.alphabet_map.get_y(qp_t)
                    if symbol_not_in_seperator(gs.I, y):
                        c2_ = []
                        if p not in c2:
                            c2_ = c2 + [p]
                            if c2_ in visited:
                                break
                        else:
                            c2_ = c2
                        gs_ = Triple(gs.l + (1, 0)[q in c1[:gs.l]],
                                     refine_seperator(gs.I, x),
                                     gs.r + (1, 0)[p in c2])
                        if not gs.equal(gs_):
                            yield from self.step_game_gen(c1, c2_, v, gs_, visited, marked, ib_trans)


def one_shot(I, T, B):
    step_memo = StepGameMemo(10000000)
    alph_m = T.get_alphabet_map()
    sst = NFATransducer(alph_m)
    work_queue = initial_state_permutations(T)
    sst.add_initial_state_list(list(map(lambda x: hash_state(list(x), 1), initial_state_permutations(T))))
    visited_queue = initial_state_permutations(T)

    while len(work_queue) != 0:
        c1 = work_queue.pop(0)

        if len((I.join(sst)).join(B).get_final_states()) != 0:  # TODO: Optimize this check
            return "Reachable"

        c1_hash = hash_state(c1, 1)
        # Add final states to the sigma x sigma transducer
        if set(c1).issubset(set(T.get_final_states())):
            if len((I.join(sst)).join(B).get_final_states()) != 0:
                step_memo.print_statistics()
                return "Reachable"

        for c2, (u, S) in transition_iterator(T):
            if step_game(c1, u, S, c2, T, False, step_memo):
                print(str(c1) + " " + str(c2))
                # Add c2 to the work queue
                if c2 not in visited_queue:
                    work_queue.append(c2)
                    visited_queue.append(c2)
                # Hash the states
                c2_hash = hash_state(c2, 1)
                # Add transitions for c1, c2
                for y in bit_map_seperator_to_inv_list(S, alph_m.get_sigma_size()):
                    sst.add_transition(c1_hash, alph_m.combine_x_and_y(y, u), c2_hash)
    step_memo.print_statistics()
    sst.to_dot("test", None)
    return "not Reachable"


# c1:            an array of the states in the from-column
# u:             the int encoding of u
# S:             a bit map encoding the seperator
# c2:            an array of the states in the to-column
# T:             the transducer T
def step_game(c1, u, S, c2, T, logging, step_memo):
    """Returns a winning strategy if the transition (c1, [u,S], c2) is part of the sigma x sigma transducer"""
    alphabet_map = T.get_alphabet_map()
    game_state = Triple(0, refine_seperator(alphabet_map.get_bit_map_sigma(), u), 0)  # Initialize game_state <l,I,r>

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
                p = p[0]  # TODO: This is a temporary fix, access the firs successor (there could be more)

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
                        # step_memo.add_node(c1, c2, u, S, refine_seperator(alphabet_map.get_bit_map_sigma(), u), False)
                        if logging:
                            log_step(n, m, S, c1, c2, cur_c1, cur_c2, q, x_y, p, game_state, alphabet_map)
                        return True

    if game_state.l < n and game_state.r < m:
        step_memo.add_node(c1, c2, u, S, -1, True)
    return False


def slice_column(col, i):
    """Slices the columns c1 and c2 by i"""
    return col[:i]


def refine_seperator(S, i):
    """Remove the 1-bit at position i from S"""
    return S & ~(1 << i)


def symbol_not_in_seperator(S, i):
    """Returns true if the symbol with index i is not in S"""
    return (S & (1 << i)) == 0


def bit_map_seperator_to_inv_list(S, n):  # TODO: can be optimized?
    """Returns the indices of all symbols that are not contained in the bit-map of the seperator S"""
    inv_list = []
    for i in range(0, n):
        if S & (1 << i) == 0:
            inv_list.append(i)
    return inv_list


def log_step(n, m, S, c1, c2, cur_c1, cur_c2, q, x_y, p, game_state, alphabet_map):
    print("n: " + str(n) + "\nm: " + str(m) + "\nS: " + strS(S) + "\nc1 " + str(c1) + " | cur_c1: " + str(cur_c1))
    print("c2: " + str(c2) + " | cur_c2: " + str(cur_c2))
    print("cur_trans: <" + str(q) + ", " + str(alphabet_map.transition_to_str(x_y)) + ", " + str(p) + ">")
    print("game_state: " + str(game_state))
