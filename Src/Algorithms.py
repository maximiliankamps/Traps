import math

from Automata import NFATransducer
import Storage
import numpy as np
from Util import Triple, strS
from itertools import chain, product, permutations


def hash_state(column_list, byte_length):
    """Combines all states in column_list in to a string and returns a hash of byte_length"""
    state_str = ""
    for state in column_list:
        state_str += str(state + 1)  # Important!!! + 1 to generate unique hash for columns with q0 states
    return int(state_str)
    # return int.from_bytes(hexlify(shake.read(byte_length)), 'big')


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


def one_shot(I, T, B):
    ctr = 0
    alph_m = T.get_alphabet_map()
    column_hashing = Storage.ColumnHashing(True)
    sst = NFATransducer(alph_m)
    work_queue = initial_state_permutations(T)
    sst.add_initial_state_list(list(map(lambda x: hash_state(list(x), 1), initial_state_permutations(T))))
    visited_queue = initial_state_permutations(T)

    while len(work_queue) != 0:
        ctr += 1
        c1 = work_queue.pop(0)

        # if len((I.join(sst)).join(B).get_final_states()) != 0:  # TODO: Optimize this check
        #    return "Reachable"

        c1_hash = hash_state(c1, 1)
        # Add final states to the sigma x sigma transducer
        if set(c1).issubset(set(T.get_final_states())):
            sst.add_final_state(c1_hash)
            if len((I.join(sst)).join(B).get_final_states()) != 0:
                return "Reachable"

        for c2, (u, S) in transition_iterator(T):
            if step_game(c1, u, S, c2, T, False):
                # Add c2 to the work queue
                if c2 not in visited_queue:
                    work_queue.append(c2)
                    visited_queue.append(c2)
                # Hash the states
                c2_hash = hash_state(c2, 1)
                column_hashing.store_column(c1_hash, c1), column_hashing.store_column(c2_hash, c2)
                # Add transitions for c1, c2
                for y in bit_map_seperator_to_inv_list(S, alph_m.get_sigma_size()):
                    sst.add_transition(c1_hash, alph_m.combine_x_and_y(y, u), c2_hash)
    return "not Reachable"


# c1:            an array of the states in the from-column
# u:             the int encoding of u
# S:             a bit map encoding the seperator
# c2:            an array of the states in the to-column
# T:             the transducer T
def step_game(c1, u, S, c2, T, logging):
    """Returns a winning strategy if the transition (c1, [u,S], c2) is part of the sigma x sigma transducer"""
    alphabet_map = T.get_alphabet_map()
    game_state = Triple(0, refine_seperator(alphabet_map.get_bit_map_sigma(), u), 0)  # Initialize game_state <l,I,r>
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
                p = p[0]  # TODO: This is a temporary fix

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

                    if logging:
                        log_step(n, m, S, c1, c2, cur_c1, cur_c2, q, x_y, p, game_state, alphabet_map)

                    # Game won
                    if game_state.l == n and game_state.r == m and game_state.I == S:
                        return True
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
