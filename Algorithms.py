import math

import crypto

from Automata import Transducer
import Storage
import numpy as np
from Util import Triple
from itertools import chain, combinations, product
from Cryptodome.Hash import SHAKE256
from binascii import hexlify


def hash_state(state_str, bit_length):  # TODO: check if function does not produce collisions
    shake = SHAKE256.new()
    shake.update(bytes(state_str, 'ascii'))
    return int.from_bytes(hexlify(shake.read(bit_length)), 'big')


def state_from_column(column, bits):  # TODO: replace with Hashing
    state = 0
    for i, q in enumerate(column):
        state += q << (bits * i)
    return state


def powerset(iterable):
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(1, len(s) + 1))


def transition_iterator(T):
    """returns a lazy iterator over all permutations (c1, c2), (u, S)"""
    state_range = range(1, T.get_state_count() + 1)
    u_range = range(0, T.get_alphabet_map().get_sigma_size())
    S_range = range(0, int(math.pow(2, T.get_alphabet_map().get_sigma_size())))
    c1_powerset = powerset(state_range)
    c2_powerset = powerset(state_range)
    u_S_cross = product(u_range, S_range)
    return product(product(c1_powerset, c2_powerset), u_S_cross)  # TODO: better ordering of columns


def built_sigma_sigma_transducer(T):
    alph_m = T.get_alphabet_map()
    result = Transducer(10000, alph_m)

    for ((c1, c2), (u, S)) in transition_iterator(T):
        if c1 == [2] and c2 == [3] and u == 1 and S == 0:
            k = 0
        winning = step_game(c1, u, S, c2, T, False)
        if winning:
            origin = state_from_column(c1, 2)
            target = state_from_column(c2, 2)
            for y in bit_map_seperator_to_inv_list(S, alph_m.get_sigma_size()):
                print("trans: " + bin(y))
                result.add_transition(origin, alph_m.combine_x_and_y(y, u), target)
            print(str(list(c1)) + ", {" + str(u) + "|" + bin(S) + "}, " + str(list(c2)))
            print(bin(origin) + " ---> " + bin(target))
            print("------------------------------------------------------------------")
    result.dot_string("sigma")


# c1:            an array of the states in the from-column
# u:             the int encoding of u
# S:             a bit map encoding the seperator
# c2:            an array of the states in the to-column
# T:             the transducer T
def step_game(c1, u, S, c2,T, logging):
    """Returns a winning strategy if the transition (c1, [u,S], c2) is part of the sigma x sigma transducer"""
    alphabet_map = T.get_alphabet_map()
    game_state = Triple(0, refine_seperator(alphabet_map.get_bit_map_sigma(), u), 0)  # Initialize game_state <l,I,r>
    winning_strategy = []  # List of <q, x_y, p>
    n = len(c1)
    m = len(c2)
    bits = alphabet_map.get_sigma_encoding_num_bits()

    cur_c1 = slice_column(c1, game_state.l)
    cur_c2 = slice_column(c2, game_state.r)
    progress = 1  # keep track if all possible transitions are losing -> progress = 0 => step game is lost

    p_visited = []  # keep track which winning p have been visited from q TODO: Replace with bitmap

    while progress:
        q = (c1[n - 1], c1[np.clip(0, n - 1, game_state.l)])[game_state.l < n]
        progress = 0
        for x_y in range(
                alphabet_map.get_num_symbols_in_sigma_x_sigma()):  # TODO: Iterate over successors instead of symbols
            p = T.get_successor(q, x_y)
            if p != -1 and p not in p_visited:
                # Verifies the 3 conditions to see if <q,[x,y],p> is part of winning strategy
                q_in_c1 = (q in cur_c1, q == c1[np.clip(0, n - 1, game_state.l)])[game_state.l < n]
                p_in_c2 = (p in cur_c2, p == c2[np.clip(0, m - 1, game_state.r)])[game_state.r < m]
                y_not_in_I = symbol_not_in_seperator(game_state.I, alphabet_map.get_y(x_y))

                if q_in_c1 and p_in_c2 and y_not_in_I:
                    progress = 1
                    p_visited.append(p)
                    winning_strategy.append([q, x_y, p])

                    # Update the game state and current columns
                    game_state.l += (0, 1)[game_state.l < n]
                    game_state.r += (0, 1)[game_state.r < m]
                    game_state.I = refine_seperator(game_state.I, alphabet_map.get_x(x_y))
                    cur_c1 = slice_column(c1, game_state.l)
                    cur_c2 = slice_column(c2, game_state.r)

                    if logging:
                        log_step(n, m, S, c1, c2, cur_c1, cur_c2, q, x_y, p, game_state, alphabet_map)

                    # Game won
                    if game_state.l == n and game_state.r == m and game_state.I == S:
                        return winning_strategy
    return []

    # if(T.get_successor())


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
    print("-----------------------")
    print("n: " + str(n) + "\nm: " + str(m) + "\nS: " + bin(S) + "\nc1 " + str(c1) + " | cur_c1: " + str(cur_c1))
    print("c2: " + str(c2) + " | cur_c2: " + str(cur_c2))
    print("cur_trans: <" + str(q) + ", " + str(alphabet_map.transition_to_str(x_y)) + ", " + str(p) + ">")
    print("game_state: " + str(game_state))
