import Storage
import numpy as np
from Util import Triple
from BitUtil import *

def built_sigma_sigma_transducer(T, alphabet_map):
    return 0


# c1:            an array of the states in the from-column
# u:             an integer encoding u from sigma
# S:             a bit map encoding the seperator
# c2:            an array of the states in the to-column
# bit_map_sigma: a bit map encoding sigma
# T:             the transducer t

def step_game(c1, u, S, c2, alphabet_map, T, logging):
    game_state = Triple(0, refine_seperator(alphabet_map.get_bit_map_sigma(), u), 0)  # <l,I,r>
    winning_strategy = []  # list of <q, x_y, p>
    n = len(c1)
    m = len(c2)
    bits = alphabet_map.get_sigma_bit_len()

    cur_c1 = slice_column(c1, game_state.l)
    cur_c2 = slice_column(c2, game_state.r)
    progress = 1

    p_visited = [] #TODO: Replace with bitmap

    while progress:
        q = (c1[n-1], c1[np.clip(0, n-1, game_state.l)])[game_state.l < n]
        progress = 0
        for x_y in range(alphabet_map.get_num_csym()):
            p = T.get_successor(q, x_y)
            if p != -1 and p not in p_visited:
                q_in_c1 = (q in cur_c1, q == c1[np.clip(0, n-1, game_state.l)])[game_state.l < n]
                p_in_c2 = (p in cur_c2, p == c2[np.clip(0, m-1, game_state.r)])[game_state.r < m]
                y_not_in_I = symbol_not_in_seperator(game_state.I, get_output(x_y, bits))

                if q_in_c1 and p_in_c2 and y_not_in_I:
                    progress = 1
                    p_visited.append(p)
                    winning_strategy.append([q, x_y, p])

                    game_state.l += (0, 1)[game_state.l < n]
                    game_state.r += (0, 1)[game_state.r < m]
                    game_state.I = refine_seperator(game_state.I, get_input(x_y, bits))
                    cur_c1 = slice_column(c1, game_state.l)
                    cur_c2 = slice_column(c2, game_state.r)

                    if logging:
                        log_step(n, m, S, c1, c2, cur_c1, cur_c2, q, x_y, p, game_state, alphabet_map)

                    if game_state.l == n and game_state.r == m and game_state.I == S:
                        return winning_strategy
    return []

        # if(T.get_successor())


# I_satisfied = (bit_pos_mask(get_output(x_y, bits)) & game_state.get_I(), 1)[game_state.I == 0]

def slice_column(col, i):
    return col[:i]


def refine_seperator(S, i):  # Remove the 1-bit at position i from S
    return S & ~(1 << i)


def symbol_not_in_seperator(S, i):
    return (S & (1 << i)) == 0


def log_step(n, m, S, c1, c2, cur_c1, cur_c2, q, x_y, p, game_state, alphabet_map):
    print("-----------------------")
    print("n: " + str(n) + "\nm: " + str(m) + "\nS: " + bin(S) + "\nc1 " + str(c1) + " | cur_c1: " + str(cur_c1))
    print("c2: " + str(c2) + " | cur_c2: " + str(cur_c2))
    print("cur_trans: <" + str(q) + ", " + str(alphabet_map.trans_str(x_y)) + ", " + str(p) + ">")
    print("game_state: " + str(game_state))

