import numpy as np
from scipy.sparse import random, dok_array

import Algorithms
import Automata
import BitUtil
import Storage


def build_simple_token_passing_transducer():
    a_m = Storage.AlphabetMap(['n', 't'])
    transducer = Automata.Transducer(3, a_m)

    transducer.add_transition(1, a_m.combine_symbols('n', 'n'), 1)
    transducer.add_transition(1, a_m.combine_symbols('t', 'n'), 2)

    transducer.add_transition(2, a_m.combine_symbols('n', 't'), 3)

    transducer.add_transition(3, a_m.combine_symbols('n', 'n'), 3)
    return transducer


def build_circular_token_passing_transducer():
    a_m = Storage.AlphabetMap(['n', 't'])
    trans = Automata.Transducer(6, a_m)
    trans.add_transition(0, a_m.combine_symbols('n', 'n'), 1)
    trans.add_transition(0, a_m.combine_symbols('t', 'n'), 2)
    trans.add_transition(0, a_m.combine_symbols('n', 't'), 4)

    trans.add_transition(1, a_m.combine_symbols('n', 'n'), 1)
    trans.add_transition(1, a_m.combine_symbols('t', 'n'), 2)

    trans.add_transition(2, a_m.combine_symbols('n', 't'), 3)

    trans.add_transition(3, a_m.combine_symbols('n', 'n'), 3)

    trans.add_transition(4, a_m.combine_symbols('n', 'n'), 4)
    trans.add_transition(4, a_m.combine_symbols('t', 'n'), 5)
    return trans


if __name__ == '__main__':
    a_m = Storage.AlphabetMap(['n', 't'])
    c1 = [0]   # q0
    c2 = [1]  # q1
    u = 0b0    # n
    S = 0b00    # {}

    Algorithms.built_sigma_sigma_transducer(build_simple_token_passing_transducer())



    #print(Algorithms.step_game(c1, u, S, c2, a_m, build_circular_token_passing_transducer(), 0))
