import numpy as np
from scipy.sparse import random, dok_array

import Algorithms
import Automata
import BitUtil
import Storage


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
    print(bin(BitUtil.shrink_bit_map(0b10101100, 1)))

    x = build_circular_token_passing_transducer()
    x.dot_string("trans")
