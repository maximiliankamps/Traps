import numpy as np
from scipy.sparse import random, dok_array

import Algorithms
import Automata
import Storage


def build_simple_token_passing_transducer(as_NFA):
    alph_map = Storage.AlphabetMap(['n', 't'])
    transducer = None
    if as_NFA:
        transducer = Automata.NFATransducer(3, alph_map)
    else:
        transducer = Automata.Transducer(3, alph_map)

    transducer.initial_state = 0
    transducer.add_final_state(2)

    transducer.add_transition(0, alph_map.combine_symbols('n', 'n'), 0)
    transducer.add_transition(0, alph_map.combine_symbols('t', 'n'), 1)

    transducer.add_transition(1, alph_map.combine_symbols('n', 't'), 2)

    transducer.add_transition(2, alph_map.combine_symbols('n', 'n'), 2)
    return transducer



# TODO: change storage so entries are shifted by one, revert labeling everywhere else to 0....n
# TODO: modify dot_string function so it displays states as columns
# TODO: build simple_token_transducer by hand and compare with code

def build_circular_token_passing_transducer():
    alph_map = Storage.AlphabetMap(['n', 't'])
    trans = Automata.Transducer(6, alph_map)
    trans.add_transition(0, alph_map.combine_symbols('n', 'n'), 1)
    trans.add_transition(0, alph_map.combine_symbols('t', 'n'), 2)
    trans.add_transition(0, alph_map.combine_symbols('n', 't'), 4)

    trans.add_transition(1, alph_map.combine_symbols('n', 'n'), 1)
    trans.add_transition(1, alph_map.combine_symbols('t', 'n'), 2)

    trans.add_transition(2, alph_map.combine_symbols('n', 't'), 3)

    trans.add_transition(3, alph_map.combine_symbols('n', 'n'), 3)

    trans.add_transition(4, alph_map.combine_symbols('n', 'n'), 4)
    trans.add_transition(4, alph_map.combine_symbols('t', 'n'), 5)
    return trans

def collatz_transucer(as_NFA):
    alph_map = Storage.AlphabetMap(['0', '1'])
    transducer = None
    if as_NFA:
        transducer = Automata.NFATransducer(7, alph_map)
    else:
        transducer = Automata.Transducer(7, alph_map)
    transducer.initial_state = 1
    transducer.final_states = [1, 2, 6]


    zz = alph_map.combine_symbols('0', '0')
    zo = alph_map.combine_symbols('0', '1')
    oz = alph_map.combine_symbols('1', '0')
    oo = alph_map.combine_symbols('1', '1')

    transducer.add_transition(1, zz, 2)
    transducer.add_transition(1, oz, 4)
    transducer.add_transition(1, zo, 3)

    transducer.add_transition(2, zz, 2)
    transducer.add_transition(2, zo, 3)

    transducer.add_transition(3, oo, 3)
    transducer.add_transition(3, oz, 2)

    transducer.add_transition(4, oo, 4)
    transducer.add_transition(4, zz, 5)

    transducer.add_transition(5, oz, 4)
    transducer.add_transition(5, zo, 6)

    transducer.add_transition(6, zz, 6)
    transducer.add_transition(6, oo, 5)
    return transducer



if __name__ == '__main__':
    NFA = collatz_transucer(True)
    DFA = collatz_transucer(False)

    #DFA.dot_string("t", None)


    (NFA.left_join(DFA))
    """
    alph_map = Storage.AlphabetMap(['n', 't'])
    DFA = Automata.Transducer(1, alph_map)
    DFA.add_transition(0, alph_map.combine_symbols('n', 'n'), 0)
    DFA.initial_state = 0

    NFA = Automata.NFATransducer(2, alph_map)
    NFA.add_transition(0, alph_map.combine_symbols('n', 'n'), 0)
    NFA.add_transition(0, alph_map.combine_symbols('n', 't'), 1)
    NFA.add_transition(1, alph_map.combine_symbols('n', 't'), 1)
    NFA.initial_state = 0
    """


    #Algorithms.built_sigma_sigma_transducer(build_simple_token_passing_transducer(), False).dot_string("sigma", None)

