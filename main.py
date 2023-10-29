import numpy as np
from scipy.sparse import random, dok_array

import Algorithms
import Automata
import Storage
from itertools import *


def build_token_parsing_input_transducer():
    alph_map = Storage.AlphabetMap(['n', 't'])
    transducer = Automata.NFATransducer(alph_map)
    transducer.set_initial_state(0)
    transducer.add_final_state(1)

    transducer.add_transition(0, alph_map.combine_symbols('n', 'n'), 0)
    transducer.add_transition(0, alph_map.combine_symbols('t', 't'), 1)
    transducer.add_transition(1, alph_map.combine_symbols('n', 'n'), 1)
    return transducer

def build_bad_word_token_parsing_transducer():
    alph_map = Storage.AlphabetMap(['n', 't'])
    transducer = Automata.NFATransducer(alph_map)
    transducer.set_initial_state(0)
    transducer.add_final_state(2)

    transducer.add_transition(0, alph_map.combine_symbols('n', 'n'), 0)
    transducer.add_transition(0, alph_map.combine_symbols('t', 't'), 1)
    transducer.add_transition(1, alph_map.combine_symbols('n', 'n'), 1)
    transducer.add_transition(1, alph_map.combine_symbols('t', 't'), 2)
    transducer.add_transition(2, alph_map.combine_symbols('n', 'n'), 2)
    return transducer


def build_simple_token_passing_transducer(as_NFA):
    alph_map = Storage.AlphabetMap(['n', 't'])
    transducer = None
    if as_NFA:
        transducer = Automata.NFATransducer(alph_map)
    else:
        transducer = Automata.Transducer(3, alph_map)

    transducer.initial_state = 0
    transducer.add_final_state(2)

    transducer.add_transition(0, alph_map.combine_symbols('n', 'n'), 0)
    transducer.add_transition(0, alph_map.combine_symbols('t', 'n'), 1)

    transducer.add_transition(1, alph_map.combine_symbols('n', 't'), 2)

    transducer.add_transition(2, alph_map.combine_symbols('n', 'n'), 2)
    return transducer


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


def collatz_transducer(as_NFA):
    alph_map = Storage.AlphabetMap(['0', '1'])
    transducer = None
    if as_NFA:
        transducer = Automata.NFATransducer(alph_map)
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
    T = build_simple_token_passing_transducer(False)
    I = build_token_parsing_input_transducer()

    x = Algorithms.verify(I, T, 0)
    x.to_dot("joined", None)
    print(x.get_final_states())

    """
    I = build_token_parsing_input_transducer()
    T = build_simple_token_passing_transducer(False)
    B = build_bad_word_token_parsing_transducer()
    T.to_dot("simple", None)
    T_prime = Algorithms.verify(I, T, B)
    T_prime.to_dot("final", None)
    print(T_prime.get_final_states())
    
    result = Algorithms.verify(I, T, B)
    result.to_dot("joined", None)
    print(result.get_final_states())
    """


