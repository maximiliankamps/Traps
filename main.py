import numpy as np
from scipy.sparse import random, dok_array

import Algorithms
import Automata
import Storage


def build_simple_token_passing_transducer():
    alph_map = Storage.AlphabetMap(['n', 't'])
    transducer = Automata.Transducer(3, alph_map)

    transducer.add_transition(1, alph_map.combine_symbols('n', 'n'), 1)
    transducer.add_transition(1, alph_map.combine_symbols('t', 'n'), 2)

    transducer.add_transition(2, alph_map.combine_symbols('n', 't'), 3)

    transducer.add_transition(3, alph_map.combine_symbols('n', 'n'), 3)
    return transducer

# TODO: change storage so entries are shifted by one, revert labeling everywhere else to 0....n
# TODO: modify dot_string function so it displays states as columns
# TODO: build simple_token_transducer by hand and compare with code

def build_circular_token_passing_transducer():
    alph_map = Storage.AlphabetMap(['n', 't'])
    trans = Automata.Transducer(6, alph_map)
    trans.add_transition(1, alph_map.combine_symbols('n', 'n'), 2)
    trans.add_transition(1, alph_map.combine_symbols('t', 'n'), 3)
    trans.add_transition(1, alph_map.combine_symbols('n', 't'), 5)

    trans.add_transition(2, alph_map.combine_symbols('n', 'n'), 2)
    trans.add_transition(2, alph_map.combine_symbols('t', 'n'), 3)

    trans.add_transition(3, alph_map.combine_symbols('n', 't'), 4)

    trans.add_transition(4, alph_map.combine_symbols('n', 'n'), 4)

    trans.add_transition(5, alph_map.combine_symbols('n', 'n'), 5)
    trans.add_transition(5, alph_map.combine_symbols('t', 'n'), 6)
    return trans


if __name__ == '__main__':
    Algorithms.built_sigma_sigma_transducer(build_simple_token_passing_transducer())
