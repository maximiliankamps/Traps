import os
import time

import Algorithms
import Automata
import Storage


def collatz_transducer():
    alph_map = Storage.AlphabetMap(['0', '1'])
    transducer = Automata.NFATransducer(alph_map)

    transducer.initial_states.append(1)
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


def build_token_parsing_input_transducer():
    alph_map = Storage.AlphabetMap(['n', 't'])
    transducer = Automata.NFATransducer(alph_map)
    transducer.set_state_count(2)
    transducer.add_initial_state(0)
    transducer.add_final_state(1)

    transducer.add_transition(0, alph_map.combine_symbols('n', 'n'), 0)
    transducer.add_transition(0, alph_map.combine_symbols('t', 't'), 1)
    transducer.add_transition(1, alph_map.combine_symbols('n', 'n'), 1)
    return transducer


def build_bad_word_token_parsing_transducer():
    alph_map = Storage.AlphabetMap(['n', 't'])
    transducer = Automata.NFATransducer(alph_map)
    transducer.add_initial_state(0)
    transducer.add_final_state(2)
    transducer.set_state_count(3)

    transducer.add_transition(0, alph_map.combine_symbols('n', 'n'), 0)
    transducer.add_transition(0, alph_map.combine_symbols('t', 't'), 1)
    transducer.add_transition(1, alph_map.combine_symbols('n', 'n'), 1)
    transducer.add_transition(1, alph_map.combine_symbols('t', 't'), 2)
    transducer.add_transition(2, alph_map.combine_symbols('n', 'n'), 2)
    return transducer


def build_simple_token_passing_transducer():
    alph_map = Storage.AlphabetMap(['n', 't'])
    transducer = Automata.NFATransducer(alph_map)

    transducer.add_initial_state(0)
    transducer.add_final_state(2)
    transducer.set_state_count(3)

    transducer.add_transition(0, alph_map.combine_symbols('n', 'n'), 0)
    transducer.add_transition(0, alph_map.combine_symbols('t', 'n'), 1)

    transducer.add_transition(1, alph_map.combine_symbols('n', 't'), 2)

    transducer.add_transition(2, alph_map.combine_symbols('n', 'n'), 2)
    return transducer


"""
if __name__ == '__main__':
    # collatz_transducer(True).join(collatz_transducer(True))

    I = build_token_parsing_input_transducer()
    T = build_simple_token_passing_transducer()
    B = build_bad_word_token_parsing_transducer()

    start_time = time.time()
    print(Algorithms.one_shot(I, T, B))
    end_time = time.time()

    # Calculate elapsed time
    elapsed_time = end_time - start_time
    print("Elapsed time: ", elapsed_time * 1000, "ms")

"""
if __name__ == '__main__':
    rts = Automata.RTS("token-passing.json")

    start_time = time.time()
    print(Algorithms.one_shot(rts.get_I(), rts.get_T(), rts.get_B("manytoken")))
    end_time = time.time()

    elapsed_time = end_time - start_time
    print("Elapsed time: ", elapsed_time * 1000, "ms")