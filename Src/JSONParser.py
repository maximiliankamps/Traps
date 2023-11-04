import json
import re

from Storage import AlphabetMap
from Automata import NFATransducer


def rts_from_json(filename):
    file = open(f'benchmark/{filename}')
    rts_dict = json.load(file)
    alphabet_map = AlphabetMap(rts_dict["alphabet"])

    initial_dict = rts_dict["initial"]
    transducer_dict = rts_dict["transducer"]
    properties_dict = rts_dict["properties"]

    I = built_id_transducer(initial_dict, alphabet_map)
    T = build_transducer(transducer_dict, alphabet_map)
    B_dict = {name: built_id_transducer(properties_dict[name], alphabet_map) for name in properties_dict}



def built_id_transducer(nfa_dict, alph_map):
    id_transducer = NFATransducer(alph_map)
    id_transducer.add_initial_state(nfa_dict["initialState"])
    id_transducer.add_final_state_list(nfa_dict["acceptingStates"])
    for t in nfa_dict["transitions"]:
        letter = t["letter"]
        symbol = alph_map.combine_symbols(letter, letter)
        id_transducer.add_transition(int(t["origin"][1:]), symbol, int(t["target"][1:]))
    return id_transducer


def build_transducer(trans_dict, alph_map):
    transducer = NFATransducer(alph_map)
    transducer.add_initial_state(trans_dict["initialState"])
    transducer.add_final_state_list(trans_dict["acceptingStates"])
    for t in trans_dict["transitions"]:
        (x, y) = tuple(parse_transition_regex(t["letter"]))
        symbol = alph_map.combine_symbols(x, y)
        transducer.add_transition(int(t["origin"][1:]), symbol, int(t["target"][1:]))
    return transducer


def parse_transition_regex(regex):
    return regex.split(",")
