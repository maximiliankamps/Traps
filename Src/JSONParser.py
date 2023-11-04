import json
from Storage import AlphabetMap


def rts_from_json(filename):
    file = open(f'benchmark/{filename}')
    rts_dict = json.load(file)
    initial_dict = rts_dict["initial"]
    transducer_dict = rts_dict["transducer"]
    properties_dict = rts_dict["properties"]

    alphabet_map = AlphabetMap(rts_dict["alphabet"])
    print(alphabet_map)


"""
def built_id_transducer(t_dict, alph_map):
    id_transducer = NFATransducer(alph_map)
    NFATransducer.add_initial_state(t_dict["initialState"])
"""


def parse_transition_regex():
    return 0


if __name__ == '__main__':
    rts_from_json("token-passing.json")
