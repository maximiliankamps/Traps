import Storage
from Util import Triple
from BitUtil import bit_pos_mask, shrink_bit_map, get_input


# c1:            an array of the states in the from-column
# u:             an integer encoding u from sigma
# S:             a bit map encoding the seperator
# c2:            an array of the states in the to-column
# bit_map_sigma: a bit map encoding sigma
# T:             the transducer t

def step_game(c1, u, S, c2, alphabet_map, T):
    game_state = Triple(0, alphabet_map.get_bit_map_sigma() ^ (1 << u), 0)  # <l,I,r>

    while 1:
        cur_c1 = shrink_bit_map(c1, game_state.l)
        cur_c2 = shrink_bit_map(c2, game_state.r)
        x_y_cand = x_y_candidates(game_state.I, alphabet_map)

        for q in range(T.get_state_count()):
            for p in range(T.get_state_count):
                for x_y in range(alphabet_map):
                    if cur_c1 & bit_pos_mask(q) or cur_c2 & bit_pos_mask(p) and x_y in x_y_cand:
                        update_game_state(game_state, q, p, x_y, cur_c1, cur_c2, alphabet_map)

        # if(T.get_successor())


# Returns a list of [x,y] (encoded as ints) for which y is in I
def x_y_candidates(I, alphabet_map):
    candidates = []
    for y in range(alphabet_map.get_sigma_size()):  # iterate over all symbols in sigma
        if bit_pos_mask(y) & I != 0:  # is sigma in I?
            for x in range(alphabet_map.get_sigma_size()):  # combine y with all x in sigma
                candidates.append(alphabet_map.combine_bits(x, y))
    return candidates

def state_candidates(l, n, c):
    return (c[:l], c[:l+1])[l<n]

def update_game_state(game_state, q, p, x_y, cur_c1, cur_c2, alphabet_map):
    if cur_c1 & q == 0:
        game_state.inc_l()
    if cur_c2 & p == 0:
        game_state.inc_r()
    game_state.xor_I(get_input(x_y, alphabet_map.get_sigma_bit_len()))





if __name__ == '__main__':
    m = Storage.AlphabetMap(['a', 'b', 'c', 'd'])
    I = 0b0011

    for b in x_y_candidates(I, m):
        print(bin(b))

