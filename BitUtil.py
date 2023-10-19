# Returns u from the bit tuple [u, v]
# n is the length of the tuple
def get_input(t_vec, n):
    return t_vec << n


# Returns v from the bit tuple [u, v]
# n is the length of the tuple
def get_output(t_vec, n):
    return t_vec & (1 << n) - 1


def bit_pos_mask(pos):
    return 1 << pos


def shrink_bit_map(bit_map, n):
    pos = 0
    while n > 0:
        if bit_map & bit_pos_mask(pos):
            n -= 1
        pos += 1
    return get_output(bit_map, pos)
