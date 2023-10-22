# Returns u from the bit tuple [u, v]
# n is the length of the tuple
def get_input(t_vec, n):
    return t_vec >> n


# Returns v from the bit tuple [u, v]
# n is the length of the tuple
def get_output(t_vec, n):
    return t_vec & (1 << n) - 1

# Creates a bit mask where the bit at position pos is set: e.g.: 3 -> 0b1000
def bit_pos_mask(pos):
    return 1 << pos

# Masks the first n bits (set to 1) in bit_map: e.g.: 0b101101, 3 -> 1101
def shrink_bit_map(bit_map, n):
    pos = 0
    while n > 0:
        if bit_map & bit_pos_mask(pos):
            n -= 1
        pos += 1
    return get_output(bit_map, pos)


# Masks the nth 1-bit in bit_map: e.g.: 0b10101, 2 -> 0b00100
def nth_bit_mask(bit_map, n):
    pos = -1
    while n > 0:
        pos += 1
        if bit_map & bit_pos_mask(pos):
            n -= 1
    return bit_pos_mask(pos)

if __name__ == '__main__':
    print(get_input(2, 1))



