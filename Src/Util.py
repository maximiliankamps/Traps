"""Different helper functions for the one_shot implementations"""


class Triple:
    """Stores the game state <l, I, r> as described in the Paper"""
    def __init__(self, l, I, r):
        self.l = l
        self.I = I
        self.r = r

    def update(self, b1, b2, I):
        if b1 is not None:
            self.l = self.l + 1
        if b2 is not None:
            self.r = self.r + 1
        self.I = I

    def xor_I(self, I):
        self.I = self.I ^ I

    def get_I(self):
        return self.I

    def get_l(self):
        return self.l

    def get_r(self):
        return self.r

    def copy(self, triple):
        self.l = triple.get_l()
        self.I = triple.get_I()
        self.r = triple.get_r()

    def equal(self, triple):
        return self.l == triple.get_l() and self.I == triple.I and self.r == triple.r

    def __str__(self):
        return "<" + str(self.l) + "," + self.I + "," + str(self.r) + ">"


def symbol_not_in_seperator(S, i):
    """Returns true if the symbol with index i is in S"""
    return (S & (1 << i)) == 0


def slice_column(col, i):
    """Slices the columns c1 and c2 by i"""
    return col[:i]


def refine_seperator(S, i):
    """Remove the 1-bit at position i from S"""
    return S & ~(1 << i)


def bit_map_seperator_to_inv_list(S, n):
    """Returns the indices of all symbols that are not contained in the bit-map of the seperator S"""
    inv_list = []
    for i in range(0, n):
        if S & (1 << i) == 0:
            inv_list.append(i)
    return inv_list


def optional_list(l):
    if l is None:
        return []
    return l
