class Triple:
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
        return "<" + str(self.l) + "," + strS(self.I) + "," + str(self.r) + ">"


def strS(S):
    if S == 0:
        return "[]"
    if S == 1:
        return "[n]"
    if S == 2:
        return "[t]"
    else:
        return "[n, t]"


def symbol_not_in_seperator(S, i):
    """Returns true if the symbol with index i is in S"""
    return (S & (1 << i)) == 0


def optional_list(l):
    if l is None:
        return []
    return l
