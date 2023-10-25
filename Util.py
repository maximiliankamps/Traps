class Triple:
    def __init__(self, l, I, r):
        self.l = l
        self.I = I
        self.r = r

    def get_l(self):
        return self.l

    def get_I(self):
        return self.I

    def get_r(self):
        return self.r

    def __str__(self):
        return "<" + str(self.l) + "," + str(bin(self.I)) + "," + str(self.r) + ">"


def optional_list(l):
    if l is None:
        return []
    return l
