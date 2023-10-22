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

    def set_l(self, new_l):
        self.l = new_l

    def set_I(self, new_I):
        self.I = new_I

    def set_r(self, new_r):
        self.r = new_r

    def inc_l(self):
        self.l += 1

    def xor_I(self, bit_map):
        self.I = self.I ^ bit_map

    def inc_r(self):
        self.r += 1

    def __str__(self):
        return "<" + str(self.l) + "," + str(bin(self.I)) + "," + str(self.r) + ">"
