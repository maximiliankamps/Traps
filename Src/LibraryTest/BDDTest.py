import math

from pyeda.inter import *
import bitarray


class StepGameMemo2:
    def __init__(self, u_bits_num, state_bit_num):
        self.u_bits_num = u_bits_num
        self.S_bits_num = int(math.pow(2, self.u_bits_num))
        self.state_bit_num = state_bit_num
        self.column_len = int(math.pow(2, self.state_bit_num))
        self.encoding_bit_num = self.u_bits_num + self.S_bits_num + 2 * (self.state_bit_num + self.column_len)
        self.bdd_vars = list(map(lambda b: bddvar(f"b{b}", b), [*range(0, self.encoding_bit_num)]))
        self.f = None
    def zero_padding(self, bin_num, length):
        return ("0" * (length -     len(bin_num))) + bin_num

    def encode(self, c1, c2, u, S):
        u_bin = self.zero_padding(format(u, "b"), self.u_bits_num)
        S_bin = self.zero_padding(format(S, "b"), self.S_bits_num)
        c1_bin = ""
        c2_bin = ""
        if c1 is not None:
            c1_bin = ("".join(list(map(lambda b: self.zero_padding(format(b, "b"), self.state_bit_num), c1))))
        if c2 is not None:
            c2_bin = "".join(list(map(lambda b: self.zero_padding(format(b, "b"), self.state_bit_num), c2)))

        return bitarray.bitarray(u_bin + S_bin + c1_bin + c2_bin)

    def add_step(self, c1, c2, u, S, I):
        encoding = self.encode(c1, c2, u, S)

        tmp = bddvar("tmp") | ~bddvar("tmp")
        for i, bit in enumerate(encoding):
            if bit:
                tmp = tmp & self.bdd_vars[i]
            else:
                tmp = tmp & ~self.bdd_vars[i]
        if self.f is None:
            self.f = tmp
        else:
            self.f = self.f | tmp
        return self.f

    def check_step(self, c1, c2, u, S, I):
        if self.f is None:
            return 0

        encoding = self.encode(c1, c2, u, S)

        point = {self.bdd_vars[i]: bit for (i, bit) in enumerate(encoding)}

        return self.f.restrict(point)

    def print_dimensions(self):
        print("u bit size: " + str(self.u_bits_num))
        print("S bit size: " + str(self.S_bits_num))
        print("state bit size: " + str(self.state_bit_num))
        print("max column len: " + str(self.column_len))
        print("max encoding length: " + str(self.encoding_bit_num))
