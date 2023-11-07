from abc import ABC, abstractmethod
from itertools import *
from Util import *
import math

from pyeda.inter import *
import bitarray


class AbstractStepGameMemo(ABC):

    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def add_node(self, c1, c2, u, S, I, miss):
        pass

    @abstractmethod
    def check_step(self, c1, c2, u, S, initial_I):
        pass

    @abstractmethod
    def print_statistics(self):
        pass


class StepGameMemo(AbstractStepGameMemo):
    """Memorizes Steps in the Step Game as a tree"""

    def __init__(self, node_size):
        self.node_size = node_size
        self.root_nodes = {}
        self.total_checks = 0
        self.total_excluded = 0
        self.total_advanced = 0

    def add_node(self, c1, c2, u, S, I, miss):
        """
        :param c1: List of states
        :param c2: List of states
        :param u: Symbol
        :param S: Seperator bit map
        :param I: Game State Seperator bit map
        :param miss: Was the step rejected by step game
        :return:
        """
        if self.node_size == 0:  # Table is full
            return
        self.node_size -= 1

        current = self.root_nodes.get((c1[0], c2[0], u, S))
        if current is None:
            current = Node(False, -1)
            self.root_nodes[(c1[0], c2[0], u, S)] = current

        for I_index, (b1, b2) in enumerate(zip_longest(c1[1:], c2[1:])):
            next_node = current.get_next(b1, b2)
            if next_node is None:
                next_node = Node(False, -1)
                current.add_next(b1, b2, next_node)
            current = next_node
        current.set_I_tmp(I)
        current.set_miss(miss)

    def check_step(self, c1, c2, u, S, initial_I):
        self.total_checks = self.total_checks + 1
        """
        :param c1: List of states
        :param c2: List of states
        :param u: Symbol
        :param S: Seperator bit map
        :param initial_I: The I with which the Game State was initialized
        :return: The precomputed Game State or None if the Step Game was lost
        """
        game_state = Triple(0, 0, 0)
        current = self.root_nodes.get((c1[0], c2[0], u, S))

        for (b1, b2) in zip_longest(c1[1:], c2[1:]):
            if current is None:
                break
            if current.is_miss():
                self.total_excluded = self.total_excluded + 1
                return None
            game_state.update(b1, b2, current.get_I_tmp())
            current = current.get_next(b1, b2)
        if game_state.l != 0 or game_state.r != 0:
            self.total_advanced += 1
        game_state.xor_I(initial_I)  # Remove all symbols that have been removed in pre-computation
        return game_state

    def print_statistics(self):
        print("Chache size left: " + str(self.node_size))
        print("total checked: " + str(self.total_checks))
        print("total excluded: " + str(self.total_excluded))
        print("total advanced: " + str(self.total_advanced))

    def __str__(self):
        string_acc = ""
        for (b1, b2, u, S) in self.root_nodes.keys():
            current = self.root_nodes[(b1, b2, u, S)]
            string_acc += "u: " + str(u) + " | S: " + bin(S) + "\n" + current.to_string(b1, b2, 1) + "\n"
        return string_acc


class Node:
    def __init__(self, I_tmp, miss):
        self.next_nodes = {}
        self.miss = miss
        self.I_tmp = I_tmp

    def add_next(self, b1, b2, node):
        self.next_nodes[(b1, b2)] = node

    def get_next(self, b1, b2):
        return self.next_nodes.get((b1, b2))

    def get_I_tmp(self):
        return self.I_tmp

    def set_I_tmp(self, I_tmp):
        self.I_tmp = I_tmp

    def set_miss(self, miss):
        self.miss = miss

    def is_miss(self):
        return self.miss

    def to_string(self, b1, b2, layer):
        string_acc = "b: " + str((b1, b2)) + " | I_tmp: " + bin(self.get_I_tmp()) + " | miss: " + str(self.is_miss())
        for (b1_next, b2_next) in self.next_nodes:
            string_acc += "\n" + ("*" * layer) + self.next_nodes[(b1_next, b2_next)].to_string(b1_next, b2_next,
                                                                                               layer + 1)
        return string_acc


class StepGameMemo2(AbstractStepGameMemo):
    """Memorizes Steps in the Step Game as BDD"""
    def __init__(self, u_bits_num, state_bit_num):
        self.u_bits_num = u_bits_num
        self.S_bits_num = int(math.pow(2, self.u_bits_num))
        self.state_bit_num = state_bit_num
        self.column_len = int(math.pow(2, self.state_bit_num))
        self.encoding_bit_num = self.u_bits_num + self.S_bits_num + 2 * (self.state_bit_num + self.column_len)
        self.bdd_vars = list(map(lambda b: bddvar(f"b{b}", b), [*range(0, self.encoding_bit_num)]))
        self.total_checks = 0
        self.total_excluded = 0
        self.f = None

    def zero_padding(self, bin_num, length):
        """pads bin_num with 0s"""
        return ("0" * (length - len(bin_num))) + bin_num

    def encode(self, c1, c2, u, S):  # TODO: Needs to become alot faster
        """encodes c1, c2, u, S as a bitarray bin(u) + bin(S) + bin(c1) + bin(c2)"""
        u_bin = self.zero_padding(format(u, "b"), self.u_bits_num)
        S_bin = self.zero_padding(format(S, "b"), self.S_bits_num)
        c1_bin = ""
        c2_bin = ""
        if c1 is not None:
            c1_bin = "".join(list(map(lambda b: self.zero_padding(format(b, "b"), self.state_bit_num), c1)))
        if c2 is not None:
            c2_bin = "".join(list(map(lambda b: self.zero_padding(format(b, "b"), self.state_bit_num), c2)))

        return bitarray.bitarray(u_bin + S_bin + c1_bin + c2_bin)

    def add_node(self, c1, c2, u, S, I, miss):
        """uses the encoding to set bbd_vars to (u & S & c1 & c2) | f_tmp"""
        if not miss:
            return

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

    def check_step(self, c1, c2, u, S, initial_I):
        if self.f is None:
            return Triple(0, initial_I, 0)
        self.total_checks += 1
        encoding = self.encode(c1, c2, u, S)

        point = {self.bdd_vars[i]: bit for (i, bit) in enumerate(encoding)}

        check = self.f.restrict(point)
        if check:
            self.total_excluded += 1
            return None
        else:
            return Triple(0, initial_I, 0)

    def print_statistics(self):
        print("u bit size: " + str(self.u_bits_num))
        print("S bit size: " + str(self.S_bits_num))
        print("state bit size: " + str(self.state_bit_num))
        print("max column len: " + str(self.column_len))
        print("max encoding length: " + str(self.encoding_bit_num))
        print("---------------------")
        print("total checked: " + str(self.total_checks))
        print("total excluded: " + str(self.total_excluded))
