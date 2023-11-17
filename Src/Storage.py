import itertools
import math

from scipy.sparse import dok_array
from abc import ABC, abstractmethod
from itertools import *


# abstract class that defines how transitions are saved and accessed
class AbstractStorage(ABC):

    @abstractmethod
    def __init__(self, state_count, symbol_count):
        pass

    @abstractmethod
    def add_transition(self, ke1, ke2, target):
        pass

    @abstractmethod
    def get_successor(self, ke1, key2):
        pass

    @abstractmethod
    def state_iterator(self):
        pass

    @abstractmethod
    def __str__(self):
        pass


# TODO: Replace with efficient implementation
class SimpleStorageNFA(AbstractStorage):
    """Maps (state, symbol) to [state]. Used for Seperator Transducer"""

    def __init__(self, state_count, symbol_count):  # symbol_count refers to the count sigma x sigma
        self.state_count = 0
        self.dictionary = {}

    def add_transition(self, origin, symbol, target):
        self.state_count += 1
        if self.dictionary.get((origin, symbol)) is None:
            self.dictionary[(origin, symbol)] = [target]
        else:
            self.dictionary[(origin, symbol)].append(target)

    def get_successor(self, origin, symbol):
        if self.dictionary.get((origin, symbol)) is not None:
            return self.dictionary[(origin, symbol)]
        return None

    def state_iterator(self):
        return range(0, self.state_count)

    def __str__(self):
        result = ""
        for (state, symbol) in self.dictionary:
            result += "state: " + str(state) + " symbol: " + str(symbol) + " target: " + str(
                self.dictionary[(state, symbol)]) + "\n"
        return result


class SparseStorage(AbstractStorage):
    """store transitions in a sparse matrix with State Action Pairs <(origin,symbol)> = target"""

    def __init__(self, row_count, column_count):
        self.states = row_count
        self.sparseMatrix = dok_array((row_count, column_count), dtype=int)

    def add_transition(self, row_index, column_index, entry):
        self.sparseMatrix[(row_index, column_index)] = entry + 1

    def get_successor(self, row_index, column_index):
        if (row_index, column_index) in self.sparseMatrix:
            return self.sparseMatrix[(row_index, column_index)] - 1
        return -1

    def state_iterator(self):
        return range(0, self.states)

    def __str__(self):
        return self.sparseMatrix.toarray().__str__()


class ColumnHashing:
    # LSSF = least Significant State First
    def __init__(self, LSSF):
        self.LSSF = LSSF
        self.mapping = {}

    def store_column(self, column_hash, column_list):
        """Maps [0, 1, 2] -> q0q1q2 and stores result in a map with key column_hash"""
        column_str = ""
        for index in column_list:
            if self.LSSF:
                column_str = column_str + "q" + str(index)  # TODO: Revert order for columns again!
            else:
                column_str = "q" + str(index) + column_str
        self.mapping[column_hash] = column_str

    def get_column_str(self, column_hash):
        """Returns the string representation of the column state encoded by column_hash"""
        return self.mapping[column_hash]


class AlphabetMap:
    """Maps the symbols in sigma to int"""

    def __init__(self, sigma):
        self.sigma = sigma
        self.bits = int(math.ceil(math.log2(len(sigma))))
        self.symbolIntMap = self.init_map()

    def init_map(self):
        """initializes the map (example: a -> 0, b -> 1, c -> 2)"""
        tmp = {}
        for i, sym in enumerate(self.sigma):
            tmp[sym] = i
        return tmp

    def sigma_iterator(self):
        return range(0, self.get_sigma_size())

    def sigma_x_sigma_iterator(self):
        return map(lambda x_y: self.combine_x_and_y(x_y[0], x_y[1]), product(self.sigma_iterator(), self.sigma_iterator()))

    def get_sigma_size(self):
        """Returns the size of the alphabet sigma """
        return len(self.sigma)

    def get_sigma_encoding_num_bits(self):
        """Returns the number of bits necessary to encode the symbols in sigma """
        return self.bits

    def get_num_symbols_in_sigma_x_sigma(self):
        """Returns |sigma * sigma|. Used to determine the amount of [u,v] pairs"""
        return len(self.sigma) * len(self.sigma)

    def get_bit_map_sigma(self):
        """Returns a bit map of length sigma"""
        return (1 << len(self.sigma)) - 1

    def symbol_to_int(self, sym):
        """Maps a symbol to its integer representation"""
        return self.symbolIntMap[sym]

    def int_to_symbol(self, x):
        return self.sigma[x]

    def combine_x_and_y(self, x, y):
        """Combines int x and y in to [x,y] (with the first bit of y being the LSB"""
        return x << self.bits | y

    def combine_symbols(self, sym_x, sym_y):
        """Combines string x and y"""
        return self.combine_x_and_y(self.symbol_to_int(sym_x), self.symbol_to_int(sym_y))

    def get_x(self, t_vec):
        """Returns x from [x,y]"""
        return t_vec >> self.bits

    def get_y(self, t_vec):
        """Returns y from [x,y]"""
        return t_vec & (1 << self.bits) - 1

    def transition_to_str(self, x_y):
        """Maps bit-representation of x_y to string """
        return "[" + self.sigma[(self.get_x(x_y))] + "," + self.sigma[(self.get_y(x_y))] + "]"

    def __str__(self):
        tmp = ""
        for sym in self.symbolIntMap:
            tmp += str(sym) + "->" + bin(self.symbolIntMap[sym]) + "\n"
        return tmp


class Statistics:
    def __init__(self):
        self.total_transitions = 0

    def log_transition(self):
        self.total_transitions += 1


if __name__ == '__main__':
    alph_map = AlphabetMap(['n', 't'])
