import math

from scipy.sparse import dok_array
from abc import ABC, abstractmethod


# abstract class that defines how transitions are saved and accessed
class AbstractStorage(ABC):

    @abstractmethod
    def __init__(self, state_count, symbol_count):
        pass

    @abstractmethod
    def add_transition(self, origin, symbol, target):
        pass

    @abstractmethod
    def get_successor(self, origin, symbol):
        pass

    @abstractmethod
    def __str__(self):
        pass


# store transitions in a sparse matrix with State Action Pairs <(origin,symbol)> = target
class SparseStorage(AbstractStorage):
    def __init__(self, state_count, symbol_count):  # symbol_count refers to the count sigma x sigma
        self.sparseMatrix = dok_array((state_count, symbol_count), dtype=int)

    def add_transition(self, origin, symbol, target):
        self.sparseMatrix[(origin-1, symbol)] = target

    def get_successor(self, origin, symbol):
        if (origin-1, symbol) in self.sparseMatrix:
            return self.sparseMatrix[(origin-1, symbol)]
        return -1

    def __str__(self):
        return self.sparseMatrix.toarray().__str__()


# Maps the symbols in Alphabet sigma to int
class AlphabetMap:
    def __init__(self, sigma):
        self.sigma = sigma
        self.bits = int(math.ceil(math.log2(len(sigma))))
        self.symbolIntMap = self.init_map()

    def init_map(self):
        tmp = {}
        for i, sym in enumerate(self.sigma):
            tmp[sym] = i
        return tmp

    def get_sigma_size(self):
        return len(self.sigma)

    def get_sigma_encoding_num_bits(self):
        return self.bits

    # Returns |sigma * sigma|
    def get_num_symbols_in_sigma_x_sigma(self):
        return len(self.sigma) * len(self.sigma)

    # Encodes sigma as a bit map
    def get_bit_map_sigma(self):
        return (1 << len(self.sigma)) - 1

    # Maps a symbol in sigma to an integer
    def symbol_to_int(self, x):
        return self.symbolIntMap[x]

    def combine_x_and_y(self, x, y):
        return x << self.bits | y

    # Combines symbols x and y to a 32bit integer
    # Used for sigma x sigma transitions
    def combine_symbols(self, sym_x, sym_y):
        return self.combine_x_and_y(self.symbol_to_int(sym_x), self.symbol_to_int(sym_y))

    # Returns u from the bit tuple [u, v]
    # n is the length of the tuple
    def get_x(self, t_vec):
        return t_vec >> self.bits

    # Returns v from the bit tuple [u, v]
    # n is the length of the tuple
    def get_y(self, t_vec):
        return t_vec & (1 << self.bits) - 1

    def transition_to_str(self, x_y):
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
