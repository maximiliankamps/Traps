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
    def __init__(self, state_count, symbol_count):  # symbol_count referees to the count sigma x sigma
        self.sparseMatrix = dok_array((state_count, symbol_count), dtype=int)

    def add_transition(self, origin, symbol, target):
        self.sparseMatrix[(origin, symbol)] = target

    def get_successor(self, origin, symbol):
        if (origin, symbol) in self.sparseMatrix:
            return self.sparseMatrix[(origin, symbol)]
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

    def get_sigma_bit_len(self):
        return self.bits

    # Returns the bit length of a sigma x sigma tuple
    def get_vec_bit_len(self):
        return self.bits * 2

    # Returns |sigma * sigma|
    def get_num_csym(self):
        return len(self.sigma) * len(self.sigma)

    # Encodes sigma as a bit map
    def get_bit_map_sigma(self):
        return (1 << len(self.sigma)) - 1

    # Maps a symbol in sigma to an integer
    def map_symbol(self, x):
        return self.symbolIntMap[x]

    def combine_bits(self, x, y):
        return x << self.bits | y

    # Combines symbols x and y to a 32bit integer
    # Used for sigma x sigma transitions
    def combine_symbols(self, x, y):
        return self.combine_bits(self.map_symbol(x), self.map_symbol(y))

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
