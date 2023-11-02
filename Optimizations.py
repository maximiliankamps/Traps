from itertools import *
from Util import *


class ColumnMemoization:
    """Memorizes Steps in the Step game as a tree"""
    def __init__(self, node_size):
        self.node_size = node_size
        self.root_nodes = {}

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
            current = Node(False, symbol_in_seperator(S, 0))
            self.root_nodes[(c1[0], c2[0], u, S)] = current

        for I_index, (b1, b2) in enumerate(zip_longest(c1[1:], c2[1:])):
            next_node = current.get_next(b1, b2)
            if next_node is None:
                next_node = Node(False, symbol_in_seperator(I, I_index + 1))
                current.add_next(b1, b2, next_node)
            current = next_node
        current.set_miss(miss)

    def check_step(self, c1, c2, u, S):
        """
        :param c1: List of states
        :param c2: List of states
        :param u: Symbol
        :param S: Seperator bit map
        :return: The precomputed Game State or None if the Step Game was lost
        """
        game_state = Triple(0, 0, 0)
        current = self.root_nodes.get((c1[0], c2[0], u, S))

        for I_index, (b1, b2) in enumerate(zip_longest(c1[1:], c2[1:])):
            if current is None:
                return game_state
            if current.is_miss():
                return None
            game_state.update(b1, b2, I_index, current.get_I_bit())
            current = current.get_next(b1, b2)
        return game_state

    def __str__(self):
        for (b1, b2, u, S) in self.root_nodes.keys():
            current = self.root_nodes[(b1, b2, u, S)]
            return "u: " + str(u) + " | S: " + bin(S) + "\n" + current.to_string(b1, b2, 1)


class Node:
    def __init__(self, I_bit, miss):
        self.next_nodes = {}
        self.miss = miss
        self.I_bit = I_bit

    def add_next(self, b1, b2, node):
        self.next_nodes[(b1, b2)] = node

    def get_next(self, b1, b2):
        return self.next_nodes.get((b1, b2))

    def get_I_bit(self):
        return int(self.I_bit)

    def set_miss(self, miss):
        self.miss = miss

    def is_miss(self):
        return self.miss

    def to_string(self, b1, b2, layer):
        string_acc = "b: " + str((b1, b2)) + " | I_bit: " + str(self.get_I_bit()) + " | miss: " + str(self.is_miss())
        for (b1_next, b2_next) in self.next_nodes:
            string_acc += "\n" + ("*" * layer) + self.next_nodes[(b1_next, b2_next)].to_string(b1_next, b2_next, layer + 1)
        return string_acc


if __name__ == '__main__':
    i = ColumnMemoization(10)
    i.add_node([0], [0], 0, 0, 0, True)
    print(i)
    print(i.check_step([0], [1, 0], 0, 0))
