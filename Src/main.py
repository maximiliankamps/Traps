import time

from Algorithms import OneshotSmart, OneShotSimple
import Automata
from ray import remote

if __name__ == '__main__':
    rts = Automata.RTS("token-passing.json")
    #(rts.get_IxB("sharedexclusive")).to_dot("test", None)
    """
    # Example of OneShotSimple
    x = Algorithms.OneShotSimple()
    x.one_shot_bfs(rts.get_I(), rts.get_T(), rts.get_B("equal"))
    """
    # Example of OneShotSmart
    time_list = []
    t = rts.get_T()
    ixb = rts.get_IxB("notoken")
    for i in range(0, 1):
        start_time = time.time()
        o = OneshotSmart(ixb, t)
        o.one_shot_bfs_multiprocessing()
        end_time = time.time()

        elapsed_time = end_time - start_time
        time_list.append(elapsed_time)
    print("Average time: ", sum(time_list) / len(time_list), "s")
