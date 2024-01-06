import time

import Algorithms
from Algorithms import refine_seperator
import Automata
from Util import Triple
from itertools import product


def log_succ(state, u, v):
    rts = Automata.RTS("voting-token-start.json")
    t = rts.get_T()
    ixb = rts.get_IxB("notokennomarked")
    gs = Triple(0, refine_seperator(rts.get_T().alphabet_map.get_bit_map_sigma(), u), 0)
    o = Algorithms.OneshotSmart(ixb, t)
    for x in o.step_game_gen_buffered_dfs(state, [], v, gs, []):
        print("")

"""
if __name__ == '__main__':
    rts = Automata.RTS("voting-token-start.json")
    a = rts.get_T().alphabet_map

    print(f'---------{a.transition_to_str(a.combine_x_and_y(0, 2))}---------')
    log_succ([1,2], 3, 2)

"""
if __name__ == '__main__':
    rts = Automata.RTS("Burns.json")

    time_list = []
    t = rts.build_SxD_transducer(1)
    ixb = rts.get_IxB("sigma")

    rts.get_T().to_dot("T", None)
    rts.T.get_deadlock_transducer().to_dot("t", None)
    t.to_dot("x", None)
    print(t.final_states)
    ixb.to_dot("initial", None)
    print(ixb.final_states)

    for i in range(0, 1):
        start_time = time.time()
        o = Algorithms.OneshotSmart(ixb, t)
        o.print_oneshot_result(o.one_shot_dfs_standard())
        end_time = time.time()

        elapsed_time = end_time - start_time
        time_list.append(elapsed_time)
    print("Average time: ", sum(time_list) / len(time_list), "s")



