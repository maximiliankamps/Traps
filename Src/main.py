import time
import signal

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

    print(f'---------{a.transition_to_str(a.combine_x_and_y(1, 1))}---------')
    log_succ([1,2], 1, 1)
"""

benchmarks = [
    ("Burns.json", ["nomutex"]),
    ("bakery.json", ["nomutex"]),
    ("MESI.json", ["modifiedmodified", "sharedmodified"]),
    ("MOESI.json", ["modifiedmodified", "exclusiveexclusive", "sharedexclusive", "ownedexclusive", "exclusivemodified",
                    "ownedmodified", "sharedmodified"]),
    ("synapse.json", ["dirtydirty", "dirtyvalid"]),
    ("dining-cryptographers.json", ["internal", "external"]),
    ("token-passing.json", ["manytoken", "notoken", "onetoken"]),
]

gen_implementations = {"buffer_bfs": Algorithms.OneshotSmart.step_game_gen_buffered_bfs,
                       "simple_dfs": Algorithms.OneshotSmart.step_game_gen_simple_dfs,
                       "buffer_dfs": Algorithms.OneshotSmart.step_game_gen_buffered_dfs}
oneshot_implementations = {"multi_disprove",
                           "min_disprove",
                           "dfs",
                           "bfs"}


class Timeout(Exception):
    pass


def try_one(func, t, gen_imp):
    result = ""

    def timeout_handler(signum, frame):
        raise Timeout()

    old_handler = signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(t)  # trigger alarm in 3 seconds

    try:
        t1 = time.time()
        result = func(gen_imp)
        t2 = time.time()

    except Timeout:
        print('{} timed out after {} seconds'.format(func.__name__, t))
        return None
    finally:
        signal.signal(signal.SIGALRM, old_handler)

    signal.alarm(0)
    return result


max_time = 300


def execute_benchmarks(benchmark_list, gen_name, oneshot_name, ignore_ambiguous):
    gen_imp = gen_implementations.get(gen_name)
    oneshot_imp = oneshot_name in oneshot_implementations

    if gen_imp is None:
        print(f'Generator "{gen_name}" implementation does not exists!')
        return
    if oneshot_imp is False:
        print(f'Oneshot "{oneshot_name}" implementation does not exists!')
        return

    print(f'Using generator: "{gen_name}" and oneshot implementation "{oneshot_name}":')
    for benchmark_name, testcases in benchmark_list:
        print("================================================")
        print(benchmark_name)
        print("================================================")
        for test in testcases:
            print(test)
            rts = Automata.RTS(benchmark_name)
            t = rts.get_T()

            ixb = rts.get_IxB(test)
            #ixb.to_dot("deadlock", None)

            start_time = time.time()

            o = Algorithms.OneshotSmart(ixb, t)
            o.ignore_ambiguous = ignore_ambiguous
            result = None
            if oneshot_name == "multi_disprove":
                result = try_one(o.multi_disprove_oneshot, max_time, gen_imp)
            elif oneshot_name == "min_disprove":
                result = try_one(o.min_sigma_disprove_oneshot, max_time, gen_imp)
            elif oneshot_name == "dfs":
                result = try_one(o.one_shot_dfs_standard, max_time, gen_imp)
            elif oneshot_name == "bfs":
                result = try_one(o.one_shot_bfs, max_time, gen_imp)

            o.print_oneshot_result(result)

            end_time = time.time()

            print(f'elapsed_time: {end_time - start_time}')
            print("------------------------------------------------")



if __name__ == '__main__':
    # execute_benchmarks(benchmarks, "buffer_dfs", "dfs", True)
    benchmarks = [("voting-token-passing.json", ["gamewon"])]
    #execute_benchmarks(benchmarks, "buffer_dfs", "dfs", True)
    execute_benchmarks(benchmarks, "buffer_bfs", "bfs", True)
    #execute_benchmarks(benchmarks, "buffer_bfs", "bfs", True)

