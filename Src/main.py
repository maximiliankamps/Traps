import time
import signal

import Algorithms
from Algorithms import refine_seperator
import Automata
from Util import Triple


def log_succ(state, u, v):
    rts = Automata.RTS("voting-token-start.json")
    t = rts.get_T()
    ixb = rts.get_IxB("notokennomarked")
    gs = Triple(0, refine_seperator(rts.get_T().alphabet_map.get_bit_map_sigma(), u), 0)
    o = Algorithms.OneshotSmart(ixb, t)
    for x in o.step_game_gen_buffered_dfs(state, [], v, gs, []):
        print(x)


"""
if __name__ == '__main__':
    rts = Automata.RTS("voting-token-start.json")
    a = rts.get_T().alphabet_map
    rts.get_T().to_dot("m", None)

    print(f'---------{a.transition_to_str(a.combine_x_and_y(3, 2))}---------')
    log_succ([1], 3, 2)
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
    ("voting-token-passing.json", ["initial", "gamewon", "notokennomarked"])
]

benchmarks2 = [("voting-token-passing.json", ["gamewon"])]
benchmarks3 = [("voting-token-passing.json", ["initial", "gamewon", "notokennomarked"])]

gen_implementations = {"buffer_bfs": Algorithms.OneshotSmart.step_game_gen_buffered_bfs,
                       "simple_dfs": Algorithms.OneshotSmart.step_game_gen_simple_dfs,
                       "buffer_dfs": Algorithms.OneshotSmart.step_game_gen_buffered_dfs}
oneshot_implementations = {"multi_disprove",
                           "min_disprove",
                           "dfs",
                           "bfs"}

max_time = 20 * 601


class Timeout(Exception):
    pass


def try_one(o, func, t, gen_imp):
    def timeout_handler(signum, frame):
        raise Timeout()

    old_handler = signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(t)

    try:
        t1 = time.time()
        result = func(gen_imp)
        t2 = time.time()
        o.print_oneshot_result(result)

    except Timeout:
        print('{} timed out after {} seconds'.format(func.__name__, t))
        return None
    finally:
        signal.signal(signal.SIGALRM, old_handler)


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
            # t.to_dot("t", None)
            # ixb.to_dot("ixb", None)

            start_time = time.time()

            o = Algorithms.OneshotSmart(ixb, t)
            o.ignore_ambiguous = ignore_ambiguous
            if oneshot_name == "multi_disprove":
                try_one(o, o.multi_disprove_oneshot, max_time, gen_imp)
            elif oneshot_name == "min_disprove":
                try_one(o, o.min_sigma_disprove_oneshot, max_time, gen_imp)
            elif oneshot_name == "dfs":
                try_one(o, o.one_shot_dfs_standard, max_time, gen_imp)
            elif oneshot_name == "bfs":
                try_one(o, o.one_shot_bfs, max_time, gen_imp)

            end_time = time.time()

            print(f'elapsed_time: {end_time - start_time}')
            print("------------------------------------------------")


"""Run all benchmarks will all implementations"""
if __name__ == '__main__':
    execute_benchmarks(benchmarks, "buffer_bfs", "bfs", True)
    print("=======================================================================================")
    print("Ignore ambiguous == false")
    print("=======================================================================================")
    execute_benchmarks(benchmarks, "buffer_bfs", "bfs", False)
    print("=======================================================================================")
    execute_benchmarks(benchmarks, "buffer_bfs", "dfs", False)
    print("=======================================================================================")
    execute_benchmarks(benchmarks, "buffer_dfs", "bfs", False)
    print("=======================================================================================")
    execute_benchmarks(benchmarks, "buffer_dfs", "dfs", False)
    print("=======================================================================================")
    print("Ignore ambiguous == true")
    print("=======================================================================================")
    print("=======================================================================================")
    execute_benchmarks(benchmarks, "buffer_bfs", "dfs", True)
    print("=======================================================================================")
    execute_benchmarks(benchmarks, "buffer_dfs", "bfs", True)
    print("=======================================================================================")
    execute_benchmarks(benchmarks, "buffer_dfs", "dfs", True)
    print("=======================================================================================")
    print("Sigma disprove")
    print("=======================================================================================")
    execute_benchmarks(benchmarks, "buffer_bfs", "min_disprove", True)
    execute_benchmarks(benchmarks, "buffer_dfs", "min_disprove", True)
