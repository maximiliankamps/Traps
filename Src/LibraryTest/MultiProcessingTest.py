from functools import cache
from multiprocessing import Pool, TimeoutError, Manager
import time
import os


@cache
def f(x):
    print('Cache miss: ', x)
    return x * x


def g():
    l = [2, 2, 1, 2, 2, 1, 2, 1, 2, 2, 1, 2, 1, 2]
    for x in l:
        yield x


if __name__ == '__main__':
    with Manager() as manager:
        d = manager.dict()
        l = manager.list(range(10))

        for i in l:
            print(i)
    # start 4 worker processes
    with Pool(processes=4) as pool:
        # print "[0, 1, 4,..., 81]"
        print(pool.map(f, g()))
