import time


def miliseconds() -> int:
    return nanoseconds() // 1_000_000

def nanoseconds() -> int:
    return time.time_ns()

def seconds() -> int:
    return int(time.time())