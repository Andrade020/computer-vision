"""
Keep Windows awake while a long CPU job runs, without changing any user power
settings. Uses SetThreadExecutionState; the request is released automatically
the moment this process exits (kill it to restore normal sleep behaviour).

    python -m hw.keep_awake        # holds the machine awake until killed
"""
import ctypes
import time

ES_CONTINUOUS = 0x80000000
ES_SYSTEM_REQUIRED = 0x00000001
ES_AWAYMODE_REQUIRED = 0x00000040


def main():
    flags = ES_CONTINUOUS | ES_SYSTEM_REQUIRED | ES_AWAYMODE_REQUIRED
    if ctypes.windll.kernel32.SetThreadExecutionState(flags) == 0:
        # away-mode not supported on this SKU; fall back without it
        flags = ES_CONTINUOUS | ES_SYSTEM_REQUIRED
        ctypes.windll.kernel32.SetThreadExecutionState(flags)
    print("keep_awake: system sleep suppressed (non-persistent)")
    try:
        while True:
            # re-assert periodically to be safe
            ctypes.windll.kernel32.SetThreadExecutionState(flags)
            time.sleep(60)
    finally:
        ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS)
        print("keep_awake: released")


if __name__ == "__main__":
    main()
