#!/usr/bin/env python

"""
Run the test suite using a separate process for each test file.
"""

import multiprocessing
import os
import platform
import shlex
import subprocess
import sys

from optparse import OptionParser

def try_timeout(command):
    if not sys.platform.startswith("win32"):
        try:
            return {0: "passed", 124: "timed out"}.get(
                subprocess.call(["timeout", "5m"] + command), "failed")
        except OSError:
            pass
        try:
            return {0: "passed", 124: "timed out"}.get(
                subprocess.call(["gtimeout", "5m"] + command), "failed")
        except OSError:
            pass
    return "passed" if subprocess.call(command) == 0 else "failed"

def process_dir(root, files, test_root, dotest_options):
    """Examine a directory for tests, and invoke any found within it."""
    timed_out = []
    failed = []
    passed = []
    for name in files:
        path = os.path.join(root, name)

        # We're only interested in the test file with the "Test*.py" naming pattern.
        if not name.startswith("Test") or not name.endswith(".py"):
            continue

        # Neither a symbolically linked file.
        if os.path.islink(path):
            continue

        command = ([sys.executable, "%s/dotest.py" % test_root] +
                   (shlex.split(dotest_options) if dotest_options else []) +
                   ["-p", name, root])
        exit_status = try_timeout(command)
        if "passed" != exit_status:
            if "timed out" == exit_status:
                timed_out.append(name)
            failed.append(name)
        else:
            passed.append(name)
    return (timed_out, failed, passed)

in_q = None
out_q = None

def process_dir_worker(arg_tuple):
    """Worker thread main loop when in multithreaded mode.
    Takes one directory specification at a time and works on it."""
    (root, files, test_root, dotest_options) = arg_tuple
    return process_dir(root, files, test_root, dotest_options)

def walk_and_invoke(test_root, dotest_options, num_threads):
    """Look for matched files and invoke test driver on each one.
    In single-threaded mode, each test driver is invoked directly.
    In multi-threaded mode, submit each test driver to a worker
    queue, and then wait for all to complete."""

    # Collect the test files that we'll run.
    test_work_items = []
    for root, dirs, files in os.walk(test_root, topdown=False):
        test_work_items.append((root, files, test_root, dotest_options))

    # Run the items, either in a pool (for multicore speedup) or
    # calling each individually.
    if num_threads > 1:
        pool = multiprocessing.Pool(num_threads)
        test_results = pool.map(process_dir_worker, test_work_items)
    else:
        test_results = []
        for work_item in test_work_items:
            test_results.append(process_dir_worker(work_item))

    timed_out = []
    failed = []
    passed = []

    for test_result in test_results:
        (dir_timed_out, dir_failed, dir_passed) = test_result
        timed_out += dir_timed_out
        failed += dir_failed
        passed += dir_passed

    return (timed_out, failed, passed)

def main():
    test_root = sys.path[0]

    parser = OptionParser(usage="""\
Run lldb test suite using a separate process for each test file.
""")
    parser.add_option('-o', '--options',
                      type='string', action='store',
                      dest='dotest_options',
                      help="""The options passed to 'dotest.py' if specified.""")

    parser.add_option('-t', '--threads',
                      type='int',
                      dest='num_threads',
                      help="""The number of threads to use when running tests separately.""")

    opts, args = parser.parse_args()
    dotest_options = opts.dotest_options

    if opts.num_threads:
        num_threads = opts.num_threads
    else:
        num_threads_str = os.environ.get("LLDB_TEST_THREADS")
        if num_threads_str:
            num_threads = int(num_threads_str)
        else:
            num_threads = multiprocessing.cpu_count()
    if num_threads < 1:
        num_threads = 1

    system_info = " ".join(platform.uname())
    (timed_out, failed, passed) = walk_and_invoke(test_root, dotest_options,
                                                  num_threads)
    num_tests = len(failed) + len(passed)

    print "Ran %d tests." % num_tests
    if len(failed) > 0:
        failed.sort()
        print "Failing Tests (%d)" % len(failed)
        for f in failed:
            print "%s: LLDB (suite) :: %s (%s)" % (
                "TIMEOUT" if f in timed_out else "FAIL", f, system_info
            )
        sys.exit(1)
    sys.exit(0)

if __name__ == '__main__':
    main()
