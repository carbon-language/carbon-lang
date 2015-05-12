#!/usr/bin/env python

"""
Run the test suite using a separate process for each test file.

Each test will run with a time limit of 10 minutes by default.

Override the default time limit of 10 minutes by setting
the environment variable LLDB_TEST_TIMEOUT.

E.g., export LLDB_TEST_TIMEOUT=10m

Override the time limit for individual tests by setting
the environment variable LLDB_[TEST NAME]_TIMEOUT.

E.g., export LLDB_TESTCONCURRENTEVENTS_TIMEOUT=2m

Set to "0" to run without time limit.

E.g., export LLDB_TEST_TIMEOUT=0
or    export LLDB_TESTCONCURRENTEVENTS_TIMEOUT=0
"""

import multiprocessing
import os
import platform
import re
import shlex
import subprocess
import sys

from optparse import OptionParser

def get_timeout_command():
    """Search for a suitable timeout command."""
    if sys.platform.startswith("win32"):
        return None
    try:
        subprocess.call("timeout")
        return "timeout"
    except OSError:
        pass
    try:
        subprocess.call("gtimeout")
        return "gtimeout"
    except OSError:
        pass
    return None

timeout_command = get_timeout_command()

default_timeout = os.getenv("LLDB_TEST_TIMEOUT") or "10m"

# Status codes for running command with timeout.
eTimedOut, ePassed, eFailed = 124, 0, 1

def call_with_timeout(command, timeout):
    """Run command with a timeout if possible."""
    if os.name != "nt":
        if timeout_command and timeout != "0":
            return subprocess.call([timeout_command, timeout] + command,
                                   stdin=subprocess.PIPE, close_fds=True)
        return (ePassed if subprocess.call(command, stdin=subprocess.PIPE, close_fds=True) == 0
                else eFailed)
    else:
        if timeout_command and timeout != "0":
            return subprocess.call([timeout_command, timeout] + command,
                                   stdin=subprocess.PIPE)
        return (ePassed if subprocess.call(command, stdin=subprocess.PIPE) == 0
                else eFailed)

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

        script_file = os.path.join(test_root, "dotest.py")
        is_posix = (os.name == "posix")
        split_args = shlex.split(dotest_options, posix=is_posix) if dotest_options else []
        command = ([sys.executable, script_file] +
                   split_args +
                   ["-p", name, root])

        timeout_name = os.path.basename(os.path.splitext(name)[0]).upper()

        timeout = os.getenv("LLDB_%s_TIMEOUT" % timeout_name) or default_timeout

        exit_status = call_with_timeout(command, timeout)

        if ePassed == exit_status:
            passed.append(name)
        else:
            if eTimedOut == exit_status:
                timed_out.append(name)
            failed.append(name)
    return (timed_out, failed, passed)

in_q = None
out_q = None

def process_dir_worker(arg_tuple):
    """Worker thread main loop when in multithreaded mode.
    Takes one directory specification at a time and works on it."""
    (root, files, test_root, dotest_options) = arg_tuple
    return process_dir(root, files, test_root, dotest_options)

def walk_and_invoke(test_directory, test_subdir, dotest_options, num_threads):
    """Look for matched files and invoke test driver on each one.
    In single-threaded mode, each test driver is invoked directly.
    In multi-threaded mode, submit each test driver to a worker
    queue, and then wait for all to complete.

    test_directory - lldb/test/ directory
    test_subdir - lldb/test/ or a subfolder with the tests we're interested in running
    """

    # Collect the test files that we'll run.
    test_work_items = []
    for root, dirs, files in os.walk(test_subdir, topdown=False):
        test_work_items.append((root, files, test_directory, dotest_options))

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

def getExpectedTimeouts(dotest_options):
    # returns a set of test filenames that might timeout
    # are we running against a remote target?
    m = re.search('\sremote-(\w+)', dotest_options)
    if m:
        target = m.group(1)
        remote = True
    else:
        target = sys.platform
        remote = False

    expected_timeout = set()

    if target.startswith("linux"):
        expected_timeout |= {
            "TestAttachResume.py",
            "TestConnectRemote.py",
            "TestCreateAfterAttach.py",
            "TestExitDuringStep.py",
            "TestThreadStepOut.py",
        }
    elif target.startswith("android"):
        expected_timeout |= {
            "TestExitDuringStep.py",
            "TestHelloWorld.py",
        }
    return expected_timeout

def main():
    # We can't use sys.path[0] to determine the script directory
    # because it doesn't work under a debugger
    test_directory = os.path.dirname(os.path.realpath(__file__))
    parser = OptionParser(usage="""\
Run lldb test suite using a separate process for each test file.

       Each test will run with a time limit of 10 minutes by default.

       Override the default time limit of 10 minutes by setting
       the environment variable LLDB_TEST_TIMEOUT.

       E.g., export LLDB_TEST_TIMEOUT=10m

       Override the time limit for individual tests by setting
       the environment variable LLDB_[TEST NAME]_TIMEOUT.

       E.g., export LLDB_TESTCONCURRENTEVENTS_TIMEOUT=2m

       Set to "0" to run without time limit.

       E.g., export LLDB_TEST_TIMEOUT=0
       or    export LLDB_TESTCONCURRENTEVENTS_TIMEOUT=0
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

    # The root directory was specified on the command line
    if len(args) == 0:
        test_subdir = test_directory
    else:
        test_subdir = os.path.join(test_directory, args[0])

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
    (timed_out, failed, passed) = walk_and_invoke(test_directory, test_subdir, dotest_options,
                                                  num_threads)
    timed_out = set(timed_out)
    num_tests = len(failed) + len(passed)

    # remove expected timeouts from failures
    expected_timeout = getExpectedTimeouts(dotest_options)
    for xtime in expected_timeout:
        if xtime in timed_out:
            timed_out.remove(xtime)
            failed.remove(xtime)

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
