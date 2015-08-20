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

To collect core files for timed out tests,
do the following before running dosep.py

OSX
ulimit -c unlimited
sudo sysctl -w kern.corefile=core.%P

Linux:
ulimit -c unlimited
echo core.%p | sudo tee /proc/sys/kernel/core_pattern
"""

import multiprocessing
import os
import fnmatch
import platform
import re
import dotest_args
import shlex
import subprocess
import sys

from optparse import OptionParser


def get_timeout_command():
    """Search for a suitable timeout command."""
    if not sys.platform.startswith("win32"):
        try:
            subprocess.call("timeout", stderr=subprocess.PIPE)
            return "timeout"
        except OSError:
            pass
    try:
        subprocess.call("gtimeout", stderr=subprocess.PIPE)
        return "gtimeout"
    except OSError:
        pass
    return None

timeout_command = get_timeout_command()

# Status codes for running command with timeout.
eTimedOut, ePassed, eFailed = 124, 0, 1

output_lock = None
test_counter = None
total_tests = None
test_name_len = None
dotest_options = None
output_on_success = False


def setup_global_variables(lock, counter, total, name_len, options):
    global output_lock, test_counter, total_tests, test_name_len
    global dotest_options
    output_lock = lock
    test_counter = counter
    total_tests = total
    test_name_len = name_len
    dotest_options = options


def report_test_failure(name, command, output):
    global output_lock
    with output_lock:
        print >> sys.stderr
        print >> sys.stderr, output
        print >> sys.stderr, "[%s FAILED]" % name
        print >> sys.stderr, "Command invoked: %s" % ' '.join(command)
        update_progress(name)


def report_test_pass(name, output):
    global output_lock, output_on_success
    with output_lock:
        if output_on_success:
            print >> sys.stderr
            print >> sys.stderr, output
            print >> sys.stderr, "[%s PASSED]" % name
        update_progress(name)


def update_progress(test_name=""):
    global output_lock, test_counter, total_tests, test_name_len
    with output_lock:
        counter_len = len(str(total_tests))
        sys.stderr.write(
            "\r%*d out of %d test suites processed - %-*s" %
            (counter_len, test_counter.value, total_tests,
             test_name_len.value, test_name))
        if len(test_name) > test_name_len.value:
            test_name_len.value = len(test_name)
        test_counter.value += 1
        sys.stdout.flush()
        sys.stderr.flush()


def parse_test_results(output):
    passes = 0
    failures = 0
    unexpected_successes = 0
    for result in output:
        pass_count = re.search("^RESULT:.*([0-9]+) passes",
                               result, re.MULTILINE)
        fail_count = re.search("^RESULT:.*([0-9]+) failures",
                               result, re.MULTILINE)
        error_count = re.search("^RESULT:.*([0-9]+) errors",
                                result, re.MULTILINE)
        unexpected_success_count = re.search("^RESULT:.*([0-9]+) unexpected successes",
                                             result, re.MULTILINE)
        this_fail_count = 0
        this_error_count = 0
        if pass_count is not None:
            passes = passes + int(pass_count.group(1))
        if fail_count is not None:
            failures = failures + int(fail_count.group(1))
        if unexpected_success_count is not None:
            unexpected_successes = unexpected_successes + int(unexpected_success_count.group(1))
        if error_count is not None:
            failures = failures + int(error_count.group(1))
        pass
    return passes, failures, unexpected_successes


def call_with_timeout(command, timeout, name):
    """Run command with a timeout if possible."""
    """-s QUIT will create a coredump if they are enabled on your system"""
    process = None
    if timeout_command and timeout != "0":
        command = [timeout_command, '-s', 'QUIT', timeout] + command
    # Specifying a value for close_fds is unsupported on Windows when using
    # subprocess.PIPE
    if os.name != "nt":
        process = subprocess.Popen(command,
                                   stdin=subprocess.PIPE,
                                   stdout=subprocess.PIPE,
                                   stderr=subprocess.PIPE,
                                   close_fds=True)
    else:
        process = subprocess.Popen(command,
                                   stdin=subprocess.PIPE,
                                   stdout=subprocess.PIPE,
                                   stderr=subprocess.PIPE)
    output = process.communicate()
    exit_status = process.returncode
    passes, failures, unexpected_successes = parse_test_results(output)
    if exit_status == 0:
        # stdout does not have any useful information from 'dotest.py',
        # only stderr does.
        report_test_pass(name, output[1])
    else:
        report_test_failure(name, command, output[1])
    return name, exit_status, passes, failures, unexpected_successes


def process_dir(root, files, test_root, dotest_argv):
    """Examine a directory for tests, and invoke any found within it."""
    results = []
    for name in files:
        script_file = os.path.join(test_root, "dotest.py")
        command = ([sys.executable, script_file] +
                   dotest_argv +
                   ["-p", name, root])

        timeout_name = os.path.basename(os.path.splitext(name)[0]).upper()

        timeout = (os.getenv("LLDB_%s_TIMEOUT" % timeout_name) or
                   getDefaultTimeout(dotest_options.lldb_platform_name))

        results.append(call_with_timeout(command, timeout, name))

    # result = (name, status, passes, failures, unexpected_successes)
    timed_out = [name for name, status, _, _, _ in results
                 if status == eTimedOut]
    passed = [name for name, status, _, _, _ in results
              if status == ePassed]
    failed = [name for name, status, _, _, _ in results
              if status != ePassed]
    unexpected_passes = [name for name, _, _, _, unexpected_successes in results
                         if unexpected_successes > 0]
    
    pass_count = sum([result[2] for result in results])
    fail_count = sum([result[3] for result in results])

    return (timed_out, passed, failed, unexpected_passes, pass_count, fail_count)

in_q = None
out_q = None


def process_dir_worker(arg_tuple):
    """Worker thread main loop when in multithreaded mode.
    Takes one directory specification at a time and works on it."""
    return process_dir(*arg_tuple)


def walk_and_invoke(test_directory, test_subdir, dotest_argv, num_threads):
    """Look for matched files and invoke test driver on each one.
    In single-threaded mode, each test driver is invoked directly.
    In multi-threaded mode, submit each test driver to a worker
    queue, and then wait for all to complete.

    test_directory - lldb/test/ directory
    test_subdir - lldb/test/ or a subfolder with the tests we're interested in
                  running
    """

    # Collect the test files that we'll run.
    test_work_items = []
    for root, dirs, files in os.walk(test_subdir, topdown=False):
        def is_test(name):
            # Not interested in symbolically linked files.
            if os.path.islink(os.path.join(root, name)):
                return False
            # Only interested in test files with the "Test*.py" naming pattern.
            return name.startswith("Test") and name.endswith(".py")

        tests = filter(is_test, files)
        test_work_items.append((root, tests, test_directory, dotest_argv))

    global output_lock, test_counter, total_tests, test_name_len
    output_lock = multiprocessing.RLock()
    # item = (root, tests, test_directory, dotest_argv)
    total_tests = sum([len(item[1]) for item in test_work_items])
    test_counter = multiprocessing.Value('i', 0)
    test_name_len = multiprocessing.Value('i', 0)
    print >> sys.stderr, "Testing: %d test suites, %d thread%s" % (
        total_tests, num_threads, (num_threads > 1) * "s")
    update_progress()

    # Run the items, either in a pool (for multicore speedup) or
    # calling each individually.
    if num_threads > 1:
        pool = multiprocessing.Pool(
            num_threads,
            initializer=setup_global_variables,
            initargs=(output_lock, test_counter, total_tests, test_name_len,
                      dotest_options))
        test_results = pool.map(process_dir_worker, test_work_items)
    else:
        test_results = map(process_dir_worker, test_work_items)

    # result = (timed_out, failed, passed, unexpected_successes, fail_count, pass_count)
    timed_out = sum([result[0] for result in test_results], [])
    passed = sum([result[1] for result in test_results], [])
    failed = sum([result[2] for result in test_results], [])
    unexpected_successes = sum([result[3] for result in test_results], [])
    pass_count = sum([result[4] for result in test_results])
    fail_count = sum([result[5] for result in test_results])

    return (timed_out, passed, failed, unexpected_successes, pass_count, fail_count)


def getExpectedTimeouts(platform_name):
    # returns a set of test filenames that might timeout
    # are we running against a remote target?
    host = sys.platform
    if platform_name is None:
        target = sys.platform
        remote = False
    else:
        m = re.search('remote-(\w+)', platform_name)
        target = m.group(1)
        remote = True

    expected_timeout = set()

    if target.startswith("linux"):
        expected_timeout |= {
            "TestAttachDenied.py",
            "TestAttachResume.py",
            "TestProcessAttach.py",
            "TestConnectRemote.py",
            "TestCreateAfterAttach.py",
            "TestEvents.py",
            "TestExitDuringStep.py",

            # Times out in ~10% of the times on the build bot
            "TestHelloWorld.py",
            "TestMultithreaded.py",
            "TestRegisters.py",  # ~12/600 dosep runs (build 3120-3122)
            "TestThreadStepOut.py",
            "TestChangeProcessGroup.py",
        }
    elif target.startswith("android"):
        expected_timeout |= {
            "TestExitDuringStep.py",
            "TestHelloWorld.py",
        }
        if host.startswith("win32"):
            expected_timeout |= {
                "TestEvents.py",
                "TestThreadStates.py",
            }
    elif target.startswith("freebsd"):
        expected_timeout |= {
            "TestBreakpointConditions.py",
            "TestChangeProcessGroup.py",
            "TestValueObjectRecursion.py",
            "TestWatchpointConditionAPI.py",
        }
    elif target.startswith("darwin"):
        expected_timeout |= {
            # times out on MBP Retina, Mid 2012
            "TestThreadSpecificBreakpoint.py",
            "TestExitDuringStep.py",
            "TestIntegerTypesExpr.py",
        }
    return expected_timeout


def getDefaultTimeout(platform_name):
    if os.getenv("LLDB_TEST_TIMEOUT"):
        return os.getenv("LLDB_TEST_TIMEOUT")

    if platform_name is None:
        platform_name = sys.platform

    if platform_name.startswith("remote-"):
        return "10m"
    else:
        return "4m"


def touch(fname, times=None):
    if os.path.exists(fname):
        os.utime(fname, times)


def find(pattern, path):
    result = []
    for root, dirs, files in os.walk(path):
        for name in files:
            if fnmatch.fnmatch(name, pattern):
                result.append(os.path.join(root, name))
    return result


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
    parser.add_option(
        '-o', '--options',
        type='string', action='store',
        dest='dotest_options',
        help="""The options passed to 'dotest.py' if specified.""")

    parser.add_option(
        '-s', '--output-on-success',
        action='store_true',
        dest='output_on_success',
        default=False,
        help="""Print full output of 'dotest.py' even when it succeeds.""")

    parser.add_option(
        '-t', '--threads',
        type='int',
        dest='num_threads',
        help="""The number of threads to use when running tests separately.""")

    opts, args = parser.parse_args()
    dotest_option_string = opts.dotest_options

    is_posix = (os.name == "posix")
    dotest_argv = (shlex.split(dotest_option_string, posix=is_posix)
                   if dotest_option_string
                   else [])

    parser = dotest_args.create_parser()
    global dotest_options
    global output_on_success
    output_on_success = opts.output_on_success
    dotest_options = dotest_args.parse_args(parser, dotest_argv)

    if not dotest_options.s:
        # no session log directory, we need to add this to prevent
        # every dotest invocation from creating its own directory
        import datetime
        # The windows platforms don't like ':' in the pathname.
        timestamp_started = datetime.datetime.now().strftime("%F-%H_%M_%S")
        dotest_argv.append('-s')
        dotest_argv.append(timestamp_started)
        dotest_options.s = timestamp_started

    session_dir = os.path.join(os.getcwd(), dotest_options.s)

    # The root directory was specified on the command line
    if len(args) == 0:
        test_subdir = test_directory
    else:
        test_subdir = os.path.join(test_directory, args[0])

    # clean core files in test tree from previous runs (Linux)
    cores = find('core.*', test_subdir)
    for core in cores:
        os.unlink(core)

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
    (timed_out, passed, failed, unexpected_successes, pass_count, fail_count) = walk_and_invoke(
        test_directory, test_subdir, dotest_argv, num_threads)

    timed_out = set(timed_out)
    num_test_files = len(passed) + len(failed)
    num_test_cases = pass_count + fail_count

    # move core files into session dir
    cores = find('core.*', test_subdir)
    for core in cores:
        dst = core.replace(test_directory, "")[1:]
        dst = dst.replace(os.path.sep, "-")
        os.rename(core, os.path.join(session_dir, dst))

    # remove expected timeouts from failures
    expected_timeout = getExpectedTimeouts(dotest_options.lldb_platform_name)
    for xtime in expected_timeout:
        if xtime in timed_out:
            timed_out.remove(xtime)
            failed.remove(xtime)
            result = "ExpectedTimeout"
        elif xtime in passed:
            result = "UnexpectedCompletion"
        else:
            result = None  # failed

        if result:
            test_name = os.path.splitext(xtime)[0]
            touch(os.path.join(session_dir, "{}-{}".format(result, test_name)))

    print
    sys.stdout.write("Ran %d test suites" % num_test_files)
    if num_test_files > 0:
        sys.stdout.write(" (%d failed) (%f%%)" % (
            len(failed), 100.0 * len(failed) / num_test_files))
    print
    sys.stdout.write("Ran %d test cases" % num_test_cases)
    if num_test_cases > 0:
        sys.stdout.write(" (%d failed) (%f%%)" % (
            fail_count, 100.0 * fail_count / num_test_cases))
    print
    exit_code = 0

    if len(failed) > 0:
        failed.sort()
        print "Failing Tests (%d)" % len(failed)
        for f in failed:
            print "%s: LLDB (suite) :: %s (%s)" % (
                "TIMEOUT" if f in timed_out else "FAIL", f, system_info
            )
        exit_code = 1

    if len(unexpected_successes) > 0:
        unexpected_successes.sort()
        print "\nUnexpected Successes (%d)" % len(unexpected_successes)
        for u in unexpected_successes:
            print "UNEXPECTED SUCCESS: LLDB (suite) :: %s (%s)" % (u, system_info)

    sys.exit(exit_code)

if __name__ == '__main__':
    main()
