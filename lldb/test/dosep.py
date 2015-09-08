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

import fnmatch
import multiprocessing
import multiprocessing.pool
import os
import platform
import Queue
import re
import signal
import subprocess
import sys
import threading

import dotest_args

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


def call_with_timeout(command, timeout, name, inferior_pid_events):
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
    inferior_pid = process.pid
    if inferior_pid_events:
        inferior_pid_events.put_nowait(('created', inferior_pid))
    output = process.communicate()
    exit_status = process.returncode
    if inferior_pid_events:
        inferior_pid_events.put_nowait(('destroyed', inferior_pid))

    passes, failures, unexpected_successes = parse_test_results(output)
    if exit_status == 0:
        # stdout does not have any useful information from 'dotest.py',
        # only stderr does.
        report_test_pass(name, output[1])
    else:
        report_test_failure(name, command, output[1])
    return name, exit_status, passes, failures, unexpected_successes


def process_dir(root, files, test_root, dotest_argv, inferior_pid_events):
    """Examine a directory for tests, and invoke any found within it."""
    results = []
    for name in files:
        script_file = os.path.join(test_root, "dotest.py")
        command = ([sys.executable, script_file] +
                   dotest_argv +
                   ["--inferior", "-p", name, root])

        timeout_name = os.path.basename(os.path.splitext(name)[0]).upper()

        timeout = (os.getenv("LLDB_%s_TIMEOUT" % timeout_name) or
                   getDefaultTimeout(dotest_options.lldb_platform_name))

        results.append(call_with_timeout(
            command, timeout, name, inferior_pid_events))

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


def process_dir_worker_multiprocessing(
        a_output_lock, a_test_counter, a_total_tests, a_test_name_len,
        a_dotest_options, job_queue, result_queue, inferior_pid_events):
    """Worker thread main loop when in multiprocessing mode.
    Takes one directory specification at a time and works on it."""

    # Shut off interrupt handling in the child process.
    signal.signal(signal.SIGINT, signal.SIG_IGN)

    # Setup the global state for the worker process.
    setup_global_variables(
        a_output_lock, a_test_counter, a_total_tests, a_test_name_len,
        a_dotest_options)

    # Keep grabbing entries from the queue until done.
    while not job_queue.empty():
        try:
            job = job_queue.get(block=False)
            result = process_dir(job[0], job[1], job[2], job[3],
                                 inferior_pid_events)
            result_queue.put(result)
        except Queue.Empty:
            # Fine, we're done.
            pass


def process_dir_worker_multiprocessing_pool(args):
    return process_dir(*args)


def process_dir_worker_threading(
        a_test_counter, a_total_tests, a_test_name_len,
        a_dotest_options, job_queue, result_queue, inferior_pid_events):
    """Worker thread main loop when in threading mode.

    This one supports the hand-rolled pooling support.

    Takes one directory specification at a time and works on it."""

    # Keep grabbing entries from the queue until done.
    while not job_queue.empty():
        try:
            job = job_queue.get(block=False)
            result = process_dir(job[0], job[1], job[2], job[3],
                                 inferior_pid_events)
            result_queue.put(result)
        except Queue.Empty:
            # Fine, we're done.
            pass


def process_dir_worker_threading_pool(args):
    return process_dir(*args)


def process_dir_mapper_inprocess(args):
    """Map adapter for running the subprocess-based, non-threaded test runner.

    @param args the process work item tuple
    @return the test result tuple
    """
    return process_dir(*args)


def collect_active_pids_from_pid_events(event_queue):
    """
    Returns the set of what should be active inferior pids based on
    the event stream.

    @param event_queue a multiprocessing.Queue containing events of the
    form:
         ('created', pid)
         ('destroyed', pid)

    @return set of inferior dotest.py pids activated but never completed.
    """
    active_pid_set = set()
    while not event_queue.empty():
        pid_event = event_queue.get_nowait()
        if pid_event[0] == 'created':
            active_pid_set.add(pid_event[1])
        elif pid_event[0] == 'destroyed':
            active_pid_set.remove(pid_event[1])
    return active_pid_set


def kill_all_worker_processes(workers, inferior_pid_events):
    """
    Kills all specified worker processes and their process tree.

    @param workers a list of multiprocess.Process worker objects.
    @param inferior_pid_events a multiprocess.Queue that contains
    all inferior create and destroy events.  Used to construct
    the list of child pids still outstanding that need to be killed.
    """
    for worker in workers:
        worker.terminate()
        worker.join()

    # Add all the child test pids created.
    active_pid_set = collect_active_pids_from_pid_events(
        inferior_pid_events)
    for inferior_pid in active_pid_set:
        print "killing inferior pid {}".format(inferior_pid)
        os.kill(inferior_pid, signal.SIGKILL)


def kill_all_worker_threads(workers, inferior_pid_events):
    """
    Kills all specified worker threads and their process tree.

    @param workers a list of multiprocess.Process worker objects.
    @param inferior_pid_events a multiprocess.Queue that contains
    all inferior create and destroy events.  Used to construct
    the list of child pids still outstanding that need to be killed.
    """

    # Add all the child test pids created.
    active_pid_set = collect_active_pids_from_pid_events(
        inferior_pid_events)
    for inferior_pid in active_pid_set:
        print "killing inferior pid {}".format(inferior_pid)
        os.kill(inferior_pid, signal.SIGKILL)

    # We don't have a way to nuke the threads.  However, since we killed
    # all the inferiors, and we drained the job queue, this will be
    # good enough.  Wait cleanly for each worker thread to wrap up.
    for worker in workers:
        worker.join()


def find_test_files_in_dir_tree(dir_root, found_func):
    """Calls found_func for all the test files in the given dir hierarchy.

    @param dir_root the path to the directory to start scanning
    for test files.  All files in this directory and all its children
    directory trees will be searched.

    @param found_func a callable object that will be passed
    the parent directory (relative to dir_root) and the list of
    test files from within that directory.
    """
    for root, _, files in os.walk(dir_root, topdown=False):
        def is_test_filename(test_dir, base_filename):
            """Returns True if the given filename matches the test name format.

            @param test_dir the directory to check.  Should be absolute or
            relative to current working directory.

            @param base_filename the base name of the filename to check for a
            dherence to the python test case filename format.

            @return True if name matches the python test case filename format.
            """
            # Not interested in symbolically linked files.
            if os.path.islink(os.path.join(test_dir, base_filename)):
                return False
            # Only interested in test files with the "Test*.py" naming pattern.
            return (base_filename.startswith("Test") and
                    base_filename.endswith(".py"))

        tests = [filename for filename in files
                 if is_test_filename(root, filename)]
        if tests:
            found_func(root, tests)


def initialize_global_vars_common(num_threads, test_work_items):
    global total_tests, test_counter, test_name_len
    total_tests = sum([len(item[1]) for item in test_work_items])
    test_counter = multiprocessing.Value('i', 0)
    test_name_len = multiprocessing.Value('i', 0)
    print >> sys.stderr, "Testing: %d test suites, %d thread%s" % (
        total_tests, num_threads, (num_threads > 1) * "s")
    update_progress()


def initialize_global_vars_multiprocessing(num_threads, test_work_items):
    # Initialize the global state we'll use to communicate with the
    # rest of the flat module.
    global output_lock
    output_lock = multiprocessing.RLock()
    initialize_global_vars_common(num_threads, test_work_items)


def initialize_global_vars_threading(num_threads, test_work_items):
    # Initialize the global state we'll use to communicate with the
    # rest of the flat module.
    global output_lock
    output_lock = threading.RLock()
    initialize_global_vars_common(num_threads, test_work_items)


def multiprocessing_test_runner(num_threads, test_work_items):
    """Provides hand-wrapped pooling test runner adapter with Ctrl-C support.

    This concurrent test runner is based on the multiprocessing
    library, and rolls its own worker pooling strategy so it
    can handle Ctrl-C properly.

    This test runner is known to have an issue running on
    Windows platforms.

    @param num_threads the number of worker processes to use.

    @param test_work_items the iterable of test work item tuples
    to run.
    """

    # Initialize our global state.
    initialize_global_vars_multiprocessing(num_threads, test_work_items)

    # Create jobs.
    job_queue = multiprocessing.Queue(len(test_work_items))
    for test_work_item in test_work_items:
        job_queue.put(test_work_item)

    result_queue = multiprocessing.Queue(len(test_work_items))

    # Create queues for started child pids.  Terminating
    # the multiprocess processes does not terminate the
    # child processes they spawn.  We can remove this tracking
    # if/when we move to having the multiprocess process directly
    # perform the test logic.  The Queue size needs to be able to
    # hold 2 * (num inferior dotest.py processes started) entries.
    inferior_pid_events = multiprocessing.Queue(4096)

    # Create workers.  We don't use multiprocessing.Pool due to
    # challenges with handling ^C keyboard interrupts.
    workers = []
    for _ in range(num_threads):
        worker = multiprocessing.Process(
            target=process_dir_worker_multiprocessing,
            args=(output_lock,
                  test_counter,
                  total_tests,
                  test_name_len,
                  dotest_options,
                  job_queue,
                  result_queue,
                  inferior_pid_events))
        worker.start()
        workers.append(worker)

    # Wait for all workers to finish, handling ^C as needed.
    try:
        for worker in workers:
            worker.join()
    except KeyboardInterrupt:
        # First try to drain the queue of work and let the
        # running tests complete.
        while not job_queue.empty():
            try:
                # Just drain it to stop more work from being started.
                job_queue.get_nowait()
            except Queue.Empty:
                pass

        print ('\nFirst KeyboardInterrupt received, stopping '
               'future work.  Press again to hard-stop existing tests.')
        try:
            for worker in workers:
                worker.join()
        except KeyboardInterrupt:
            print ('\nSecond KeyboardInterrupt received, killing '
                   'all worker process trees.')
            kill_all_worker_processes(workers, inferior_pid_events)

    test_results = []
    while not result_queue.empty():
        test_results.append(result_queue.get(block=False))
    return test_results


def multiprocessing_test_runner_pool(num_threads, test_work_items):
    # Initialize our global state.
    initialize_global_vars_multiprocessing(num_threads, test_work_items)

    pool = multiprocessing.Pool(
        num_threads,
        initializer=setup_global_variables,
        initargs=(output_lock, test_counter, total_tests, test_name_len,
                  dotest_options))
    return pool.map(process_dir_worker_multiprocessing_pool, test_work_items)


def threading_test_runner(num_threads, test_work_items):
    """Provides hand-wrapped pooling threading-based test runner adapter
    with Ctrl-C support.

    This concurrent test runner is based on the threading
    library, and rolls its own worker pooling strategy so it
    can handle Ctrl-C properly.

    @param num_threads the number of worker processes to use.

    @param test_work_items the iterable of test work item tuples
    to run.
    """

    # Initialize our global state.
    initialize_global_vars_threading(num_threads, test_work_items)

    # Create jobs.
    job_queue = Queue.Queue()
    for test_work_item in test_work_items:
        job_queue.put(test_work_item)

    result_queue = Queue.Queue()

    # Create queues for started child pids.  Terminating
    # the threading threads does not terminate the
    # child processes they spawn.
    inferior_pid_events = Queue.Queue()

    # Create workers. We don't use multiprocessing.pool.ThreadedPool
    # due to challenges with handling ^C keyboard interrupts.
    workers = []
    for _ in range(num_threads):
        worker = threading.Thread(
            target=process_dir_worker_threading,
            args=(test_counter,
                  total_tests,
                  test_name_len,
                  dotest_options,
                  job_queue,
                  result_queue,
                  inferior_pid_events))
        worker.start()
        workers.append(worker)

    # Wait for all workers to finish, handling ^C as needed.
    try:
        # We do some trickery here to ensure we can catch keyboard
        # interrupts.
        while len(workers) > 0:
            # Make a pass throug the workers, checking for who is done.
            dead_workers = []
            for worker in workers:
                # This non-blocking join call is what allows us
                # to still receive keyboard interrupts.
                worker.join(0.01)
                if not worker.isAlive():
                    dead_workers.append(worker)
            # Clear out the completed workers
            for dead_worker in dead_workers:
                workers.remove(dead_worker)

    except KeyboardInterrupt:
        # First try to drain the queue of work and let the
        # running tests complete.
        while not job_queue.empty():
            try:
                # Just drain it to stop more work from being started.
                job_queue.get_nowait()
            except Queue.Empty:
                pass

        print ('\nFirst KeyboardInterrupt received, stopping '
               'future work.  Press again to hard-stop existing tests.')
        try:
            for worker in workers:
                worker.join()
        except KeyboardInterrupt:
            print ('\nSecond KeyboardInterrupt received, killing '
                   'all worker process trees.')
            kill_all_worker_threads(workers, inferior_pid_events)

    test_results = []
    while not result_queue.empty():
        test_results.append(result_queue.get(block=False))
    return test_results


def threading_test_runner_pool(num_threads, test_work_items):
    # Initialize our global state.
    initialize_global_vars_threading(num_threads, test_work_items)

    pool = multiprocessing.pool.ThreadPool(
        num_threads
        # initializer=setup_global_variables,
        # initargs=(output_lock, test_counter, total_tests, test_name_len,
        #           dotest_options)
    )
    return pool.map(process_dir_worker_threading_pool, test_work_items)


def inprocess_exec_test_runner(test_work_items):
    # Initialize our global state.
    initialize_global_vars_multiprocessing(1, test_work_items)
    return map(process_dir_mapper_inprocess, test_work_items)


def walk_and_invoke(test_directory, test_subdir, dotest_argv,
                    test_runner_func):
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
    find_test_files_in_dir_tree(
        test_subdir, lambda testdir, test_files: test_work_items.append([
            test_subdir, test_files, test_directory, dotest_argv, None]))

    # Convert test work items into test results using whatever
    # was provided as the test run function.
    test_results = test_runner_func(test_work_items)

    # Summarize the results and return to caller.
    timed_out = sum([result[0] for result in test_results], [])
    passed = sum([result[1] for result in test_results], [])
    failed = sum([result[2] for result in test_results], [])
    unexpected_successes = sum([result[3] for result in test_results], [])
    pass_count = sum([result[4] for result in test_results])
    fail_count = sum([result[5] for result in test_results])

    return (timed_out, passed, failed, unexpected_successes, pass_count,
            fail_count)


def getExpectedTimeouts(platform_name):
    # returns a set of test filenames that might timeout
    # are we running against a remote target?
    host = sys.platform
    if platform_name is None:
        target = sys.platform
    else:
        m = re.search('remote-(\w+)', platform_name)
        target = m.group(1)

    expected_timeout = set()

    if target.startswith("linux"):
        expected_timeout |= {
            "TestAttachDenied.py",
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


def get_test_runner_strategies(num_threads):
    """Returns the test runner strategies by name in a dictionary.

    @param num_threads specifies the number of threads/processes
    that will be used for concurrent test runners.

    @return dictionary with key as test runner strategy name and
    value set to a callable object that takes the test work item
    and returns a test result tuple.
    """
    return {
        # multiprocessing supports ctrl-c and does not use
        # multiprocessing.Pool.
        "multiprocessing":
        (lambda work_items: multiprocessing_test_runner(
            num_threads, work_items)),

        # multiprocessing-pool uses multiprocessing.Pool but
        # does not support Ctrl-C.
        "multiprocessing-pool":
        (lambda work_items: multiprocessing_test_runner_pool(
            num_threads, work_items)),

        # threading uses a hand-rolled worker pool much
        # like multiprocessing, but instead uses in-process
        # worker threads.  This one supports Ctrl-C.
        "threading":
        (lambda work_items: threading_test_runner(num_threads, work_items)),

        # threading-pool uses threading for the workers (in-process)
        # and uses the multiprocessing.pool thread-enabled pool.
        "threading-pool":
        (lambda work_items: threading_test_runner_pool(
            num_threads, work_items)),

        # serial uses the subprocess-based, single process
        # test runner.  This provides process isolation but
        # no concurrent test running.
        "serial":
        inprocess_exec_test_runner
    }


def main(print_details_on_success, num_threads, test_subdir,
         test_runner_name):
    """Run dotest.py in inferior mode in parallel.

    @param print_details_on_success the parsed value of the output-on-success
    command line argument.  When True, details of a successful dotest inferior
    are printed even when everything succeeds.  The normal behavior is to
    not print any details when all the inferior tests pass.

    @param num_threads the parsed value of the num-threads command line
    argument.

    @param test_subdir optionally specifies a subdir to limit testing
    within.  May be None if the entire test tree is to be used.  This subdir
    is assumed to be relative to the lldb/test root of the test hierarchy.

    @param test_runner_name if specified, contains the test runner
    name which selects the strategy used to run the isolated and
    optionally concurrent test runner. Specify None to allow the
    system to choose the most appropriate test runner given desired
    thread count and OS type.

    """

    dotest_argv = sys.argv[1:]

    global output_on_success
    output_on_success = print_details_on_success

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
    parser = dotest_args.create_parser()
    global dotest_options
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
    if test_subdir and len(test_subdir) > 0:
        test_subdir = os.path.join(test_directory, test_subdir)
    else:
        test_subdir = test_directory

    # clean core files in test tree from previous runs (Linux)
    cores = find('core.*', test_subdir)
    for core in cores:
        os.unlink(core)

    if not num_threads:
        num_threads_str = os.environ.get("LLDB_TEST_THREADS")
        if num_threads_str:
            num_threads = int(num_threads_str)
        else:
            num_threads = multiprocessing.cpu_count()
    if num_threads < 1:
        num_threads = 1

    system_info = " ".join(platform.uname())

    # Figure out which testrunner strategy we'll use.
    runner_strategies_by_name = get_test_runner_strategies(num_threads)

    # If the user didn't specify a test runner strategy, determine
    # the default now based on number of threads and OS type.
    if not test_runner_name:
        if num_threads == 1:
            # Use the serial runner.
            test_runner_name = "serial"
        elif os.name == "nt":
            # Currently the multiprocessing test runner with ctrl-c
            # support isn't running correctly on nt.  Use the pool
            # support without ctrl-c.
            test_runner_name = "multiprocessing-pool"
        else:
            # For everyone else, use the ctrl-c-enabled
            # multiprocessing support.
            test_runner_name = "multiprocessing"

    if test_runner_name not in runner_strategies_by_name:
        raise Exception("specified testrunner name '{}' unknown. "
               "Valid choices: {}".format(
                   test_runner_name,
                   runner_strategies_by_name.keys()))
    test_runner_func = runner_strategies_by_name[test_runner_name]

    summary_results = walk_and_invoke(
        test_directory, test_subdir, dotest_argv, test_runner_func)

    (timed_out, passed, failed, unexpected_successes, pass_count,
     fail_count) = summary_results

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
    sys.stderr.write(
        "error: dosep.py no longer supports being called directly. "
        "Please call dotest.py directly.  The dosep.py-specific arguments "
        "have been added under the Parallel processing arguments.\n")
    sys.exit(128)
