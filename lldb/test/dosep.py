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

import asyncore
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
import test_results
import dotest_channels
import dotest_args


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
RESULTS_FORMATTER = None
RUNNER_PROCESS_ASYNC_MAP = None
RESULTS_LISTENER_CHANNEL = None

"""Contains an optional function pointer that can return the worker index
   for the given thread/process calling it.  Returns a 0-based index."""
GET_WORKER_INDEX = None


def setup_global_variables(
        lock, counter, total, name_len, options, worker_index_map):
    global output_lock, test_counter, total_tests, test_name_len
    global dotest_options
    output_lock = lock
    test_counter = counter
    total_tests = total
    test_name_len = name_len
    dotest_options = options

    if worker_index_map is not None:
        # We'll use the output lock for this to avoid sharing another lock.
        # This won't be used much.
        index_lock = lock

        def get_worker_index_use_pid():
            """Returns a 0-based, process-unique index for the worker."""
            pid = os.getpid()
            with index_lock:
                if pid not in worker_index_map:
                    worker_index_map[pid] = len(worker_index_map)
                return worker_index_map[pid]

        global GET_WORKER_INDEX
        GET_WORKER_INDEX = get_worker_index_use_pid


def report_test_failure(name, command, output):
    global output_lock
    with output_lock:
        if not (RESULTS_FORMATTER and RESULTS_FORMATTER.is_using_terminal()):
            print >> sys.stderr
            print >> sys.stderr, output
            print >> sys.stderr, "[%s FAILED]" % name
            print >> sys.stderr, "Command invoked: %s" % ' '.join(command)
        update_progress(name)


def report_test_pass(name, output):
    global output_lock, output_on_success
    with output_lock:
        if not (RESULTS_FORMATTER and RESULTS_FORMATTER.is_using_terminal()):
            if output_on_success:
                print >> sys.stderr
                print >> sys.stderr, output
                print >> sys.stderr, "[%s PASSED]" % name
        update_progress(name)


def update_progress(test_name=""):
    global output_lock, test_counter, total_tests, test_name_len
    with output_lock:
        counter_len = len(str(total_tests))
        if not (RESULTS_FORMATTER and RESULTS_FORMATTER.is_using_terminal()):
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
    """Run command with a timeout if possible.
    -s QUIT will create a coredump if they are enabled on your system
    """
    process = None
    if timeout_command and timeout != "0":
        command = [timeout_command, '-s', 'QUIT', timeout] + command

    if GET_WORKER_INDEX is not None:
        try:
            worker_index = GET_WORKER_INDEX()
            command.extend([
                "--event-add-entries", "worker_index={}:int".format(worker_index)])
        except:
            # Ctrl-C does bad things to multiprocessing.Manager.dict() lookup.
            pass

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

    # The inferior should now be entirely wrapped up.
    exit_status = process.returncode
    if exit_status is None:
        raise Exception(
            "no exit status available after the inferior dotest.py "
            "should have completed")

    if inferior_pid_events:
        inferior_pid_events.put_nowait(('destroyed', inferior_pid))

    passes, failures, unexpected_successes = parse_test_results(output)
    if exit_status == 0:
        # stdout does not have any useful information from 'dotest.py',
        # only stderr does.
        report_test_pass(name, output[1])
    else:
        # TODO need to differentiate a failing test from a run that
        # was broken out of by a SIGTERM/SIGKILL, reporting those as
        # an error.  If a signal-based completion, need to call that
        # an error.
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
        a_dotest_options, job_queue, result_queue, inferior_pid_events,
        worker_index_map):
    """Worker thread main loop when in multiprocessing mode.
    Takes one directory specification at a time and works on it."""

    # Shut off interrupt handling in the child process.
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    signal.signal(signal.SIGHUP, signal.SIG_IGN)

    # Setup the global state for the worker process.
    setup_global_variables(
        a_output_lock, a_test_counter, a_total_tests, a_test_name_len,
        a_dotest_options, worker_index_map)

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


def process_dir_worker_threading(job_queue, result_queue, inferior_pid_events):
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
    if not (RESULTS_FORMATTER and RESULTS_FORMATTER.is_using_terminal()):
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
    """Initializes global variables used in threading mode.
    @param num_threads specifies the number of workers used.
    @param test_work_items specifies all the work items
    that will be processed.
    """
    # Initialize the global state we'll use to communicate with the
    # rest of the flat module.
    global output_lock
    output_lock = threading.RLock()

    index_lock = threading.RLock()
    index_map = {}

    def get_worker_index_threading():
        """Returns a 0-based, thread-unique index for the worker thread."""
        thread_id = threading.current_thread().ident
        with index_lock:
            if thread_id not in index_map:
                index_map[thread_id] = len(index_map)
            return index_map[thread_id]


    global GET_WORKER_INDEX
    GET_WORKER_INDEX = get_worker_index_threading

    initialize_global_vars_common(num_threads, test_work_items)


def ctrl_c_loop(main_op_func, done_func, ctrl_c_handler):
    """Provides a main loop that is Ctrl-C protected.

    The main loop calls the main_op_func() repeatedly until done_func()
    returns true.  The ctrl_c_handler() method is called with a single
    int parameter that contains the number of times the ctrl_c has been
    hit (starting with 1).  The ctrl_c_handler() should mutate whatever
    it needs to have the done_func() return True as soon as it is desired
    to exit the loop.
    """
    done = False
    ctrl_c_count = 0

    while not done:
        try:
            # See if we're done.  Start with done check since it is
            # the first thing executed after a Ctrl-C handler in the
            # following loop.
            done = done_func()
            if not done:
                # Run the main op once.
                main_op_func()

        except KeyboardInterrupt:
            ctrl_c_count += 1
            ctrl_c_handler(ctrl_c_count)


def pump_workers_and_asyncore_map(workers, asyncore_map):
    """Prunes out completed workers and maintains the asyncore loop.

    The asyncore loop contains the optional socket listener
    and handlers.  When all workers are complete, this method
    takes care of stopping the listener.  It also runs the
    asyncore loop for the given async map for 10 iterations.

    @param workers the list of worker Thread/Process instances.

    @param asyncore_map the asyncore threading-aware map that
    indicates which channels are in use and still alive.
    """

    # Check on all the workers, removing them from the workers
    # list as they complete.
    dead_workers = []
    for worker in workers:
        # This non-blocking join call is what allows us
        # to still receive keyboard interrupts.
        worker.join(0.01)
        if not worker.is_alive():
            dead_workers.append(worker)
            # Clear out the completed workers
    for dead_worker in dead_workers:
        workers.remove(dead_worker)

    # If there are no more workers and there is a listener,
    # close the listener.
    global RESULTS_LISTENER_CHANNEL
    if len(workers) == 0 and RESULTS_LISTENER_CHANNEL is not None:
        RESULTS_LISTENER_CHANNEL.close()
        RESULTS_LISTENER_CHANNEL = None

    # Pump the asyncore map if it isn't empty.
    if len(asyncore_map) > 0:
        asyncore.loop(0.1, False, asyncore_map, 10)


def handle_ctrl_c(ctrl_c_count, job_queue, workers, inferior_pid_events,
                  stop_all_inferiors_func):
    """Performs the appropriate ctrl-c action for non-pool parallel test runners

    @param ctrl_c_count starting with 1, indicates the number of times ctrl-c
    has been intercepted.  The value is 1 on the first intercept, 2 on the
    second, etc.

    @param job_queue a Queue object that contains the work still outstanding
    (i.e. hasn't been assigned to a worker yet).

    @param workers list of Thread or Process workers.

    @param inferior_pid_events specifies a Queue of inferior process
    construction and destruction events.  Used to build the list of inferior
    processes that should be killed if we get that far.

    @param stop_all_inferiors_func a callable object that takes the
    workers and inferior_pid_events parameters (in that order) if a hard
    stop is to be used on the workers.
    """

    # Print out which Ctrl-C we're handling.
    key_name = [
        "first",
        "second",
        "third",
        "many"]

    if ctrl_c_count < len(key_name):
        name_index = ctrl_c_count - 1
    else:
        name_index = len(key_name) - 1
    message = "\nHandling {} KeyboardInterrupt".format(key_name[name_index])
    with output_lock:
        print message

    if ctrl_c_count == 1:
        # Remove all outstanding items from the work queue so we stop
        # doing any more new work.
        while not job_queue.empty():
            try:
                # Just drain it to stop more work from being started.
                job_queue.get_nowait()
            except Queue.Empty:
                pass
        with output_lock:
            print "Stopped more work from being started."
    elif ctrl_c_count == 2:
        # Try to stop all inferiors, even the ones currently doing work.
        stop_all_inferiors_func(workers, inferior_pid_events)
    else:
        with output_lock:
            print "All teardown activities kicked off, should finish soon."


def workers_and_async_done(workers, async_map):
    """Returns True if the workers list and asyncore channels are all done.

    @param workers list of workers (threads/processes).  These must adhere
    to the threading Thread or multiprocessing.Process interface.

    @param async_map the threading-aware asyncore channel map to check
    for live channels.

    @return False if the workers list exists and has any entries in it, or
    if the async_map exists and has any entries left in it; otherwise, True.
    """
    if workers is not None and len(workers) > 0:
        # We're not done if we still have workers left.
        return False
    if async_map is not None and len(async_map) > 0:
        return False
    # We're done.
    return True


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

    # Worker dictionary allows each worker to figure out its worker index.
    manager = multiprocessing.Manager()
    worker_index_map = manager.dict()

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
                  inferior_pid_events,
                  worker_index_map))
        worker.start()
        workers.append(worker)

    # Main loop: wait for all workers to finish and wait for
    # the socket handlers to wrap up.
    ctrl_c_loop(
        # Main operation of loop
        lambda: pump_workers_and_asyncore_map(
            workers, RUNNER_PROCESS_ASYNC_MAP),

        # Return True when we're done with the main loop.
        lambda: workers_and_async_done(workers, RUNNER_PROCESS_ASYNC_MAP),

        # Indicate what we do when we receive one or more Ctrl-Cs.
        lambda ctrl_c_count: handle_ctrl_c(
            ctrl_c_count, job_queue, workers, inferior_pid_events,
            kill_all_worker_processes))

    # Reap the test results.
    test_results = []
    while not result_queue.empty():
        test_results.append(result_queue.get(block=False))
    return test_results


def map_async_run_loop(future, channel_map, listener_channel):
    """Blocks until the Pool.map_async completes and the channel completes.

    @param future an AsyncResult instance from a Pool.map_async() call.

    @param channel_map the asyncore dispatch channel map that should be pumped.
    Optional: may be None.

    @param listener_channel the channel representing a listener that should be
    closed once the map_async results are available.

    @return the results from the async_result instance.
    """
    map_results = None

    done = False
    while not done:
        # Check if we need to reap the map results.
        if map_results is None:
            if future.ready():
                # Get the results.
                map_results = future.get()

                # Close the runner process listener channel if we have
                # one: no more connections will be incoming.
                if listener_channel is not None:
                    listener_channel.close()

        # Pump the asyncore loop if we have a listener socket.
        if channel_map is not None:
            asyncore.loop(0.01, False, channel_map, 10)

        # Figure out if we're done running.
        done = map_results is not None
        if channel_map is not None:
            # We have a runner process async map.  Check if it
            # is complete.
            if len(channel_map) > 0:
                # We still have an asyncore channel running.  Not done yet.
                done = False

    return map_results


def multiprocessing_test_runner_pool(num_threads, test_work_items):
    # Initialize our global state.
    initialize_global_vars_multiprocessing(num_threads, test_work_items)

    manager = multiprocessing.Manager()
    worker_index_map = manager.dict()

    pool = multiprocessing.Pool(
        num_threads,
        initializer=setup_global_variables,
        initargs=(output_lock, test_counter, total_tests, test_name_len,
                  dotest_options, worker_index_map))

    # Start the map operation (async mode).
    map_future = pool.map_async(
        process_dir_worker_multiprocessing_pool, test_work_items)
    return map_async_run_loop(
        map_future, RUNNER_PROCESS_ASYNC_MAP, RESULTS_LISTENER_CHANNEL)


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
            args=(job_queue,
                  result_queue,
                  inferior_pid_events))
        worker.start()
        workers.append(worker)

    # Main loop: wait for all workers to finish and wait for
    # the socket handlers to wrap up.
    ctrl_c_loop(
        # Main operation of loop
        lambda: pump_workers_and_asyncore_map(
            workers, RUNNER_PROCESS_ASYNC_MAP),

        # Return True when we're done with the main loop.
        lambda: workers_and_async_done(workers, RUNNER_PROCESS_ASYNC_MAP),

        # Indicate what we do when we receive one or more Ctrl-Cs.
        lambda ctrl_c_count: handle_ctrl_c(
            ctrl_c_count, job_queue, workers, inferior_pid_events,
            kill_all_worker_threads))

    # Reap the test results.
    test_results = []
    while not result_queue.empty():
        test_results.append(result_queue.get(block=False))
    return test_results


def threading_test_runner_pool(num_threads, test_work_items):
    # Initialize our global state.
    initialize_global_vars_threading(num_threads, test_work_items)

    pool = multiprocessing.pool.ThreadPool(num_threads)
    map_future = pool.map_async(
        process_dir_worker_threading_pool, test_work_items)

    return map_async_run_loop(
        map_future, RUNNER_PROCESS_ASYNC_MAP, RESULTS_LISTENER_CHANNEL)


def asyncore_run_loop(channel_map):
    try:
        asyncore.loop(None, False, channel_map)
    except:
        # Swallow it, we're seeing:
        #   error: (9, 'Bad file descriptor')
        # when the listener channel is closed.  Shouldn't be the case.
        pass


def inprocess_exec_test_runner(test_work_items):
    # Initialize our global state.
    initialize_global_vars_multiprocessing(1, test_work_items)

    # We're always worker index 0
    global GET_WORKER_INDEX
    GET_WORKER_INDEX = lambda: 0

    # Run the listener and related channel maps in a separate thread.
    # global RUNNER_PROCESS_ASYNC_MAP
    global RESULTS_LISTENER_CHANNEL
    if RESULTS_LISTENER_CHANNEL is not None:
        socket_thread = threading.Thread(
            target=lambda: asyncore_run_loop(RUNNER_PROCESS_ASYNC_MAP))
        socket_thread.start()

    # Do the work.
    test_results = map(process_dir_mapper_inprocess, test_work_items)

    # If we have a listener channel, shut it down here.
    if RESULTS_LISTENER_CHANNEL is not None:
        # Close down the channel.
        RESULTS_LISTENER_CHANNEL.close()
        RESULTS_LISTENER_CHANNEL = None

        # Wait for the listener and handlers to complete.
        socket_thread.join()

    return test_results

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
    # The async_map is important to keep all thread-related asyncore
    # channels distinct when we call asyncore.loop() later on.
    global RESULTS_LISTENER_CHANNEL, RUNNER_PROCESS_ASYNC_MAP
    RUNNER_PROCESS_ASYNC_MAP = {}

    # If we're outputting side-channel test results, create the socket
    # listener channel and tell the inferior to send results to the
    # port on which we'll be listening.
    if RESULTS_FORMATTER is not None:
        forwarding_func = RESULTS_FORMATTER.handle_event
        RESULTS_LISTENER_CHANNEL = (
            dotest_channels.UnpicklingForwardingListenerChannel(
                RUNNER_PROCESS_ASYNC_MAP, "localhost", 0, forwarding_func))
        dotest_argv.append("--results-port")
        dotest_argv.append(str(RESULTS_LISTENER_CHANNEL.address[1]))

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
        m = re.search(r'remote-(\w+)', platform_name)
        target = m.group(1)

    expected_timeout = set()

    if target.startswith("linux"):
        expected_timeout |= {
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
        # This does not properly support Ctrl-C.
        "threading-pool":
        (lambda work_items: threading_test_runner_pool(
            num_threads, work_items)),

        # serial uses the subprocess-based, single process
        # test runner.  This provides process isolation but
        # no concurrent test execution.
        "serial":
        inprocess_exec_test_runner
    }


def _remove_option(args, option_name, removal_count):
    """Removes option and related option arguments from args array.
    @param args the array of command line arguments (in/out)
    @param option_name the full command line representation of the
    option that will be removed (including '--' or '-').
    @param the count of elements to remove.  A value of 1 will remove
    just the found option, while 2 will remove the option and its first
    argument.
    """
    try:
        index = args.index(option_name)
        # Handle the exact match case.
        del args[index:index+removal_count]
        return
    except ValueError:
        # Thanks to argparse not handling options with known arguments
        # like other options parsing libraries (see
        # https://bugs.python.org/issue9334), we need to support the
        # --results-formatter-options={second-level-arguments} (note
        # the equal sign to fool the first-level arguments parser into
        # not treating the second-level arguments as first-level
        # options). We're certainly at risk of getting this wrong
        # since now we're forced into the business of trying to figure
        # out what is an argument (although I think this
        # implementation will suffice).
        regex_string = "^" + option_name + "="
        regex = re.compile(regex_string)
        for index in range(len(args)):
            match = regex.match(args[index])
            if match:
                del args[index]
                return
        print "failed to find regex '{}'".format(regex_string)

    # We didn't find the option but we should have.
    raise Exception("failed to find option '{}' in args '{}'".format(
        option_name, args))


def adjust_inferior_options(dotest_argv):
    """Adjusts the commandline args array for inferiors.

    This method adjusts the inferior dotest commandline options based
    on the parallel test runner's options.  Some of the inferior options
    will need to change to properly handle aggregation functionality.
    """
    global dotest_options

    # If we don't have a session directory, create one.
    if not dotest_options.s:
        # no session log directory, we need to add this to prevent
        # every dotest invocation from creating its own directory
        import datetime
        # The windows platforms don't like ':' in the pathname.
        timestamp_started = datetime.datetime.now().strftime("%F-%H_%M_%S")
        dotest_argv.append('-s')
        dotest_argv.append(timestamp_started)
        dotest_options.s = timestamp_started

    # Adjust inferior results formatter options - if the parallel
    # test runner is collecting into the user-specified test results,
    # we'll have inferiors spawn with the --results-port option and
    # strip the original test runner options.
    if dotest_options.results_file is not None:
        _remove_option(dotest_argv, "--results-file", 2)
    if dotest_options.results_port is not None:
        _remove_option(dotest_argv, "--results-port", 2)
    if dotest_options.results_formatter is not None:
        _remove_option(dotest_argv, "--results-formatter", 2)
    if dotest_options.results_formatter_options is not None:
        _remove_option(dotest_argv, "--results-formatter-options", 2)

    # Remove test runner name if present.
    if dotest_options.test_runner_name is not None:
        _remove_option(dotest_argv, "--test-runner-name", 2)


def main(print_details_on_success, num_threads, test_subdir,
         test_runner_name, results_formatter):
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

    @param results_formatter if specified, provides the TestResultsFormatter
    instance that will format and output test result data from the
    side-channel test results.  When specified, inferior dotest calls
    will send test results side-channel data over a socket to the parallel
    test runner, which will forward them on to results_formatter.
    """

    # Do not shut down on sighup.
    signal.signal(signal.SIGHUP, signal.SIG_IGN)

    dotest_argv = sys.argv[1:]

    global output_on_success, RESULTS_FORMATTER, output_lock
    output_on_success = print_details_on_success
    RESULTS_FORMATTER = results_formatter
    if RESULTS_FORMATTER is not None:
        RESULTS_FORMATTER.set_lock(output_lock)

    # We can't use sys.path[0] to determine the script directory
    # because it doesn't work under a debugger
    parser = dotest_args.create_parser()
    global dotest_options
    dotest_options = dotest_args.parse_args(parser, dotest_argv)

    adjust_inferior_options(dotest_argv)

    session_dir = os.path.join(os.getcwd(), dotest_options.s)

    # The root directory was specified on the command line
    test_directory = os.path.dirname(os.path.realpath(__file__))
    if test_subdir and len(test_subdir) > 0:
        test_subdir = os.path.join(test_directory, test_subdir)
    else:
        test_subdir = test_directory

    # clean core files in test tree from previous runs (Linux)
    cores = find('core.*', test_subdir)
    for core in cores:
        os.unlink(core)

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
