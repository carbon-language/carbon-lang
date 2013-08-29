import os
import threading
import time
import traceback

try:
    import win32api
except ImportError:
    win32api = None

import lit.Test

###
# Test Execution Implementation

class TestProvider(object):
    def __init__(self, tests):
        self.iter = iter(range(len(tests)))
        self.lock = threading.Lock()
        self.canceled = False

    def cancel(self):
        self.lock.acquire()
        self.canceled = True
        self.lock.release()

    def get(self):
        # Check if we are cancelled.
        self.lock.acquire()
        if self.canceled:
          self.lock.release()
          return None

        # Otherwise take the next test.
        for item in self.iter:
            break
        else:
            item = None
        self.lock.release()
        return item

class Tester(object):
    def __init__(self, run_instance, provider, consumer):
        self.run_instance = run_instance
        self.provider = provider
        self.consumer = consumer

    def run(self):
        while 1:
            item = self.provider.get()
            if item is None:
                break
            self.run_test(item)
        self.consumer.task_finished()

    def run_test(self, test_index):
        test = self.run_instance.tests[test_index]
        try:
            self.run_instance.execute_test(test)
        except KeyboardInterrupt:
            # This is a sad hack. Unfortunately subprocess goes
            # bonkers with ctrl-c and we start forking merrily.
            print('\nCtrl-C detected, goodbye.')
            os.kill(0,9)
        self.consumer.update(test_index, test)

class ThreadResultsConsumer(object):
    def __init__(self, display):
        self.display = display
        self.lock = threading.Lock()

    def update(self, test_index, test):
        self.lock.acquire()
        try:
            self.display.update(test)
        finally:
            self.lock.release()

    def task_finished(self):
        pass

    def handle_results(self):
        pass

def run_one_tester(run, provider, display):
    tester = Tester(run, provider, display)
    tester.run()

###

class Run(object):
    """
    This class represents a concrete, configured testing run.
    """

    def __init__(self, lit_config, tests):
        self.lit_config = lit_config
        self.tests = tests

    def execute_test(self, test):
        result = None
        start_time = time.time()
        try:
            result = test.config.test_format.execute(test, self.lit_config)

            # Support deprecated result from execute() which returned the result
            # code and additional output as a tuple.
            if isinstance(result, tuple):
                code, output = result
                result = lit.Test.Result(code, output)
            elif not isinstance(result, lit.Test.Result):
                raise ValueError("unexpected result from test execution")
        except KeyboardInterrupt:
            raise
        except:
            if self.lit_config.debug:
                raise
            output = 'Exception during script execution:\n'
            output += traceback.format_exc()
            output += '\n'
            result = lit.Test.Result(lit.Test.UNRESOLVED, output)
        result.elapsed = time.time() - start_time

        test.setResult(result)

    def execute_tests(self, display, jobs, max_time=None):
        """
        execute_tests(display, jobs, [max_time])

        Execute each of the tests in the run, using up to jobs number of
        parallel tasks, and inform the display of each individual result. The
        provided tests should be a subset of the tests available in this run
        object.

        If max_time is non-None, it should be a time in seconds after which to
        stop executing tests.

        The display object will have its update method called with each test as
        it is completed. The calls are guaranteed to be locked with respect to
        one another, but are *not* guaranteed to be called on the same thread as
        this method was invoked on.

        Upon completion, each test in the run will have its result
        computed. Tests which were not actually executed (for any reason) will
        be given an UNRESOLVED result.
        """

        # Create the test provider object.
        provider = TestProvider(self.tests)

        # Install a console-control signal handler on Windows.
        if win32api is not None:
            def console_ctrl_handler(type):
                provider.cancel()
                return True
            win32api.SetConsoleCtrlHandler(console_ctrl_handler, True)

        # Install a timeout handler, if requested.
        if max_time is not None:
            def timeout_handler():
                provider.cancel()
            timeout_timer = threading.Timer(max_time, timeout_handler)
            timeout_timer.start()

        # Actually execute the tests.
        self._execute_tests_with_provider(provider, display, jobs)

        # Cancel the timeout handler.
        if max_time is not None:
            timeout_timer.cancel()

        # Update results for any tests which weren't run.
        for test in self.tests:
            if test.result is None:
                test.setResult(lit.Test.Result(lit.Test.UNRESOLVED, '', 0.0))

    def _execute_tests_with_provider(self, provider, display, jobs):
        consumer = ThreadResultsConsumer(display)

        # If only using one testing thread, don't use tasks at all; this lets us
        # profile, among other things.
        if jobs == 1:
            run_one_tester(self, provider, consumer)
            return

        # Start all of the tasks.
        tasks = [threading.Thread(target=run_one_tester,
                                  args=(self, provider, consumer))
                 for i in range(jobs)]
        for t in tasks:
            t.start()

        # Allow the consumer to handle results, if necessary.
        consumer.handle_results()

        # Wait for all the tasks to complete.
        for t in tasks:
            t.join()
