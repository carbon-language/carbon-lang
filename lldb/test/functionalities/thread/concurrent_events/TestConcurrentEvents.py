"""
A stress-test of sorts for LLDB's handling of threads in the inferior.

This test sets a breakpoint in the main thread where test parameters (numbers of
threads) can be adjusted, runs the inferior to that point, and modifies the
locals that control the event thread counts. This test also sets a breakpoint in
breakpoint_func (the function executed by each 'breakpoint' thread) and a
watchpoint on a global modified in watchpoint_func. The inferior is continued
until exit or a crash takes place, and the number of events seen by LLDB is
verified to match the expected number of events.
"""

import os, time
import unittest2
import lldb
from lldbtest import *
import lldbutil

class ConcurrentEventsTestCase(TestBase):

    mydir = os.path.join("functionalities", "thread", "concurrent_events")

    #
    ## Tests for multiple threads that generate a single event.
    #
    @unittest2.skipIf(TestBase.skipLongRunningTest(), "Skip this long running test")
    @dwarf_test
    def test_many_breakpoints_dwarf(self):
        """Test 100 breakpoints from 100 threads."""
        self.buildDwarf(dictionary=self.getBuildFlags())
        self.do_thread_actions(num_breakpoint_threads=100)

    @skipIfDarwin # llvm.org/pr16566 -- new threads do not respect watchpoints
    @unittest2.skipIf(TestBase.skipLongRunningTest(), "Skip this long running test")
    @dwarf_test
    def test_many_watchpoints_dwarf(self):
        """Test 100 watchpoints from 100 threads."""
        self.buildDwarf(dictionary=self.getBuildFlags())
        self.do_thread_actions(num_watchpoint_threads=100)

    @unittest2.skipIf(TestBase.skipLongRunningTest(), "Skip this long running test")
    @skipIfDarwin # llvm.org/pr16567 -- thread count is incorrect during signal delivery
    @dwarf_test
    def test_many_signals_dwarf(self):
        """Test 100 signals from 100 threads."""
        self.buildDwarf(dictionary=self.getBuildFlags())
        self.do_thread_actions(num_signal_threads=100)

    @unittest2.skipIf(TestBase.skipLongRunningTest(), "Skip this long running test")
    @dwarf_test
    def test_many_crash_dwarf(self):
        """Test 100 threads that cause a segfault."""
        self.buildDwarf(dictionary=self.getBuildFlags())
        self.do_thread_actions(num_crash_threads=100)


    #
    ## Tests for concurrent signal and breakpoint
    #
    @skipIfDarwin # llvm.org/pr16567 -- thread count is incorrect during signal delivery
    @skipIfLinux # llvm.org/pr16575 -- LLDB crashes with assertion failure "Unexpected SIGTRAP code!"
    @dwarf_test
    def test_signal_break_dwarf(self):
        """Test signal and a breakpoint in multiple threads."""
        self.buildDwarf(dictionary=self.getBuildFlags())
        self.do_thread_actions(num_breakpoint_threads=1, num_signal_threads=1)

    @skipIfDarwin # llvm.org/pr16567 -- thread count is incorrect during signal delivery
    @skipIfLinux # llvm.org/pr16575 -- LLDB crashes with assertion failure "Unexpected SIGTRAP code!"
    @dwarf_test
    def test_delay_signal_break_dwarf(self):
        """Test (1-second delay) signal and a breakpoint in multiple threads."""
        self.buildDwarf(dictionary=self.getBuildFlags())
        self.do_thread_actions(num_breakpoint_threads=1, num_delay_signal_threads=1)

    @skipIfDarwin # llvm.org/pr16567 -- thread count is incorrect during signal delivery
    @skipIfLinux # llvm.org/pr16575 -- LLDB crashes with assertion failure "Unexpected SIGTRAP code!"
    @dwarf_test
    def test_signal_delay_break_dwarf(self):
        """Test signal and a (1 second delay) breakpoint in multiple threads."""
        self.buildDwarf(dictionary=self.getBuildFlags())
        self.do_thread_actions(num_delay_breakpoint_threads=1, num_signal_threads=1)


    #
    ## Tests for concurrent watchpoint and breakpoint
    #
    @skipIfDarwin # llvm.org/pr16566 -- new threads do not respect watchpoints
    @dwarf_test
    def test_watch_break_dwarf(self):
        """Test watchpoint and a breakpoint in multiple threads."""
        self.buildDwarf(dictionary=self.getBuildFlags())
        self.do_thread_actions(num_breakpoint_threads=1, num_watchpoint_threads=1)

    @skipIfDarwin # llvm.org/pr16566 -- new threads do not respect watchpoints
    @dwarf_test
    def test_delay_watch_break_dwarf(self):
        """Test (1-second delay) watchpoint and a breakpoint in multiple threads."""
        self.buildDwarf(dictionary=self.getBuildFlags())
        self.do_thread_actions(num_breakpoint_threads=1, num_delay_watchpoint_threads=1)

    @skipIfDarwin # llvm.org/pr16566 -- new threads do not respect watchpoints
    @dwarf_test
    def test_watch_break_dwarf(self):
        """Test watchpoint and a (1 second delay) breakpoint in multiple threads."""
        self.buildDwarf(dictionary=self.getBuildFlags())
        self.do_thread_actions(num_delay_breakpoint_threads=1, num_watchpoint_threads=1)

    #
    ## Tests for concurrent signal and watchpoint
    #
    @skipIfDarwin # llvm.org/pr16566 -- new threads do not respect watchpoints
    @dwarf_test
    def test_signal_watch_dwarf(self):
        """Test a watchpoint and a signal in multiple threads."""
        self.buildDwarf(dictionary=self.getBuildFlags())
        self.do_thread_actions(num_signal_threads=1, num_watchpoint_threads=1)

    @skipIfDarwin # llvm.org/pr16566 -- new threads do not respect watchpoints
    @dwarf_test
    def test_delay_signal_watch_dwarf(self):
        """Test a watchpoint and a (1 second delay) signal in multiple threads."""
        self.buildDwarf(dictionary=self.getBuildFlags())
        self.do_thread_actions(num_delay_signal_threads=1, num_watchpoint_threads=1)

    @skipIfDarwin # llvm.org/pr16566 -- new threads do not respect watchpoints
    @dwarf_test
    def test_signal_delay_watch_dwarf(self):
        """Test a (1 second delay) watchpoint and a signal in multiple threads."""
        self.buildDwarf(dictionary=self.getBuildFlags())
        self.do_thread_actions(num_signal_threads=1, num_delay_watchpoint_threads=1)


    #
    ## Tests for multiple breakpoint threads
    #
    @dwarf_test
    def test_two_breakpoint_threads_dwarf(self):
        """Test two threads that trigger a breakpoint. """
        self.buildDwarf(dictionary=self.getBuildFlags())
        self.do_thread_actions(num_breakpoint_threads=2)

    @dwarf_test
    def test_breakpoint_one_delay_breakpoint_threads_dwarf(self):
        """Test threads that trigger a breakpoint where one thread has a 1 second delay. """
        self.buildDwarf(dictionary=self.getBuildFlags())
        self.do_thread_actions(num_breakpoint_threads=1,
                               num_delay_breakpoint_threads=1)

    @skipIfDarwin # llvm.org/pr16567 -- thread count is incorrect during signal delivery
    @skipIfLinux # llvm.org/pr16575 -- LLDB crashes with assertion failure "Unexpected SIGTRAP code!"
    @dwarf_test
    def test_two_breakpoints_one_signal_dwarf(self):
        """Test two threads that trigger a breakpoint and one signal thread. """
        self.buildDwarf(dictionary=self.getBuildFlags())
        self.do_thread_actions(num_breakpoint_threads=2, num_signal_threads=1)

    @skipIfDarwin # llvm.org/pr16567 -- thread count is incorrect during signal delivery
    @skipIfLinux # llvm.org/pr16575 -- LLDB crashes with assertion failure "Unexpected SIGTRAP code!"
    @dwarf_test
    def test_breakpoint_delay_breakpoint_one_signal_dwarf(self):
        """Test two threads that trigger a breakpoint (one with a 1 second delay) and one signal thread. """
        self.buildDwarf(dictionary=self.getBuildFlags())
        self.do_thread_actions(num_breakpoint_threads=1,
                               num_delay_breakpoint_threads=1,
                               num_signal_threads=1)

    @skipIfDarwin # llvm.org/pr16567 -- thread count is incorrect during signal delivery
    @skipIfLinux # llvm.org/pr16575 -- LLDB crashes with assertion failure "Unexpected SIGTRAP code!"
    @dwarf_test
    def test_two_breakpoints_one_delay_signal_dwarf(self):
        """Test two threads that trigger a breakpoint and one (1 second delay) signal thread. """
        self.buildDwarf(dictionary=self.getBuildFlags())
        self.do_thread_actions(num_breakpoint_threads=2, num_delay_signal_threads=1)

    @skipIfDarwin # llvm.org/pr16566 -- new threads do not respect watchpoints
    @dwarf_test
    def test_two_breakpoints_one_watchpoint_dwarf(self):
        """Test two threads that trigger a breakpoint and one watchpoint thread. """
        self.buildDwarf(dictionary=self.getBuildFlags())
        self.do_thread_actions(num_breakpoint_threads=2, num_watchpoint_threads=1)

    @skipIfDarwin # llvm.org/pr16566 -- new threads do not respect watchpoints
    @dwarf_test
    def test_breakpoints_delayed_breakpoint_one_watchpoint_dwarf(self):
        """Test a breakpoint, a delayed breakpoint, and one watchpoint thread. """
        self.buildDwarf(dictionary=self.getBuildFlags())
        self.do_thread_actions(num_breakpoint_threads=1,
                               num_delay_breakpoint_threads=1,
                               num_watchpoint_threads=1)

    #
    ## Tests for multiple watchpoint threads
    #
    @skipIfDarwin # llvm.org/pr16566 -- new threads do not respect watchpoints
    @dwarf_test
    def test_two_watchpoint_threads_dwarf(self):
        """Test two threads that trigger a watchpoint. """
        self.buildDwarf(dictionary=self.getBuildFlags())
        self.do_thread_actions(num_watchpoint_threads=2)

    @skipIfDarwin # llvm.org/pr16566 -- new threads do not respect watchpoints
    @dwarf_test
    def test_watchpoint_with_delay_waychpoint_threads_dwarf(self):
        """Test two threads that trigger a watchpoint where one thread has a 1 second delay. """
        self.buildDwarf(dictionary=self.getBuildFlags())
        self.do_thread_actions(num_watchpoint_threads=1,
                               num_delay_watchpoint_threads=1)

    @skipIfDarwin # llvm.org/pr16566 -- new threads do not respect watchpoints
    @dwarf_test
    def test_two_watchpoints_one_breakpoint_dwarf(self):
        """Test two threads that trigger a watchpoint and one breakpoint thread. """
        self.buildDwarf(dictionary=self.getBuildFlags())
        self.do_thread_actions(num_watchpoint_threads=2, num_breakpoint_threads=1)

    @skipIfDarwin # llvm.org/pr16566 -- new threads do not respect watchpoints
    @dwarf_test
    def test_two_watchpoints_one_delay_breakpoint_dwarf(self):
        """Test two threads that trigger a watchpoint and one (1 second delay) breakpoint thread. """
        self.buildDwarf(dictionary=self.getBuildFlags())
        self.do_thread_actions(num_watchpoint_threads=2, num_delay_breakpoint_threads=1)

    @skipIfDarwin # llvm.org/pr16566 -- new threads do not respect watchpoints
    @dwarf_test
    def test_watchpoint_delay_watchpoint_one_breakpoint_dwarf(self):
        """Test two threads that trigger a watchpoint (one with a 1 second delay) and one breakpoint thread. """
        self.buildDwarf(dictionary=self.getBuildFlags())
        self.do_thread_actions(num_watchpoint_threads=1,
                               num_delay_watchpoint_threads=1,
                               num_breakpoint_threads=1)

    @skipIfDarwin # llvm.org/pr16566 -- new threads do not respect watchpoints
    @dwarf_test
    def test_two_watchpoints_one_signal_dwarf(self):
        """Test two threads that trigger a watchpoint and one signal thread. """
        self.buildDwarf(dictionary=self.getBuildFlags())
        self.do_thread_actions(num_watchpoint_threads=2, num_signal_threads=1)

    #
    ## Test for watchpoint, signal and breakpoint happening concurrently
    #
    @skipIfDarwin # llvm.org/pr16566 -- new threads do not respect watchpoints
    @skipIfLinux # llvm.org/pr16575 -- LLDB crashes with assertion failure "Unexpected SIGTRAP code!"
    @dwarf_test
    def test_signal_watch_break_dwarf(self):
        """Test a signal/watchpoint/breakpoint in multiple threads."""
        self.buildDwarf(dictionary=self.getBuildFlags())
        self.do_thread_actions(num_signal_threads=1,
                               num_watchpoint_threads=1,
                               num_breakpoint_threads=1)

    @skipIfDarwin # llvm.org/pr16566 -- new threads do not respect watchpoints
    @skipIfLinux # llvm.org/pr16575 -- LLDB crashes with assertion failure "Unexpected SIGTRAP code!"
    @dwarf_test
    def test_signal_watch_break_dwarf(self):
        """Test one signal thread with 5 watchpoint and breakpoint threads."""
        self.buildDwarf(dictionary=self.getBuildFlags())
        self.do_thread_actions(num_signal_threads=1,
                               num_watchpoint_threads=5,
                               num_breakpoint_threads=5)

    @skipIfDarwin # llvm.org/pr16566 -- new threads do not respect watchpoints
    @dwarf_test
    def test_signal_watch_break_dwarf(self):
        """Test with 5 watchpoint and breakpoint threads."""
        self.buildDwarf(dictionary=self.getBuildFlags())
        self.do_thread_actions(num_watchpoint_threads=5,
                               num_breakpoint_threads=5)


    #
    ## Test for crashing threads happening concurrently with other events
    #
    @dwarf_test
    def test_crash_with_break_dwarf(self):
        """ Test a thread that crashes while another thread hits a breakpoint."""
        self.buildDwarf(dictionary=self.getBuildFlags())
        self.do_thread_actions(num_crash_threads=1, num_breakpoint_threads=1)

    @skipIfDarwin # llvm.org/pr16566 -- new threads do not respect watchpoints
    @dwarf_test
    def test_crash_with_watchpoint_dwarf(self):
        """ Test a thread that crashes while another thread hits a watchpoint."""
        self.buildDwarf(dictionary=self.getBuildFlags())
        self.do_thread_actions(num_crash_threads=1, num_watchpoint_threads=1)

    @skipIfDarwin # llvm.org/pr16567 -- thread count is incorrect during signal delivery
    @dwarf_test
    def test_crash_with_signal_dwarf(self):
        """ Test a thread that crashes while another thread generates a signal."""
        self.buildDwarf(dictionary=self.getBuildFlags())
        self.do_thread_actions(num_crash_threads=1, num_signal_threads=1)

    @skipIfDarwin # llvm.org/pr16566 -- new threads do not respect watchpoints
    @skipIfLinux # llvm.org/pr16575 -- LLDB crashes with assertion failure "Unexpected SIGTRAP code!"
    @dwarf_test
    def test_crash_with_watchpoint_breakpoint_signal_dwarf(self):
        """ Test a thread that crashes while other threads generate a signal and hit a watchpoint and breakpoint. """
        self.buildDwarf(dictionary=self.getBuildFlags())
        self.do_thread_actions(num_crash_threads=1,
                               num_breakpoint_threads=1,
                               num_signal_threads=1,
                               num_watchpoint_threads=1)

    @skipIfDarwin # llvm.org/pr16566 -- new threads do not respect watchpoints
    @dwarf_test
    def test_delayed_crash_with_breakpoint_watchpoint_dwarf(self):
        """ Test a thread with a delayed crash while other threads hit a watchpoint and a breakpoint. """
        self.buildDwarf(dictionary=self.getBuildFlags())
        self.do_thread_actions(num_delay_crash_threads=1,
                               num_breakpoint_threads=1,
                               num_watchpoint_threads=1)

    @skipIfDarwin # llvm.org/pr16567 -- thread count is incorrect during signal delivery
    @dwarf_test
    def test_delayed_crash_with_breakpoint_signal_dwarf(self):
        """ Test a thread with a delayed crash while other threads generate a signal and hit a breakpoint. """
        self.buildDwarf(dictionary=self.getBuildFlags())
        self.do_thread_actions(num_delay_crash_threads=1,
                               num_breakpoint_threads=1,
                               num_signal_threads=1)


    def setUp(self):
        # Call super's setUp().
        TestBase.setUp(self)
        # Find the line number for our breakpoint.
        self.thread_breakpoint = line_number('main.cpp', '// Set breakpoint here')
        self.setup_breakpoint = line_number('main.cpp', '// Break here and adjust num')

    def print_threads(self, threads):
        ret = ""
        for x in threads:
            ret += "\t thread %d stopped due to reason %s" % (x.GetIndexID(), lldbutil.stop_reason_to_str(x.GetStopReason()))
        return ret

    def debug_threads(self, bps, crashed, exiting, wps, signals, others):
        print "%d threads stopped at bp:\n%s" % (len(bps), self.print_threads(bps))
        print "%d threads crashed:\n%s" % (len(crashed), self.print_threads(crashed))
        print "%d threads stopped due to watchpoint:\n%s" % (len(wps), self.print_threads(wps))
        print "%d threads stopped at signal:\n%s" % (len(signals), self.print_threads(signals))
        print "%d threads exiting:\n%s" % (len(exiting), self.print_threads(exiting))
        print "%d threads stopped due to other/unknown reason:\n%s" % (len(others), self.print_threads(others))

    def do_thread_actions(self,
                          num_breakpoint_threads = 0,
                          num_signal_threads = 0,
                          num_watchpoint_threads = 0,
                          num_crash_threads = 0,
                          num_delay_breakpoint_threads = 0,
                          num_delay_signal_threads = 0,
                          num_delay_watchpoint_threads = 0,
                          num_delay_crash_threads = 0):
        """ Sets a breakpoint in the main thread where test parameters (numbers of threads) can be adjusted, runs the inferior
            to that point, and modifies the locals that control the event thread counts. Also sets a breakpoint in
            breakpoint_func (the function executed by each 'breakpoint' thread) and a watchpoint on a global modified in
            watchpoint_func. The inferior is continued until exit or a crash takes place, and the number of events seen by LLDB
            is verified to match the expected number of events.
        """
        exe = os.path.join(os.getcwd(), "a.out")
        self.runCmd("file " + exe, CURRENT_EXECUTABLE_SET)

        # Initialize all the breakpoints (main thread/aux thread)
        lldbutil.run_break_set_by_file_and_line (self, "main.cpp", self.setup_breakpoint,
            num_expected_locations=1)
        lldbutil.run_break_set_by_file_and_line (self, "main.cpp", self.thread_breakpoint,
            num_expected_locations=1)

        # The breakpoint list should show 2 breakpoints with 1 location.
        self.expect("breakpoint list -f", "Breakpoint location shown correctly",
            substrs = ["1: file = 'main.cpp', line = %d, locations = 1" % self.setup_breakpoint,
                       "2: file = 'main.cpp', line = %d, locations = 1" % self.thread_breakpoint])

        # Run the program.
        self.runCmd("run", RUN_SUCCEEDED)

        # Check we are at line self.setup_breakpoint
        self.expect("thread backtrace", STOPPED_DUE_TO_BREAKPOINT,
            substrs = ["stop reason = breakpoint 1."])

        # Initialize the watchpoint on the global variable (g_watchme)
        if num_watchpoint_threads + num_delay_watchpoint_threads > 0:
            self.runCmd("watchpoint set variable g_watchme")

        # Get the target process
        target = self.dbg.GetSelectedTarget()
        process = target.GetProcess()

        # We should be stopped at the setup site where we can set the number of
        # threads doing each action (break/crash/signal/watch)
        self.assertEqual(process.GetNumThreads(), 1, 'Expected to stop before any additional threads are spawned.')

        self.runCmd("expr num_breakpoint_threads=%d" % num_breakpoint_threads)
        self.runCmd("expr num_crash_threads=%d" % num_crash_threads)
        self.runCmd("expr num_signal_threads=%d" % num_signal_threads)
        self.runCmd("expr num_watchpoint_threads=%d" % num_watchpoint_threads)

        self.runCmd("expr num_delay_breakpoint_threads=%d" % num_delay_breakpoint_threads)
        self.runCmd("expr num_delay_crash_threads=%d" % num_delay_crash_threads)
        self.runCmd("expr num_delay_signal_threads=%d" % num_delay_signal_threads)
        self.runCmd("expr num_delay_watchpoint_threads=%d" % num_delay_watchpoint_threads)

        self.runCmd("continue")

        # Make sure we see all the threads. The inferior program's threads all synchronize with a pseudo-barrier; that is,
        # the inferior program ensures all threads are started and running before any thread triggers its 'event'.
        num_threads = process.GetNumThreads()
        expected_num_threads = num_breakpoint_threads + num_delay_breakpoint_threads \
                             + num_signal_threads + num_delay_signal_threads \
                             + num_watchpoint_threads + num_delay_watchpoint_threads \
                             + num_crash_threads + num_delay_crash_threads + 1
        self.assertEqual(num_threads, expected_num_threads,
            'Number of expected threads and actual threads do not match.')

        # Get the thread objects
        (breakpoint_threads, crashed_threads, exiting_threads, other_threads, signal_threads, watchpoint_threads) = ([], [], [], [], [], [])
        lldbutil.sort_stopped_threads(process,
                                      breakpoint_threads=breakpoint_threads,
                                      crashed_threads=crashed_threads,
                                      exiting_threads=exiting_threads,
                                      signal_threads=signal_threads,
                                      watchpoint_threads=watchpoint_threads,
                                      other_threads=other_threads)

        if self.TraceOn():
            self.debug_threads(breakpoint_threads, crashed_threads, exiting_threads, watchpoint_threads, signal_threads, other_threads)

        # The threads that are doing signal handling must be unblocked or the inferior will hang. We keep
        # a counter of threads that stop due to a signal so we have something to verify later on.
        seen_signal_threads = len(signal_threads)
        seen_breakpoint_threads = len(breakpoint_threads)
        seen_watchpoint_threads = len(watchpoint_threads)
        seen_crashed_threads = len(crashed_threads)

        # Run to completion
        while len(crashed_threads) == 0 and process.GetState() != lldb.eStateExited:
            if self.TraceOn():
                self.runCmd("thread backtrace all")
                self.debug_threads(breakpoint_threads, crashed_threads, exiting_threads, watchpoint_threads, signal_threads, other_threads)

            self.runCmd("continue")
            lldbutil.sort_stopped_threads(process,
                                          breakpoint_threads=breakpoint_threads,
                                          crashed_threads=crashed_threads,
                                          exiting_threads=exiting_threads,
                                          signal_threads=signal_threads,
                                          watchpoint_threads=watchpoint_threads,
                                          other_threads=other_threads)
            seen_signal_threads += len(signal_threads)
            seen_breakpoint_threads += len(breakpoint_threads)
            seen_watchpoint_threads += len(watchpoint_threads)
            seen_crashed_threads += len(crashed_threads)

        if num_crash_threads > 0 or num_delay_crash_threads > 0:
            # Expecting a crash
            self.assertTrue(seen_crashed_threads > 0, "Expecting at least one thread to crash")

            # Ensure the zombie process is reaped
            self.runCmd("process kill")

        elif num_crash_threads == 0 and num_delay_crash_threads == 0:
            # The inferior process should have exited without crashing
            self.assertEqual(0, seen_crashed_threads, "Unexpected thread(s) in crashed state")
            self.assertTrue(process.GetState() == lldb.eStateExited, PROCESS_EXITED)

            # Verify the number of actions took place matches expected numbers
            self.assertEqual(num_delay_breakpoint_threads + num_breakpoint_threads, seen_breakpoint_threads)
            self.assertEqual(num_delay_signal_threads + num_signal_threads, seen_signal_threads)
            self.assertEqual(num_delay_watchpoint_threads + num_watchpoint_threads, seen_watchpoint_threads)


if __name__ == '__main__':
    import atexit
    lldb.SBDebugger.Initialize()
    atexit.register(lambda: lldb.SBDebugger.Terminate())
    unittest2.main()
