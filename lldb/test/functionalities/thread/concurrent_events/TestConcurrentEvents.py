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

import os, signal, time
import unittest2
import lldb
from lldbtest import *
import lldbutil

# ==================================================
# Dictionary of signal names
# ==================================================
signal_names = dict((getattr(signal, n), n) \
        for n in dir(signal) if n.startswith('SIG') and '_' not in n )


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

    @unittest2.skipIf(TestBase.skipLongRunningTest(), "Skip this long running test")
    @dwarf_test
    def test_many_watchpoints_dwarf(self):
        """Test 100 watchpoints from 100 threads."""
        self.buildDwarf(dictionary=self.getBuildFlags())
        self.do_thread_actions(num_watchpoint_threads=100)

    @unittest2.skipIf(TestBase.skipLongRunningTest(), "Skip this long running test")
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
    @dwarf_test
    def test_signal_break_dwarf(self):
        """Test signal and a breakpoint in multiple threads."""
        self.buildDwarf(dictionary=self.getBuildFlags())
        self.do_thread_actions(num_breakpoint_threads=1, num_signal_threads=1)

    @dwarf_test
    def test_delay_signal_break_dwarf(self):
        """Test (1-second delay) signal and a breakpoint in multiple threads."""
        self.buildDwarf(dictionary=self.getBuildFlags())
        self.do_thread_actions(num_breakpoint_threads=1, num_delay_signal_threads=1)

    @dwarf_test
    def test_signal_delay_break_dwarf(self):
        """Test signal and a (1 second delay) breakpoint in multiple threads."""
        self.buildDwarf(dictionary=self.getBuildFlags())
        self.do_thread_actions(num_delay_breakpoint_threads=1, num_signal_threads=1)


    #
    ## Tests for concurrent watchpoint and breakpoint
    #
    @dwarf_test
    def test_watch_break_dwarf(self):
        """Test watchpoint and a breakpoint in multiple threads."""
        self.buildDwarf(dictionary=self.getBuildFlags())
        self.do_thread_actions(num_breakpoint_threads=1, num_watchpoint_threads=1)

    @dwarf_test
    def test_delay_watch_break_dwarf(self):
        """Test (1-second delay) watchpoint and a breakpoint in multiple threads."""
        self.buildDwarf(dictionary=self.getBuildFlags())
        self.do_thread_actions(num_breakpoint_threads=1, num_delay_watchpoint_threads=1)

    @dwarf_test
    def test_watch_break_dwarf(self):
        """Test watchpoint and a (1 second delay) breakpoint in multiple threads."""
        self.buildDwarf(dictionary=self.getBuildFlags())
        self.do_thread_actions(num_delay_breakpoint_threads=1, num_watchpoint_threads=1)

    #
    ## Tests for concurrent signal and watchpoint
    #
    @dwarf_test
    def test_signal_watch_dwarf(self):
        """Test a watchpoint and a signal in multiple threads."""
        self.buildDwarf(dictionary=self.getBuildFlags())
        self.do_thread_actions(num_signal_threads=1, num_watchpoint_threads=1)

    @dwarf_test
    def test_delay_signal_watch_dwarf(self):
        """Test a watchpoint and a (1 second delay) signal in multiple threads."""
        self.buildDwarf(dictionary=self.getBuildFlags())
        self.do_thread_actions(num_delay_signal_threads=1, num_watchpoint_threads=1)

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

    @dwarf_test
    def test_two_breakpoints_one_signal_dwarf(self):
        """Test two threads that trigger a breakpoint and one signal thread. """
        self.buildDwarf(dictionary=self.getBuildFlags())
        self.do_thread_actions(num_breakpoint_threads=2, num_signal_threads=1)

    @dwarf_test
    def test_breakpoint_delay_breakpoint_one_signal_dwarf(self):
        """Test two threads that trigger a breakpoint (one with a 1 second delay) and one signal thread. """
        self.buildDwarf(dictionary=self.getBuildFlags())
        self.do_thread_actions(num_breakpoint_threads=1,
                               num_delay_breakpoint_threads=1,
                               num_signal_threads=1)

    @dwarf_test
    def test_two_breakpoints_one_delay_signal_dwarf(self):
        """Test two threads that trigger a breakpoint and one (1 second delay) signal thread. """
        self.buildDwarf(dictionary=self.getBuildFlags())
        self.do_thread_actions(num_breakpoint_threads=2, num_delay_signal_threads=1)

    @dwarf_test
    def test_two_breakpoints_one_watchpoint_dwarf(self):
        """Test two threads that trigger a breakpoint and one watchpoint thread. """
        self.buildDwarf(dictionary=self.getBuildFlags())
        self.do_thread_actions(num_breakpoint_threads=2, num_watchpoint_threads=1)

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
    @dwarf_test
    def test_two_watchpoint_threads_dwarf(self):
        """Test two threads that trigger a watchpoint. """
        self.buildDwarf(dictionary=self.getBuildFlags())
        self.do_thread_actions(num_watchpoint_threads=2)

    @dwarf_test
    def test_watchpoint_with_delay_waychpoint_threads_dwarf(self):
        """Test two threads that trigger a watchpoint where one thread has a 1 second delay. """
        self.buildDwarf(dictionary=self.getBuildFlags())
        self.do_thread_actions(num_watchpoint_threads=1,
                               num_delay_watchpoint_threads=1)

    @dwarf_test
    def test_two_watchpoints_one_breakpoint_dwarf(self):
        """Test two threads that trigger a watchpoint and one breakpoint thread. """
        self.buildDwarf(dictionary=self.getBuildFlags())
        self.do_thread_actions(num_watchpoint_threads=2, num_breakpoint_threads=1)

    @dwarf_test
    def test_two_watchpoints_one_delay_breakpoint_dwarf(self):
        """Test two threads that trigger a watchpoint and one (1 second delay) breakpoint thread. """
        self.buildDwarf(dictionary=self.getBuildFlags())
        self.do_thread_actions(num_watchpoint_threads=2, num_delay_breakpoint_threads=1)

    @dwarf_test
    def test_watchpoint_delay_watchpoint_one_breakpoint_dwarf(self):
        """Test two threads that trigger a watchpoint (one with a 1 second delay) and one breakpoint thread. """
        self.buildDwarf(dictionary=self.getBuildFlags())
        self.do_thread_actions(num_watchpoint_threads=1,
                               num_delay_watchpoint_threads=1,
                               num_breakpoint_threads=1)

    @dwarf_test
    def test_two_watchpoints_one_signal_dwarf(self):
        """Test two threads that trigger a watchpoint and one signal thread. """
        self.buildDwarf(dictionary=self.getBuildFlags())
        self.do_thread_actions(num_watchpoint_threads=2, num_signal_threads=1)

    #
    ## Test for watchpoint, signal and breakpoint happening concurrently
    #
    @dwarf_test
    def test_signal_watch_break_dwarf(self):
        """Test a signal/watchpoint/breakpoint in multiple threads."""
        self.buildDwarf(dictionary=self.getBuildFlags())
        self.do_thread_actions(num_signal_threads=1,
                               num_watchpoint_threads=1,
                               num_breakpoint_threads=1)

    @dwarf_test
    def test_signal_watch_break_dwarf(self):
        """Test one signal thread with 5 watchpoint and breakpoint threads."""
        self.buildDwarf(dictionary=self.getBuildFlags())
        self.do_thread_actions(num_signal_threads=1,
                               num_watchpoint_threads=5,
                               num_breakpoint_threads=5)

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

    @dwarf_test
    def test_crash_with_watchpoint_dwarf(self):
        """ Test a thread that crashes while another thread hits a watchpoint."""
        self.buildDwarf(dictionary=self.getBuildFlags())
        self.do_thread_actions(num_crash_threads=1, num_watchpoint_threads=1)

    @dwarf_test
    def test_crash_with_signal_dwarf(self):
        """ Test a thread that crashes while another thread generates a signal."""
        self.buildDwarf(dictionary=self.getBuildFlags())
        self.do_thread_actions(num_crash_threads=1, num_signal_threads=1)

    @dwarf_test
    def test_crash_with_watchpoint_breakpoint_signal_dwarf(self):
        """ Test a thread that crashes while other threads generate a signal and hit a watchpoint and breakpoint. """
        self.buildDwarf(dictionary=self.getBuildFlags())
        self.do_thread_actions(num_crash_threads=1,
                               num_breakpoint_threads=1,
                               num_signal_threads=1,
                               num_watchpoint_threads=1)

    @dwarf_test
    def test_delayed_crash_with_breakpoint_watchpoint_dwarf(self):
        """ Test a thread with a delayed crash while other threads hit a watchpoint and a breakpoint. """
        self.buildDwarf(dictionary=self.getBuildFlags())
        self.do_thread_actions(num_delay_crash_threads=1,
                               num_breakpoint_threads=1,
                               num_watchpoint_threads=1)

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
        self.filename = 'main.cpp'
        self.thread_breakpoint_line = line_number(self.filename, '// Set breakpoint here')
        self.setup_breakpoint_line = line_number(self.filename, '// Break here and adjust num')
        self.finish_breakpoint_line = line_number(self.filename, '// Break here and verify one thread is active')

    def describe_threads(self):
        ret = []
        for x in self.inferior_process:
            id = x.GetIndexID()
            reason = x.GetStopReason()
            status = "stopped" if x.IsStopped() else "running"
            reason_str = lldbutil.stop_reason_to_str(reason)
            if reason == lldb.eStopReasonBreakpoint:
                bpid = x.GetStopReasonDataAtIndex(0)
                bp = self.inferior_target.FindBreakpointByID(bpid)
                reason_str = "%s hit %d times" % (lldbutil.get_description(bp), bp.GetHitCount())
            elif reason == lldb.eStopReasonWatchpoint:
                watchid = x.GetStopReasonDataAtIndex(0)
                watch = self.inferior_target.FindWatchpointByID(watchid)
                reason_str = "%s hit %d times" % (lldbutil.get_description(watch), watch.GetHitCount())
            elif reason == lldb.eStopReasonSignal:
                reason_str = "signal %s" % (signal_names[x.GetStopReasonDataAtIndex(0)])

            location = "\t".join([lldbutil.get_description(x.GetFrameAtIndex(i)) for i in range(x.GetNumFrames())])
            ret.append("thread %d %s due to %s at\n\t%s" % (id, status, reason_str, location))
        return ret

    def add_breakpoint(self, line, descriptions):
        """ Adds a breakpoint at self.filename:line and appends its description to descriptions, and
            returns the LLDB SBBreakpoint object.
        """

        bpno = lldbutil.run_break_set_by_file_and_line(self, self.filename, line, num_expected_locations=-1)
        bp = self.inferior_target.FindBreakpointByID(bpno)
        descriptions.append(": file = 'main.cpp', line = %d" % self.finish_breakpoint_line)
        return bp

    def inferior_done(self):
        """ Returns true if the inferior is done executing all the event threads (and is stopped at self.finish_breakpoint, 
            or has terminated execution.
        """
        return self.finish_breakpoint.GetHitCount() > 0 or \
                self.crash_count > 0 or \
                self.inferior_process.GetState == lldb.eStateExited

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

        # Get the target
        self.inferior_target = self.dbg.GetSelectedTarget()

        expected_bps = []

        # Initialize all the breakpoints (main thread/aux thread)
        self.setup_breakpoint = self.add_breakpoint(self.setup_breakpoint_line, expected_bps)
        self.finish_breakpoint = self.add_breakpoint(self.finish_breakpoint_line, expected_bps)

        # Set the thread breakpoint
        if num_breakpoint_threads + num_delay_breakpoint_threads > 0:
            self.thread_breakpoint = self.add_breakpoint(self.thread_breakpoint_line, expected_bps)

        # Verify breakpoints
        self.expect("breakpoint list -f", "Breakpoint locations shown correctly", substrs = expected_bps)

        # Run the program.
        self.runCmd("run", RUN_SUCCEEDED)

        # Check we are at line self.setup_breakpoint
        self.expect("thread backtrace", STOPPED_DUE_TO_BREAKPOINT,
            substrs = ["stop reason = breakpoint 1."])

        # Initialize the (single) watchpoint on the global variable (g_watchme)
        if num_watchpoint_threads + num_delay_watchpoint_threads > 0:
            self.runCmd("watchpoint set variable g_watchme")
            for w in self.inferior_target.watchpoint_iter():
                self.thread_watchpoint = w
                self.assertTrue("g_watchme" in str(self.thread_watchpoint), "Watchpoint location not shown correctly")

        # Get the process
        self.inferior_process = self.inferior_target.GetProcess()

        # We should be stopped at the setup site where we can set the number of
        # threads doing each action (break/crash/signal/watch)
        self.assertEqual(self.inferior_process.GetNumThreads(), 1, 'Expected to stop before any additional threads are spawned.')

        self.runCmd("expr num_breakpoint_threads=%d" % num_breakpoint_threads)
        self.runCmd("expr num_crash_threads=%d" % num_crash_threads)
        self.runCmd("expr num_signal_threads=%d" % num_signal_threads)
        self.runCmd("expr num_watchpoint_threads=%d" % num_watchpoint_threads)

        self.runCmd("expr num_delay_breakpoint_threads=%d" % num_delay_breakpoint_threads)
        self.runCmd("expr num_delay_crash_threads=%d" % num_delay_crash_threads)
        self.runCmd("expr num_delay_signal_threads=%d" % num_delay_signal_threads)
        self.runCmd("expr num_delay_watchpoint_threads=%d" % num_delay_watchpoint_threads)

        # Continue the inferior so threads are spawned
        self.runCmd("continue")

        # Make sure we see all the threads. The inferior program's threads all synchronize with a pseudo-barrier; that is,
        # the inferior program ensures all threads are started and running before any thread triggers its 'event'.
        num_threads = self.inferior_process.GetNumThreads()
        expected_num_threads = num_breakpoint_threads + num_delay_breakpoint_threads \
                             + num_signal_threads + num_delay_signal_threads \
                             + num_watchpoint_threads + num_delay_watchpoint_threads \
                             + num_crash_threads + num_delay_crash_threads + 1
        self.assertEqual(num_threads, expected_num_threads,
            'Expected to see %d threads, but seeing %d. Details:\n%s' % (expected_num_threads,
                                                                         num_threads,
                                                                         "\n\t".join(self.describe_threads())))

        self.signal_count = len(lldbutil.get_stopped_threads(self.inferior_process, lldb.eStopReasonSignal))
        self.crash_count = len(lldbutil.get_stopped_threads(self.inferior_process, lldb.eStopReasonException))

        # Run to completion (or crash)
        while not self.inferior_done(): 
            if self.TraceOn():
                self.runCmd("thread backtrace all")
            self.runCmd("continue")
            self.signal_count += len(lldbutil.get_stopped_threads(self.inferior_process, lldb.eStopReasonSignal))
            self.crash_count += len(lldbutil.get_stopped_threads(self.inferior_process, lldb.eStopReasonException))

        if num_crash_threads > 0 or num_delay_crash_threads > 0:
            # Expecting a crash
            self.assertTrue(self.crash_count > 0,
                "Expecting at least one thread to crash. Details: %s" % "\t\n".join(self.describe_threads()))

            # Ensure the zombie process is reaped
            self.runCmd("process kill")

        elif num_crash_threads == 0 and num_delay_crash_threads == 0:
            # There should be a single active thread (the main one) which hit the breakpoint after joining
            self.assertEqual(1, self.finish_breakpoint.GetHitCount(), "Expected main thread (finish) breakpoint to be hit once")

            # llvm.org/pr16603 -- LLDB on Linux sometimes reports exited threads as still 'running'
            #num_threads = self.inferior_process.GetNumThreads()
            #self.assertEqual(1, num_threads, "Expecting 1 thread but seeing %d. Details:%s" % (num_threads,
            #                                                                                 "\n\t".join(self.describe_threads())))
            self.runCmd("continue")

            # The inferior process should have exited without crashing
            self.assertEqual(0, self.crash_count, "Unexpected thread(s) in crashed state")
            self.assertTrue(self.inferior_process.GetState() == lldb.eStateExited, PROCESS_EXITED)

            # Verify the number of actions took place matches expected numbers
            expected_breakpoint_threads = num_delay_breakpoint_threads + num_breakpoint_threads
            breakpoint_hit_count = self.thread_breakpoint.GetHitCount() if expected_breakpoint_threads > 0 else 0
            self.assertEqual(expected_breakpoint_threads, breakpoint_hit_count,
                "Expected %d breakpoint hits, but got %d" % (expected_breakpoint_threads, breakpoint_hit_count))

            expected_signal_threads = num_delay_signal_threads + num_signal_threads
            self.assertEqual(expected_signal_threads, self.signal_count,
                "Expected %d stops due to signal delivery, but got %d" % (expected_breakpoint_threads, self.signal_count))

            expected_watchpoint_threads = num_delay_watchpoint_threads + num_watchpoint_threads
            watchpoint_hit_count = self.thread_watchpoint.GetHitCount() if expected_watchpoint_threads > 0 else 0
            self.assertEqual(expected_watchpoint_threads, watchpoint_hit_count,
                "Expected %d watchpoint hits, got %d" % (expected_watchpoint_threads, watchpoint_hit_count))


if __name__ == '__main__':
    import atexit
    lldb.SBDebugger.Initialize()
    atexit.register(lambda: lldb.SBDebugger.Terminate())
    unittest2.main()
