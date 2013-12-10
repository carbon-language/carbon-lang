"""
Test that lldb watchpoint works for multiple threads.
"""

import os, time
import unittest2
import re
import lldb
from lldbtest import *
import lldbutil

class WatchpointForMultipleThreadsTestCase(TestBase):

    mydir = TestBase.compute_mydir(__file__)

    @unittest2.skipUnless(sys.platform.startswith("darwin"), "requires Darwin")
    @dsym_test
    def test_watchpoint_multiple_threads_with_dsym(self):
        """Test that lldb watchpoint works for multiple threads."""
        self.buildDsym(dictionary=self.d)
        self.setTearDownCleanup(dictionary=self.d)
        self.hello_multiple_threads()

    @expectedFailureFreeBSD('llvm.org/pr16706') # Watchpoints fail on FreeBSD
    @skipIfGcc # causes intermittent gcc debian buildbot failures, skip until we can investigate
    @dwarf_test
    def test_watchpoint_multiple_threads_with_dwarf(self):
        """Test that lldb watchpoint works for multiple threads."""
        self.buildDwarf(dictionary=self.d)
        self.setTearDownCleanup(dictionary=self.d)
        self.hello_multiple_threads()

    @unittest2.skipUnless(sys.platform.startswith("darwin"), "requires Darwin")
    @dsym_test
    def test_watchpoint_multiple_threads_wp_set_and_then_delete_with_dsym(self):
        """Test that lldb watchpoint works for multiple threads, and after the watchpoint is deleted, the watchpoint event should no longer fires."""
        self.buildDsym(dictionary=self.d)
        self.setTearDownCleanup(dictionary=self.d)
        self.hello_multiple_threads_wp_set_and_then_delete()

    @expectedFailureFreeBSD('llvm.org/pr16706') # Watchpoints fail on FreeBSD
    @skipIfGcc # causes intermittent gcc debian buildbot failures, skip until we can investigate 
    @dwarf_test
    def test_watchpoint_multiple_threads_wp_set_and_then_delete_with_dwarf(self):
        """Test that lldb watchpoint works for multiple threads, and after the watchpoint is deleted, the watchpoint event should no longer fires."""
        self.buildDwarf(dictionary=self.d)
        self.setTearDownCleanup(dictionary=self.d)
        self.hello_multiple_threads_wp_set_and_then_delete()

    def setUp(self):
        # Call super's setUp().
        TestBase.setUp(self)
        # Our simple source filename.
        self.source = 'main.cpp'
        # Find the line number to break inside main().
        self.first_stop = line_number(self.source, '// Set break point at this line')
        # Build dictionary to have unique executable names for each test method.
        self.exe_name = self.testMethodName
        self.d = {'CXX_SOURCES': self.source, 'EXE': self.exe_name}

    def hello_multiple_threads(self):
        """Test that lldb watchpoint works for multiple threads."""
        exe = os.path.join(os.getcwd(), self.exe_name)
        self.runCmd("file " + exe, CURRENT_EXECUTABLE_SET)

        # Add a breakpoint to set a watchpoint when stopped on the breakpoint.
        lldbutil.run_break_set_by_file_and_line (self, None, self.first_stop, num_expected_locations=1)

        # Run the program.
        self.runCmd("run", RUN_SUCCEEDED)

        # We should be stopped again due to the breakpoint.
        # The stop reason of the thread should be breakpoint.
        self.expect("thread list", STOPPED_DUE_TO_BREAKPOINT,
            substrs = ['stopped',
                       'stop reason = breakpoint'])

        # Now let's set a write-type watchpoint for variable 'g_val'.
        # The main.cpp, by design, misbehaves by not following the agreed upon
        # protocol of using a mutex while accessing the global pool and by not
        # writing to the variable.
        self.expect("watchpoint set variable -w write g_val", WATCHPOINT_CREATED,
            substrs = ['Watchpoint created', 'size = 4', 'type = w'])

        # Use the '-v' option to do verbose listing of the watchpoint.
        # The hit count should be 0 initially.
        self.expect("watchpoint list -v",
            substrs = ['hit_count = 0'])

        while True:
            self.runCmd("process continue")

            self.runCmd("thread list")
            if "stop reason = watchpoint" in self.res.GetOutput():
                # Good, we verified that the watchpoint works!
                self.runCmd("thread backtrace all")
                break
            else:
                self.fail("The stop reason should be either break or watchpoint")

        # Use the '-v' option to do verbose listing of the watchpoint.
        # The hit count should now be 1.
        self.expect("watchpoint list -v",
            substrs = ['hit_count = 1'])

    def hello_multiple_threads_wp_set_and_then_delete(self):
        """Test that lldb watchpoint works for multiple threads, and after the watchpoint is deleted, the watchpoint event should no longer fires."""
        exe = os.path.join(os.getcwd(), self.exe_name)
        self.runCmd("file " + exe, CURRENT_EXECUTABLE_SET)

        # Add a breakpoint to set a watchpoint when stopped on the breakpoint.
        lldbutil.run_break_set_by_file_and_line (self, None, self.first_stop, num_expected_locations=1)

        # Run the program.
        self.runCmd("run", RUN_SUCCEEDED)

        # We should be stopped again due to the breakpoint.
        # The stop reason of the thread should be breakpoint.
        self.expect("thread list", STOPPED_DUE_TO_BREAKPOINT,
            substrs = ['stopped',
                       'stop reason = breakpoint'])

        # Now let's set a write-type watchpoint for variable 'g_val'.
        # The main.cpp, by design, misbehaves by not following the agreed upon
        # protocol of using a mutex while accessing the global pool and by not
        # writing to the variable.
        self.expect("watchpoint set variable -w write g_val", WATCHPOINT_CREATED,
            substrs = ['Watchpoint created', 'size = 4', 'type = w'])

        # Use the '-v' option to do verbose listing of the watchpoint.
        # The hit count should be 0 initially.
        self.expect("watchpoint list -v",
            substrs = ['hit_count = 0'])

        watchpoint_stops = 0
        while True:
            self.runCmd("process continue")
            self.runCmd("process status")
            if re.search("Process .* exited", self.res.GetOutput()):
                # Great, we are done with this test!
                break

            self.runCmd("thread list")
            if "stop reason = watchpoint" in self.res.GetOutput():
                self.runCmd("thread backtrace all")
                watchpoint_stops += 1
                if watchpoint_stops > 1:
                    self.fail("Watchpoint hits not supposed to exceed 1 by design!")
                # Good, we verified that the watchpoint works!  Now delete the watchpoint.
                if self.TraceOn():
                    print "watchpoint_stops=%d at the moment we delete the watchpoint" % watchpoint_stops
                self.runCmd("watchpoint delete 1")
                self.expect("watchpoint list -v",
                    substrs = ['No watchpoints currently set.'])
                continue
            else:
                self.fail("The stop reason should be either break or watchpoint")


if __name__ == '__main__':
    import atexit
    lldb.SBDebugger.Initialize()
    atexit.register(lambda: lldb.SBDebugger.Terminate())
    unittest2.main()
