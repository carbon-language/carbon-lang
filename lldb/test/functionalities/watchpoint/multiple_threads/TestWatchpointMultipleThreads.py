"""
Test that lldb watchpoint works for multiple threads.
"""

import os, time
import unittest2
import lldb
from lldbtest import *

class WatchpointForMultipleThreadsTestCase(TestBase):

    mydir = os.path.join("functionalities", "watchpoint", "multiple_threads")

    @unittest2.skipUnless(sys.platform.startswith("darwin"), "requires Darwin")
    def test_watchpoint_multiple_threads_with_dsym(self):
        """Test that lldb watchpoint works for multiple threads."""
        self.buildDsym(dictionary=self.d)
        self.setTearDownCleanup(dictionary=self.d)
        self.hello_multiple_threads()

    def test_watchpoint_multiple_threads_with_dwarf(self):
        """Test that lldb watchpoint works for multiple threads."""
        self.buildDwarf(dictionary=self.d)
        self.setTearDownCleanup(dictionary=self.d)
        self.hello_multiple_threads()

    def setUp(self):
        # Call super's setUp().
        TestBase.setUp(self)
        # Our simple source filename.
        self.source = 'main.cpp'
        # Find the line number to break inside main().
        self.line = line_number(self.source, '// Set break point at this line.')
        # Build dictionary to have unique executable names for each test method.
        self.exe_name = self.testMethodName
        self.d = {'CXX_SOURCES': self.source, 'EXE': self.exe_name}

    def hello_multiple_threads(self):
        """Test that lldb watchpoint works for multiple threads."""
        exe = os.path.join(os.getcwd(), self.exe_name)
        self.runCmd("file " + exe, CURRENT_EXECUTABLE_SET)

        # Add a breakpoint to set a watchpoint when stopped on the breakpoint.
        self.expect("breakpoint set -l %d" % self.line, BREAKPOINT_CREATED,
            startstr = "Breakpoint created: 1: file ='%s', line = %d, locations = 1" %
                       (self.source, self.line))

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
        self.expect("frame variable -w write -g g_val", WATCHPOINT_CREATED,
            substrs = ['Watchpoint created', 'size = 4', 'type = w'])

        # Use the '-v' option to do verbose listing of the watchpoint.
        # The hit count should be 0 initially.
        self.expect("watchpoint list -v",
            substrs = ['hit_count = 0'])

        self.runCmd("process continue")

        # We should be stopped again due to the watchpoint (write type) in a
        # different work thread.  And the stop reason of the thread should be
        # watchpoint.
        self.expect("thread list", STOPPED_DUE_TO_WATCHPOINT,
            substrs = ['stopped',
                       'stop reason = watchpoint'])

        # Use the '-v' option to do verbose listing of the watchpoint.
        # The hit count should now be 1.
        self.expect("watchpoint list -v",
            substrs = ['hit_count = 1'])

        self.runCmd("thread backtrace all")


if __name__ == '__main__':
    import atexit
    lldb.SBDebugger.Initialize()
    atexit.register(lambda: lldb.SBDebugger.Terminate())
    unittest2.main()
