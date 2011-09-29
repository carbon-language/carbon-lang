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
        self.first_stop = line_number(self.source, '// Set break point at this line')
        self.thread_function = line_number(self.source, '// Break here in order to allow the thread')
        # Build dictionary to have unique executable names for each test method.
        self.exe_name = self.testMethodName
        self.d = {'CXX_SOURCES': self.source, 'EXE': self.exe_name}

    def hello_multiple_threads(self):
        """Test that lldb watchpoint works for multiple threads."""
        exe = os.path.join(os.getcwd(), self.exe_name)
        self.runCmd("file " + exe, CURRENT_EXECUTABLE_SET)

        # Add a breakpoint to set a watchpoint when stopped on the breakpoint.
        self.expect("breakpoint set -l %d" % self.first_stop, BREAKPOINT_CREATED,
            startstr = "Breakpoint created: 1: file ='%s', line = %d, locations = 1" %
                       (self.source, self.first_stop))

        # Set this breakpoint to allow newly created thread to inherit the global watchpoint state.
        self.expect("breakpoint set -l %d" % self.thread_function, BREAKPOINT_CREATED,
            startstr = "Breakpoint created: 2: file ='%s', line = %d, locations = 1" %
                       (self.source, self.thread_function))

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

        breakpoint_stops = 0
        while True:
            self.runCmd("process continue")

            self.runCmd("thread list")
            if "stop reason = breakpoint" in self.res.GetOutput():
                breakpoint_stops += 1
                if breakpoint_stops > 3:
                    self.fail("Do not expect to break more than 3 times")
                continue
            elif "stop reason = watchpoint" in self.res.GetOutput():
                # Good, we verified that the watchpoint works!
                self.runCmd("thread backtrace all")
                break
            else:
                self.fail("The stop reason should be either break or watchpoint")

        # Use the '-v' option to do verbose listing of the watchpoint.
        # The hit count should now be 1.
        self.expect("watchpoint list -v",
            substrs = ['hit_count = 1'])


if __name__ == '__main__':
    import atexit
    lldb.SBDebugger.Initialize()
    atexit.register(lambda: lldb.SBDebugger.Terminate())
    unittest2.main()
