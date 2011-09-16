"""
Test my first lldb watchpoint.
"""

import os, time
import unittest2
import lldb
from lldbtest import *

class HelloWatchpointTestCase(TestBase):

    mydir = os.path.join("functionalities", "watchpoint", "hello_watchpoint")

    @unittest2.skipUnless(sys.platform.startswith("darwin"), "requires Darwin")
    def test_hello_watchpoint_with_dsym(self):
        """Test a simple sequence of watchpoint creation and watchpoint hit."""
        self.buildDsym(dictionary=self.d)
        self.setTearDownCleanup(dictionary=self.d)
        self.hello_watchpoint()

    def test_hello_watchpoint_with_dwarf(self):
        """Test a simple sequence of watchpoint creation and watchpoint hit."""
        self.buildDwarf(dictionary=self.d)
        self.setTearDownCleanup(dictionary=self.d)
        self.hello_watchpoint()

    def setUp(self):
        # Call super's setUp().
        TestBase.setUp(self)
        # Our simple source filename.
        self.source = 'main.c'
        # Find the line number to break inside main().
        self.line = line_number(self.source, '// Set break point at this line.')
        # And the watchpoint variable declaration line number.
        self.decl = line_number(self.source, '// Watchpoint variable declaration.')
        # Build dictionary to have unique executable names for each test method.
        self.exe_name = self.testMethodName
        self.d = {'C_SOURCES': self.source, 'EXE': self.exe_name}

    def hello_watchpoint(self):
        """Test a simple sequence of watchpoint creation and watchpoint hit."""
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

        # Now let's set a write-type watchpoint for 'global'.
        # There should be only one watchpoint hit (see main.c).
        self.expect("frame variable -w write -g -L global", WATCHPOINT_CREATED,
            substrs = ['Watchpoint created', 'size = 4', 'type = w',
                       '%s:%d' % (self.source, self.decl)])

        self.runCmd("process continue")

        # We should be stopped again due to the watchpoint (write type), but
        # only once.  The stop reason of the thread should be watchpoint.
        self.expect("thread list", STOPPED_DUE_TO_WATCHPOINT,
            substrs = ['stopped',
                       'stop reason = watchpoint'])

        self.runCmd("process continue")
        # Don't expect the read of 'global' to trigger a stop exception.
        # The process status should be 'exited'.
        self.expect("process status",
            substrs = ['exited'])


if __name__ == '__main__':
    import atexit
    lldb.SBDebugger.Initialize()
    atexit.register(lambda: lldb.SBDebugger.Terminate())
    unittest2.main()
