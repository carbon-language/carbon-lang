"""
Test error cases for the 'watchpoint set' command to make sure it errors out when necessary.
"""

import os, time
import unittest2
import lldb
from lldbtest import *

class WatchpointSetErrorTestCase(TestBase):

    mydir = os.path.join("functionalities", "watchpoint", "watchpoint_set_command")

    def test_error_cases_with_watchpoint_set(self):
        """Test error cases with the 'watchpoint set' command."""
        self.buildDwarf(dictionary=self.d)
        self.setTearDownCleanup(dictionary=self.d)
        self.error_cases_with_watchpoint_set()

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

    def error_cases_with_watchpoint_set(self):
        """Test error cases with the 'watchpoint set' command."""
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

        # Try some error conditions:

        # No argument is an error.
        self.expect("watchpoint set", error=True,
            startstr = 'error: invalid combination of options for the given command')
        self.runCmd("watchpoint set -v -w read_write", check=False)

        # 'watchpoint set' now takes a mandatory '-v' or '-e' option to
        # indicate watching for either variable or address.
        self.expect("watchpoint set -w write global", error=True,
            startstr = 'error: invalid combination of options for the given command')

        # Wrong size parameter is an error.
        self.expect("watchpoint set -x -128", error=True,
            substrs = ['invalid enumeration value'])


if __name__ == '__main__':
    import atexit
    lldb.SBDebugger.Initialize()
    atexit.register(lambda: lldb.SBDebugger.Terminate())
    unittest2.main()
