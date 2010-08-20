"""Test settings and readings of program variables."""

import os, time
import unittest2
import lldb
from lldbtest import *

class TestSetValues(TestBase):

    mydir = "set_values"

    def test_set_values(self):
        """Test settings and readings of program variables."""
        exe = os.path.join(os.getcwd(), "a.out")
        self.runCmd("file " + exe, CURRENT_EXECUTABLE_SET)

        # Set breakpoints on several places to set program variables.
        self.expect("breakpoint set -f main.c -l 15", BREAKPOINT_CREATED,
            startstr = "Breakpoint created: 1: file ='main.c', line = 15, locations = 1")

        self.expect("breakpoint set -f main.c -l 36", BREAKPOINT_CREATED,
            startstr = "Breakpoint created: 2: file ='main.c', line = 36, locations = 1")

        self.expect("breakpoint set -f main.c -l 57", BREAKPOINT_CREATED,
            startstr = "Breakpoint created: 3: file ='main.c', line = 57, locations = 1")

        self.expect("breakpoint set -f main.c -l 78", BREAKPOINT_CREATED,
            startstr = "Breakpoint created: 4: file ='main.c', line = 78, locations = 1")

        self.expect("breakpoint set -f main.c -l 85", BREAKPOINT_CREATED,
            startstr = "Breakpoint created: 5: file ='main.c', line = 85, locations = 1")

        self.runCmd("run", RUN_STOPPED)

        # The stop reason of the thread should be breakpoint.
        self.expect("thread list", STOPPED_DUE_TO_BREAKPOINT,
            substrs = ['state is Stopped',
                       'stop reason = breakpoint'])

        # The breakpoint should have a hit count of 1.
        self.expect("breakpoint list", BREAKPOINT_HIT_ONCE,
            substrs = [' resolved, hit count = 1'])

        # main.c:15
        # Check that 'variable list' displays the correct data type and value.
        self.expect("variable list", VARIABLES_DISPLAYED_CORRECTLY,
            startstr = "i = (char) 'a'")

        # TODO:
        # Now set variable 'i' and check that it is correctly displayed.

        self.runCmd("continue")

        # main.c:36
        # Check that 'variable list' displays the correct data type and value.
        self.expect("variable list", VARIABLES_DISPLAYED_CORRECTLY,
            startstr = "i = (short unsigned int) 0x0021")

        # TODO:
        # Now set variable 'i' and check that it is correctly displayed.

        self.runCmd("continue")

        # main.c:57
        # Check that 'variable list' displays the correct data type and value.
        self.expect("variable list", VARIABLES_DISPLAYED_CORRECTLY,
            startstr = "i = (long int) 33")

        # TODO:
        # Now set variable 'i' and check that it is correctly displayed.

        self.runCmd("continue")

        # main.c:78
        # Check that 'variable list' displays the correct data type and value.
        self.expect("variable list", VARIABLES_DISPLAYED_CORRECTLY,
            startstr = "i = (double) 3.14159")

        # TODO:
        # Now set variable 'i' and check that it is correctly displayed.

        self.runCmd("continue")

        # main.c:85
        # Check that 'variable list' displays the correct data type and value.
        self.expect("variable list", VARIABLES_DISPLAYED_CORRECTLY,
            startstr = "i = (long double) ")

        # TODO:
        # Now set variable 'i' and check that it is correctly displayed.


if __name__ == '__main__':
    import atexit
    lldb.SBDebugger.Initialize()
    atexit.register(lambda: lldb.SBDebugger.Terminate())
    unittest2.main()
