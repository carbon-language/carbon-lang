"""Test breakpoint by file/line number; and list variables with array types."""

import os, time
import unittest2
import lldb
from lldbtest import *

class TestArrayTypes(TestBase):

    mydir = "array_types"

    def test_array_types(self):
        """Test 'variable list var_name' on some variables with array types."""
        exe = os.path.join(os.getcwd(), "a.out")
        self.runCmd("file " + exe, CURRENT_EXECUTABLE_SET)

        # Break on line 42 inside main().
        self.expect("breakpoint set -f main.c -l 42", BREAKPOINT_CREATED,
            startstr = "Breakpoint created: 1: file ='main.c', line = 42, locations = 1")

        self.runCmd("run", RUN_STOPPED)

        # The stop reason of the thread should be breakpoint.
        self.expect("thread list", STOPPED_DUE_TO_BREAKPOINT,
            substrs = ['state is Stopped',
                       'stop reason = breakpoint'])

        # The breakpoint should have a hit count of 1.
        self.expect("breakpoint list", BREAKPOINT_HIT_ONCE,
            substrs = ['resolved, hit count = 1'])

        # Issue 'variable list' command on several array-type variables.

        self.expect("variable list strings", VARIABLES_DISPLAYED_CORRECTLY,
            startstr = '(char *[4])',
            substrs = ['(char *) strings[0]',
                       '(char *) strings[1]',
                       '(char *) strings[2]',
                       '(char *) strings[3]',
                       'Hello',
                       'Hola',
                       'Bonjour',
                       'Guten Tag'])

        self.expect("variable list char_16", VARIABLES_DISPLAYED_CORRECTLY,
            substrs = ['(char) char_16[0]',
                       '(char) char_16[15]'])

        self.expect("variable list ushort_matrix", VARIABLES_DISPLAYED_CORRECTLY,
            startstr = '(unsigned short [2][3])')

        self.expect("variable list long_6", VARIABLES_DISPLAYED_CORRECTLY,
            startstr = '(long [6])')


if __name__ == '__main__':
    import atexit
    lldb.SBDebugger.Initialize()
    atexit.register(lambda: lldb.SBDebugger.Terminate())
    unittest2.main()
