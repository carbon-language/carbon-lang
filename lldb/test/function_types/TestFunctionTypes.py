"""Test variable with function ptr type and that break on the function works."""

import os, time
import unittest2
import lldb
from lldbtest import *

class TestFunctionTypes(TestBase):

    mydir = "function_types"

    def test_function_types(self):
        """Test 'callback' has function ptr type, then break on the function."""
        exe = os.path.join(os.getcwd(), "a.out")
        self.runCmd("file " + exe, CURRENT_EXECUTABLE_SET)

        # Break inside the main.
        self.expect("breakpoint set -f main.c -l 21", BREAKPOINT_CREATED,
            startstr = "Breakpoint created: 1: file ='main.c', line = 21, locations = 1")

        self.runCmd("run", RUN_SUCCEEDED)

        # The stop reason of the thread should be breakpoint.
        self.expect("thread list", STOPPED_DUE_TO_BREAKPOINT,
            substrs = ['state is Stopped',
                       'stop reason = breakpoint'])

        # The breakpoint should have a hit count of 1.
        self.expect("breakpoint list", BREAKPOINT_HIT_ONCE,
            substrs = [' resolved, hit count = 1'])

        # Check that the 'callback' variable display properly.
        self.expect("variable list callback", VARIABLES_DISPLAYED_CORRECTLY,
            startstr = '(int (*)(char const *)) callback =')

        # And that we can break on the callback function.
        self.runCmd("breakpoint set -n string_not_empty", BREAKPOINT_CREATED)
        self.runCmd("continue")

        # Check that we do indeed stop on the string_not_empty function.
        self.expect("process status", STOPPED_DUE_TO_BREAKPOINT,
            substrs = ['where = a.out`string_not_empty',
                       'main.c:12',
                       'stop reason = breakpoint'])


if __name__ == '__main__':
    import atexit
    lldb.SBDebugger.Initialize()
    atexit.register(lambda: lldb.SBDebugger.Terminate())
    unittest2.main()
