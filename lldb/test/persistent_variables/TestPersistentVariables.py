"""
Test that lldb persistent variables works correctly.
"""

import os, time
import unittest2
import lldb
from lldbtest import *

class TestPersistentVariables(TestBase):

    mydir = "persistent_variables"

    def test_persistent_variables(self):
        """Test that lldb persistent variables works correctly."""
        self.runCmd("file ../array_types/a.out", CURRENT_EXECUTABLE_SET)

        self.runCmd("breakpoint set --name main")

        self.runCmd("run", RUN_SUCCEEDED)

        self.runCmd("expr int $i = 5; $i + 1")
        # $0 = (int)6

        self.runCmd("expr $i + 3")
        # $1 = (int)8

        self.runCmd("expr $1 + $0")
        # $2 = (int)14

        self.expect("expr $2",
            startstr = "$3 = (int) 14")
        # $3 = (int)14


if __name__ == '__main__':
    import atexit
    lldb.SBDebugger.Initialize()
    atexit.register(lambda: lldb.SBDebugger.Terminate())
    unittest2.main()
