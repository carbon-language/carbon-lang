"""
Test that nested persistent types work.
"""

import os, time
import unittest2
import lldb
from lldbtest import *

class NestedPersistentTypesTestCase(TestBase):

    mydir = TestBase.compute_mydir(__file__)

    def test_persistent_types(self):
        """Test that nested persistent types work."""
        self.buildDefault()

        self.runCmd("file a.out", CURRENT_EXECUTABLE_SET)

        self.runCmd("breakpoint set --name main")

        self.runCmd("run", RUN_SUCCEEDED)

        self.runCmd("expression struct $foo { int a; int b; };")

        self.runCmd("expression struct $bar { struct $foo start; struct $foo end; };")

        self.runCmd("expression struct $bar $my_bar = {{ 2, 3 }, { 4, 5 }};")

        self.expect("expression $my_bar",
                    substrs = ['a = 2', 'b = 3', 'a = 4', 'b = 5'])

        self.expect("expression $my_bar.start.b",
                    substrs = ['(int)', '3'])

        self.expect("expression $my_bar.end.b",
                    substrs = ['(int)', '5'])

if __name__ == '__main__':
    import atexit
    lldb.SBDebugger.Initialize()
    atexit.register(lambda: lldb.SBDebugger.Terminate())
    unittest2.main()
