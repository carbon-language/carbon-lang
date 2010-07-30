"""
Test that lldb command "command source" works correctly.
"""

import os, time
import unittest
import lldb
import lldbtest

class TestCommandSource(lldbtest.TestBase):

    mydir = "command_source"

    def test_command_source(self):
        """Test that lldb command "command source" works correctly."""
        res = self.res

        # Sourcing .lldb in the current working directory, which in turn imports
        # the "my" package that defines the date() function.
        self.ci.HandleCommand("command source .lldb", res)
        self.assertTrue(res.Succeeded())

        self.ci.HandleCommand("script my.date()", res)
        if (not res.Succeeded()):
            print res.GetError()
        self.assertTrue(res.Succeeded())

        time.sleep(1)

if __name__ == '__main__':
    lldb.SBDebugger.Initialize()
    unittest.main()
    lldb.SBDebugger.Terminate()
