"""
Test that lldb command "command source" works correctly.

See also http://llvm.org/viewvc/llvm-project?view=rev&revision=109673.
"""

import os, time
import unittest2
import lldb
from lldbtest import *

class TestCommandSource(TestBase):

    mydir = "command_source"

    def test_command_source(self):
        """Test that lldb command "command source" works correctly."""
        res = self.res

        # Sourcing .lldb in the current working directory, which in turn imports
        # the "my" package that defines the date() function.
        self.ci.HandleCommand("command source .lldb", res)
        self.assertTrue(res.Succeeded(), CMD_MSG('command source .lldb'))

        # Python should evaluate "my.date()" successfully.
        self.ci.HandleCommand("script my.date()", res)
        if (not res.Succeeded()):
            print res.GetError()
        self.assertTrue(res.Succeeded(), CMD_MSG('script my.date()'))


if __name__ == '__main__':
    import atexit
    lldb.SBDebugger.Initialize()
    atexit.register(lambda: lldb.SBDebugger.Terminate())
    unittest2.main()
