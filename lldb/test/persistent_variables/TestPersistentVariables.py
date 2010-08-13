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
        res = self.res

        self.ci.HandleCommand("file ../array_types/a.out", res)
        self.assertTrue(res.Succeeded(), CURRENT_EXECUTABLE_SET)

        self.ci.HandleCommand("breakpoint set --name main", res)
        self.assertTrue(res.Succeeded())

        self.ci.HandleCommand("run", res)
        self.assertTrue(res.Succeeded(), RUN_STOPPED)

        self.ci.HandleCommand("expr int $i = 5; $i + 1", res)
        self.assertTrue(res.Succeeded(), CMD_MSG('expr int $i = 5; $i + 1'))
        #print res.GetOutput()
        # $0 = (int)6

        self.ci.HandleCommand("expr $i + 3", res)
        self.assertTrue(res.Succeeded(), CMD_MSG('expr $i + 3'))
        #print res.GetOutput()
        # $1 = (int)8

        self.ci.HandleCommand("expr $1 + $0", res)
        self.assertTrue(res.Succeeded(), CMD_MSG('expr $1 + $0'))
        #print res.GetOutput()
        # $2 = (int)14

        self.ci.HandleCommand("expr $2", res)
        self.assertTrue(res.Succeeded() and
                        res.GetOutput().startswith("$3 = (int) 14"),
                        CMD_MSG('expr $2'))
        #print res.GetOutput()
        # $3 = (int)14

        self.ci.HandleCommand("continue", res)
        self.ci.HandleCommand("quit", res)


if __name__ == '__main__':
    import atexit
    lldb.SBDebugger.Initialize()
    atexit.register(lambda: lldb.SBDebugger.Terminate())
    unittest2.main()
