"""Test variable with function ptr type and that break on the function works."""

import os, time
import unittest2
import lldb
from lldbtest import *

class TestFunctionTypes(TestBase):

    mydir = "function_types"

    def test_function_types(self):
        """Test 'callback' has function ptr type, then break on the function."""
        res = self.res
        exe = os.path.join(os.getcwd(), "a.out")
        self.ci.HandleCommand("file " + exe, res)
        self.assertTrue(res.Succeeded(), CURRENT_EXECUTABLE_SET)

        # Break inside the main.
        self.ci.HandleCommand("breakpoint set -f main.c -l 21", res)
        self.assertTrue(res.Succeeded(), CMD_MSG('breakpoint set'))
        self.assertTrue(res.GetOutput().startswith(
            "Breakpoint created: 1: file ='main.c', line = 21, locations = 1"),
                        BREAKPOINT_CREATED)

        self.ci.HandleCommand("run", res)
        self.runStarted = True
        self.assertTrue(res.Succeeded(), RUN_STOPPED)

        # The stop reason of the thread should be breakpoint.
        self.ci.HandleCommand("thread list", res)
        #print "thread list ->", res.GetOutput()
        self.assertTrue(res.Succeeded(), CMD_MSG('thread list'))
        self.assertTrue(res.GetOutput().find('state is Stopped') > 0 and
                        res.GetOutput().find('stop reason = breakpoint') > 0,
                        STOPPED_DUE_TO_BREAKPOINT)

        # The breakpoint should have a hit count of 1.
        self.ci.HandleCommand("breakpoint list", res)
        self.assertTrue(res.Succeeded(), CMD_MSG('breakpoint list'))
        self.assertTrue(res.GetOutput().find(' resolved, hit count = 1') > 0,
                        BREAKPOINT_HIT_ONCE)

        # Check that the 'callback' variable display properly.
        self.ci.HandleCommand("variable list callback", res);
        self.assertTrue(res.Succeeded(), CMD_MSG('variable list ...'))
        output = res.GetOutput()
        self.assertTrue(output.startswith('(int (*)(char const *)) callback ='),
                        VARIABLES_DISPLAYED_CORRECTLY)

        # And that we can break on the callback function.
        self.ci.HandleCommand("breakpoint set -n string_not_empty", res);
        self.assertTrue(res.Succeeded(), BREAKPOINT_CREATED)
        self.ci.HandleCommand("continue", res)
        self.assertTrue(res.Succeeded(), CMD_MSG('continue'))

        # Check that we do indeed stop on the string_not_empty function.
        self.ci.HandleCommand("process status", res)
        self.assertTrue(res.Succeeded(), CMD_MSG('process status'))
        output = res.GetOutput()
        #print "process status =", output
        self.assertTrue(output.find('where = a.out`string_not_empty') > 0 and
                        output.find('main.c:12') > 0 and
                        output.find('stop reason = breakpoint') > 0,
                        STOPPED_DUE_TO_BREAKPOINT)


if __name__ == '__main__':
    import atexit
    lldb.SBDebugger.Initialize()
    atexit.register(lambda: lldb.SBDebugger.Terminate())
    unittest2.main()
