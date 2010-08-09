"""Show global variables and check that they do indeed have global scopes."""

import os, time
import unittest2
import lldb
from lldbtest import *

class TestGlobalVariables(TestBase):

    mydir = "global_variables"

    def test_global_variables(self):
        """Test 'variable list -s -a' which omits args and shows scopes."""
        res = self.res
        exe = os.path.join(os.getcwd(), "a.out")
        self.ci.HandleCommand("file " + exe, res)
        self.assertTrue(res.Succeeded(), CURRENT_EXECUTABLE_SET)

        # Break inside the main.
        self.ci.HandleCommand("breakpoint set -f main.c -l 20", res)
        self.assertTrue(res.Succeeded())
        self.assertTrue(res.GetOutput().startswith(
            "Breakpoint created: 1: file ='main.c', line = 20, locations = 1"),
                        BREAKPOINT_CREATED)

        self.ci.HandleCommand("run", res)
        #time.sleep(0.1)
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
        self.assertTrue(res.Succeeded())
        self.assertTrue(res.GetOutput().find(' resolved, hit count = 1') > 0,
                        BREAKPOINT_HIT_ONCE)

        # Check that GLOBAL scopes are indicated for the variables.
        self.ci.HandleCommand("variable list -s -a", res);
        self.assertTrue(res.Succeeded())
        output = res.GetOutput()
        self.assertTrue(output.find('GLOBAL: g_file_static_cstr') > 0 and
                        output.find('g_file_static_cstr') > 0 and
                        output.find('GLOBAL: g_file_global_int') > 0 and
                        output.find('(int) 42') > 0 and
                        output.find('GLOBAL: g_file_global_cstr') > 0 and
                        output.find('g_file_global_cstr') > 0,
                        VARIABLES_DISPLAYED_CORRECTLY)

        self.ci.HandleCommand("continue", res)
        self.assertTrue(res.Succeeded())


if __name__ == '__main__':
    import atexit
    lldb.SBDebugger.Initialize()
    atexit.register(lambda: lldb.SBDebugger.Terminate())
    unittest2.main()
