"""Test settings and readings of program variables."""

import os, time
import unittest2
import lldb
from lldbtest import *

class TestSetValues(TestBase):

    mydir = "set_values"

    def test_set_values(self):
        """Test settings and readings of program variables."""
        res = self.res
        exe = os.path.join(os.getcwd(), "a.out")
        self.ci.HandleCommand("file " + exe, res)
        self.assertTrue(res.Succeeded(), CURRENT_EXECUTABLE_SET)

        # Set breakpoints on several places to set program variables.
        self.ci.HandleCommand("breakpoint set -f main.c -l 15", res)
        self.assertTrue(res.Succeeded())
        self.assertTrue(res.GetOutput().startswith(
            "Breakpoint created: 1: file ='main.c', line = 15, locations = 1"),
                        BREAKPOINT_CREATED)
        self.ci.HandleCommand("breakpoint set -f main.c -l 36", res)
        self.assertTrue(res.Succeeded())
        self.assertTrue(res.GetOutput().startswith(
            "Breakpoint created: 2: file ='main.c', line = 36, locations = 1"),
                        BREAKPOINT_CREATED)
        self.ci.HandleCommand("breakpoint set -f main.c -l 57", res)
        self.assertTrue(res.Succeeded())
        self.assertTrue(res.GetOutput().startswith(
            "Breakpoint created: 3: file ='main.c', line = 57, locations = 1"),
                        BREAKPOINT_CREATED)
        self.ci.HandleCommand("breakpoint set -f main.c -l 78", res)
        self.assertTrue(res.Succeeded())
        self.assertTrue(res.GetOutput().startswith(
            "Breakpoint created: 4: file ='main.c', line = 78, locations = 1"),
                        BREAKPOINT_CREATED)
        self.ci.HandleCommand("breakpoint set -f main.c -l 85", res)
        self.assertTrue(res.Succeeded())
        self.assertTrue(res.GetOutput().startswith(
            "Breakpoint created: 5: file ='main.c', line = 85, locations = 1"),
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

        # main.c:15
        # Check that 'variable list' displays the correct data type and value.
        self.ci.HandleCommand("variable list", res);
        self.assertTrue(res.Succeeded())
        self.assertTrue(res.GetOutput().startswith("i = (char) 'a'"),
                        VARIABLES_DISPLAYED_CORRECTLY)
        # TODO:
        # Now set variable 'i' and check that it is correctly displayed.

        self.ci.HandleCommand("continue", res)
        self.assertTrue(res.Succeeded())

        # main.c:36
        # Check that 'variable list' displays the correct data type and value.
        self.ci.HandleCommand("variable list", res);
        self.assertTrue(res.Succeeded())
        self.assertTrue(res.GetOutput().startswith(
                "i = (short unsigned int) 0x0021"),
                        VARIABLES_DISPLAYED_CORRECTLY)
        # TODO:
        # Now set variable 'i' and check that it is correctly displayed.

        self.ci.HandleCommand("continue", res)
        self.assertTrue(res.Succeeded())

        # main.c:57
        # Check that 'variable list' displays the correct data type and value.
        self.ci.HandleCommand("variable list", res);
        self.assertTrue(res.Succeeded())
        self.assertTrue(res.GetOutput().startswith("i = (long int) 33"),
                        VARIABLES_DISPLAYED_CORRECTLY)
        # TODO:
        # Now set variable 'i' and check that it is correctly displayed.

        self.ci.HandleCommand("continue", res)
        self.assertTrue(res.Succeeded())

        # main.c:78
        # Check that 'variable list' displays the correct data type and value.
        self.ci.HandleCommand("variable list", res);
        self.assertTrue(res.Succeeded())
        self.assertTrue(res.GetOutput().startswith("i = (double) 3.14159"),
                        VARIABLES_DISPLAYED_CORRECTLY)
        # TODO:
        # Now set variable 'i' and check that it is correctly displayed.

        self.ci.HandleCommand("continue", res)
        self.assertTrue(res.Succeeded())

        # main.c:85
        # Check that 'variable list' displays the correct data type and value.
        self.ci.HandleCommand("variable list", res);
        self.assertTrue(res.Succeeded())
        self.assertTrue(res.GetOutput().startswith("i = (long double) "),
                        VARIABLES_DISPLAYED_CORRECTLY)
        # TODO:
        # Now set variable 'i' and check that it is correctly displayed.

        self.ci.HandleCommand("continue", res)
        self.assertTrue(res.Succeeded())


if __name__ == '__main__':
    import atexit
    lldb.SBDebugger.Initialize()
    atexit.register(lambda: lldb.SBDebugger.Terminate())
    unittest2.main()
