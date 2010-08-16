"""Test breakpoint by file/line number; and list variables with array types."""

import os, time
import unittest2
import lldb
from lldbtest import *

class TestArrayTypes(TestBase):

    mydir = "array_types"

    def test_array_types(self):
        """Test 'variable list var_name' on some variables with array types."""
        res = self.res
        exe = os.path.join(os.getcwd(), "a.out")
        self.ci.HandleCommand("file " + exe, res)
        self.assertTrue(res.Succeeded(), CURRENT_EXECUTABLE_SET)

        # Break on line 42 inside main().
        self.ci.HandleCommand("breakpoint set -f main.c -l 42", res)
        self.assertTrue(res.Succeeded())
        self.assertTrue(res.GetOutput().startswith(
            "Breakpoint created: 1: file ='main.c', line = 42, locations = 1"),
                        BREAKPOINT_CREATED)

        self.ci.HandleCommand("run", res)
        self.runStarted = True
        self.assertTrue(res.Succeeded(), RUN_STOPPED)

        # The stop reason of the thread should be breakpoint.
        self.ci.HandleCommand("thread list", res)
        #print "thread list ->", res.GetOutput()
        self.assertTrue(res.Succeeded())
        self.assertTrue(res.GetOutput().find('state is Stopped') > 0 and
                        res.GetOutput().find('stop reason = breakpoint') > 0,
                        STOPPED_DUE_TO_BREAKPOINT)

        # The breakpoint should have a hit count of 1.
        self.ci.HandleCommand("breakpoint list", res)
        self.assertTrue(res.Succeeded())
        self.assertTrue(res.GetOutput().find('resolved, hit count = 1') > 0,
                        BREAKPOINT_HIT_ONCE)

        # Issue 'variable list' command on several array-type variables.

        self.ci.HandleCommand("variable list strings", res);
        self.assertTrue(res.Succeeded())
        output = res.GetOutput()
        self.assertTrue(output.startswith('(char *[4])') and
                        output.find('(char *) strings[0]') > 0 and
                        output.find('(char *) strings[1]') > 0 and
                        output.find('(char *) strings[2]') > 0 and
                        output.find('(char *) strings[3]') > 0 and
                        output.find('Hello') > 0 and
                        output.find('Hola') > 0 and
                        output.find('Bonjour') > 0 and
                        output.find('Guten Tag') > 0,
                        VARIABLES_DISPLAYED_CORRECTLY)

        self.ci.HandleCommand("variable list char_16", res);
        self.assertTrue(res.Succeeded())
        self.assertTrue(res.GetOutput().find('(char) char_16[0]') > 0 and
                        res.GetOutput().find('(char) char_16[15]') > 0,
                        VARIABLES_DISPLAYED_CORRECTLY)

        self.ci.HandleCommand("variable list ushort_matrix", res);
        self.assertTrue(res.Succeeded())
        self.assertTrue(res.GetOutput().startswith('(unsigned short [2][3])'),
                        VARIABLES_DISPLAYED_CORRECTLY)

        self.ci.HandleCommand("variable list long_6", res);
        self.assertTrue(res.Succeeded())
        self.assertTrue(res.GetOutput().startswith('(long [6])'),
                        VARIABLES_DISPLAYED_CORRECTLY)


if __name__ == '__main__':
    import atexit
    lldb.SBDebugger.Initialize()
    atexit.register(lambda: lldb.SBDebugger.Terminate())
    unittest2.main()
