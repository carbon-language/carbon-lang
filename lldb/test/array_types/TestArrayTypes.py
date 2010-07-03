"""Test breakpoint by file/line number; and list variables with array types."""

import os, time
import unittest
import lldb
import lldbtest

class TestArrayTypes(lldbtest.TestBase):

    mydir = "array_types"

    def test_array_types(self):
        """Test 'variable list var_name' on some variables with array types."""
        res = self.res
        exe = os.path.join(os.getcwd(), "a.out")
        self.ci.HandleCommand("file " + exe, res)
        self.assertTrue(res.Succeeded())

        # Break on line 42 inside main().
        self.ci.HandleCommand("breakpoint set -f main.c -l 42", res)
        self.assertTrue(res.Succeeded())
        self.assertTrue(res.GetOutput().startswith(
            "Breakpoint created: 1: file ='main.c', line = 42, locations = 1"))

        self.ci.HandleCommand("run", res)
        time.sleep(0.1)
        self.assertTrue(res.Succeeded())

        # The stop reason of the thread should be breakpoint.
        self.ci.HandleCommand("thread list", res)
        print "thread list ->", res.GetOutput()
        self.assertTrue(res.Succeeded())
        self.assertTrue(res.GetOutput().find('state is Stopped') and
                        res.GetOutput().find('stop reason = breakpoint'))

        # The breakpoint should have a hit count of 1.
        self.ci.HandleCommand("breakpoint list", res)
        self.assertTrue(res.Succeeded())
        self.assertTrue(res.GetOutput().find('resolved, hit count = 1'))

        # Issue 'variable list' command on several array-type variables.

        self.ci.HandleCommand("variable list strings", res);
        self.assertTrue(res.Succeeded())
        output = res.GetOutput()
        self.assertTrue(output.startswith('(char *[4])') and
                        output.find('(char *) strings[0]') and
                        output.find('(char *) strings[1]') and
                        output.find('(char *) strings[2]') and
                        output.find('(char *) strings[3]') and
                        output.find('Hello') and
                        output.find('Hola') and
                        output.find('Bonjour') and
                        output.find('Guten Tag'))

        self.ci.HandleCommand("variable list char_16", res);
        self.assertTrue(res.Succeeded())
        self.assertTrue(res.GetOutput().find('(char) char_16[0]') and
                        res.GetOutput().find('(char) char_16[15]'))

        self.ci.HandleCommand("variable list ushort_matrix", res);
        self.assertTrue(res.Succeeded())
        self.assertTrue(res.GetOutput().startswith('(unsigned short [2][3])'))

        self.ci.HandleCommand("variable list long_6", res);
        self.assertTrue(res.Succeeded())
        self.assertTrue(res.GetOutput().startswith('(long [6])'))

        self.ci.HandleCommand("continue", res)
        self.assertTrue(res.Succeeded())


if __name__ == '__main__':
    lldb.SBDebugger.Initialize()
    unittest.main()
    lldb.SBDebugger.Terminate()
