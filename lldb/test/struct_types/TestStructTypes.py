"""
Test that break on a struct declaration has no effect.

Instead, the first executable statement is set as the breakpoint.
"""

import os, time
import unittest
import lldb
import lldbtest

class TestStructTypes(lldbtest.TestBase):

    mydir = "struct_types"

    def test_struct_types(self):
        """Test that break on a struct declaration has no effect."""
        res = self.res
        exe = os.path.join(os.getcwd(), "a.out")
        self.ci.HandleCommand("file " + exe, res)
        self.assertTrue(res.Succeeded())

        # Break on the ctor function of class C.
        self.ci.HandleCommand("breakpoint set -f main.c -l 14", res)
        self.assertTrue(res.Succeeded())
        self.assertTrue(res.GetOutput().startswith(
            "Breakpoint created: 1: file ='main.c', line = 14, locations = 1"))

        self.ci.HandleCommand("run", res)
        time.sleep(0.1)
        self.assertTrue(res.Succeeded())

        # We should be stopped on the first executable statement within the
        # function where the original breakpoint was attempted.
        self.ci.HandleCommand("thread backtrace", res)
        print "thread backtrace ->", res.GetOutput()
        self.assertTrue(res.Succeeded())
        output = res.GetOutput()
        self.assertTrue(output.find('main.c:20') > 0 and
                        output.find('stop reason = breakpoint') > 0)

        # The breakpoint should have a hit count of 1.
        self.ci.HandleCommand("breakpoint list", res)
        self.assertTrue(res.Succeeded())
        self.assertTrue(res.GetOutput().find(' resolved, hit count = 1') > 0)

        self.ci.HandleCommand("continue", res)
        self.assertTrue(res.Succeeded())


if __name__ == '__main__':
    import atexit
    lldb.SBDebugger.Initialize()
    atexit.register(lambda: lldb.SBDebugger.Terminate())
    unittest.main()
