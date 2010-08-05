"""Test breakpoint on a class constructor; and variable list the this object."""

import os, time
import unittest2
import lldb
import lldbtest

class TestClassTypes(lldbtest.TestBase):

    mydir = "class_types"

    def test_class_types(self):
        """Test 'variable list this' when stopped on a class constructor."""
        res = self.res
        exe = os.path.join(os.getcwd(), "a.out")
        self.ci.HandleCommand("file " + exe, res)
        self.assertTrue(res.Succeeded())

        # Break on the ctor function of class C.
        self.ci.HandleCommand("breakpoint set -f main.cpp -l 73", res)
        self.assertTrue(res.Succeeded())
        self.assertTrue(res.GetOutput().startswith(
            "Breakpoint created: 1: file ='main.cpp', line = 73, locations = 1"))

        self.ci.HandleCommand("run", res)
        time.sleep(0.1)
        self.assertTrue(res.Succeeded())

        # The stop reason of the thread should be breakpoint.
        self.ci.HandleCommand("thread list", res)
        print "thread list ->", res.GetOutput()
        self.assertTrue(res.Succeeded())
        self.assertTrue(res.GetOutput().find('state is Stopped') > 0 and
                        res.GetOutput().find('stop reason = breakpoint') > 0)

        # The breakpoint should have a hit count of 1.
        self.ci.HandleCommand("breakpoint list", res)
        self.assertTrue(res.Succeeded())
        self.assertTrue(res.GetOutput().find(' resolved, hit count = 1') > 0)

        # We should be stopped on the ctor function of class C.
        self.ci.HandleCommand("variable list this", res);
        self.assertTrue(res.Succeeded())
        self.assertTrue(res.GetOutput().startswith('(class C *const) this = '))

        self.ci.HandleCommand("continue", res)
        self.assertTrue(res.Succeeded())


if __name__ == '__main__':
    import atexit
    lldb.SBDebugger.Initialize()
    atexit.register(lambda: lldb.SBDebugger.Terminate())
    unittest2.main()
