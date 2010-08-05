"""Test variable with function ptr type and that break on the function works."""

import os, time
import unittest2
import lldb
import lldbtest

class TestFunctionTypes(lldbtest.TestBase):

    mydir = "function_types"

    def test_function_types(self):
        """Test 'callback' has function ptr type, then break on the function."""
        res = self.res
        exe = os.path.join(os.getcwd(), "a.out")
        self.ci.HandleCommand("file " + exe, res)
        self.assertTrue(res.Succeeded())

        # Break inside the main.
        self.ci.HandleCommand("breakpoint set -f main.c -l 21", res)
        self.assertTrue(res.Succeeded())
        self.assertTrue(res.GetOutput().startswith(
            "Breakpoint created: 1: file ='main.c', line = 21, locations = 1"))

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

        # Check that the 'callback' variable display properly.
        self.ci.HandleCommand("variable list callback", res);
        self.assertTrue(res.Succeeded())
        output = res.GetOutput()
        self.assertTrue(output.startswith('(int (*)(char const *)) callback ='))

        # And that we can break on the callback function.
        self.ci.HandleCommand("breakpoint set -n string_not_empty", res);
        self.assertTrue(res.Succeeded())
        self.ci.HandleCommand("continue", res)
        self.assertTrue(res.Succeeded())

        # Check that we do indeed stop on the string_not_empty function.
        self.ci.HandleCommand("process status", res)
        self.assertTrue(res.Succeeded())
        output = res.GetOutput()
        #print "process status =", output
        self.assertTrue(output.find('where = a.out`string_not_empty') > 0 and
                        output.find('main.c:12') > 0 and
                        output.find('stop reason = breakpoint') > 0)

        self.ci.HandleCommand("continue", res)
        self.assertTrue(res.Succeeded())


if __name__ == '__main__':
    import atexit
    lldb.SBDebugger.Initialize()
    atexit.register(lambda: lldb.SBDebugger.Terminate())
    unittest2.main()
