"""Test variable with function ptr type and that break on the function works."""

import os, time
import lldb
import unittest

main = False

class TestClassTypes(unittest.TestCase):

    def setUp(self):
        global main

        # Save old working directory.
        self.oldcwd = os.getcwd()
        # Change current working directory if ${LLDB_TEST} is defined.
        if ("LLDB_TEST" in os.environ):
            os.chdir(os.path.join(os.environ["LLDB_TEST"], "function_types"));
        self.dbg = lldb.SBDebugger.Create() if main else lldb.DBG
        if not self.dbg.IsValid():
            raise Exception('Invalid debugger instance')
        self.dbg.SetAsync(False)
        self.ci = self.dbg.GetCommandInterpreter()
        if not self.ci:
            raise Exception('Could not get the command interpreter')

    def tearDown(self):
        # Restore old working directory.
        os.chdir(self.oldcwd)
        del self.dbg

    def test_function_types(self):
        """Test 'callback' has function ptr type, then break on the function."""
        res = lldb.SBCommandReturnObject()
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
        self.assertTrue(res.GetOutput().find('state is Stopped') and
                        res.GetOutput().find('stop reason = breakpoint'))

        # The breakpoint should have a hit count of 1.
        self.ci.HandleCommand("breakpoint list", res)
        self.assertTrue(res.Succeeded())
        self.assertTrue(res.GetOutput().find(' resolved, hit count = 1'))

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
        self.assertTrue(output.find('where = a.out`string_not_empty') and
                        output.find('main.c:12') and
                        output.find('stop reason = breakpoint'))

        self.ci.HandleCommand("continue", res)
        self.assertTrue(res.Succeeded())


if __name__ == '__main__':
    lldb.SBDebugger.Initialize()
    main = True
    unittest.main()
    lldb.SBDebugger.Terminate()
