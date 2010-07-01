"""Test breakpoint on a class constructor; and variable list the this object."""

import os, time
import lldb
import unittest

class TestClassTypes(unittest.TestCase):

    def setUp(self):
        # Save old working directory.
        self.oldcwd = os.getcwd()
        # Change current working directory if ${LLDB_TEST} is defined.
        if ("LLDB_TEST" in os.environ):
            os.chdir(os.path.join(os.environ["LLDB_TEST"], "class_types"));
        self.dbg = lldb.SBDebugger.Create()
        if not self.dbg.IsValid():
            raise Exception('Invalid debugger instance')
        self.dbg.SetAsync(False)
        self.ci = self.dbg.GetCommandInterpreter()
        if not self.ci:
            raise Exception('Could not get the command interpreter')

    def tearDown(self):
        # Restore old working directory.
        os.chdir(self.oldcwd)

    def test_class_types(self):
        """Test 'variable list this' when stopped on a class constructor."""
        res = lldb.SBCommandReturnObject()
        exe = os.path.join(os.getcwd(), "a.out")
        self.ci.HandleCommand("file " + exe, res)
        time.sleep(0.1)
        self.assertTrue(res.Succeeded())

        # Break on the ctor function of class C.
        self.ci.HandleCommand("breakpoint set -f main.cpp -l 73", res)
        time.sleep(0.1)
        self.assertTrue(res.Succeeded())
        self.assertTrue(res.GetOutput().startswith(
            "Breakpoint created: 1: file ='main.cpp', line = 73, locations = 1"))

        self.ci.HandleCommand("run", res)
        time.sleep(0.1)
        self.assertTrue(res.Succeeded())

        # The breakpoint should have a hit count of 1.
        self.ci.HandleCommand("breakpoint list", res)
        time.sleep(0.1)
        self.assertTrue(res.Succeeded())
        self.assertTrue(res.GetOutput().find('resolved, hit count = 1'))

        # And the stop reason of the thread should be breakpoint.
        self.ci.HandleCommand("thread list", res)
        time.sleep(0.1)
        self.assertTrue(res.Succeeded())
        self.assertTrue(res.GetOutput().find('state is Stopped') and
                        res.GetOutput().find('stop reason = breakpoint'))

        # We should be stopped on the ctor function of class C.
        self.ci.HandleCommand("variable list this", res);
        time.sleep(0.1)
        self.assertTrue(res.Succeeded())
        self.assertTrue(res.GetOutput().startswith('(class C *const) this = '))

        self.ci.HandleCommand("continue", res)
        time.sleep(0.1)
        self.assertTrue(res.Succeeded())


if __name__ == '__main__':
    lldb.SBDebugger.Initialize()
    unittest.main()
    lldb.SBDebugger.Terminate()
