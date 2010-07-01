"""Test lldb help command."""

import os, time
import lldb
import unittest

main = False

class TestHelpCommand(unittest.TestCase):

    def setUp(self):
        global main

        # Save old working directory.
        self.oldcwd = os.getcwd()
        # Change current working directory if ${LLDB_TEST} is defined.
        if ("LLDB_TEST" in os.environ):
            os.chdir(os.path.join(os.environ["LLDB_TEST"], "help"));
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

    def test_simplehelp(self):
        """A simple test of 'help' command and its output."""
        res = lldb.SBCommandReturnObject()
        self.ci.HandleCommand("help", res)
        time.sleep(0.1)
        self.assertTrue(res.Succeeded())
        self.assertTrue(res.GetOutput().startswith(
            'The following is a list of built-in, permanent debugger commands'))

    def test_help_should_not_hang_emacsshell(self):
        """Command 'set term-width 0' should not hang the help command."""
        res = lldb.SBCommandReturnObject()
        self.ci.HandleCommand("set term-width 0", res)
        time.sleep(0.1)
        self.assertTrue(res.Succeeded())
        self.ci.HandleCommand("help", res)
        time.sleep(0.1)
        self.assertTrue(res.Succeeded())
        self.assertTrue(res.GetOutput().startswith(
            'The following is a list of built-in, permanent debugger commands'))


if __name__ == '__main__':
    lldb.SBDebugger.Initialize()
    main = True
    unittest.main()
    lldb.SBDebugger.Terminate()
