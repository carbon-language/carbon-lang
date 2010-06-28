"""Test lldb help command."""

import lldb
import unittest

class TestHelpCommand(unittest.TestCase):

    def setUp(self):
        self.debugger = lldb.SBDebugger.Create()
        self.debugger.SetAsync(True)
        self.ci = self.debugger.GetCommandInterpreter()
        if not self.ci:
            raise Exception('Could not get the command interpreter')

    def tearDown(self):
        pass

    def test_simplehelp(self):
        """A simple test of 'help' command and its output."""
        res = lldb.SBCommandReturnObject()
        self.ci.HandleCommand("help", res)
        self.assertTrue(res.Succeeded())
        self.assertTrue(res.GetOutput().startswith(
            'The following is a list of built-in, permanent debugger commands'))
        #print res.GetOutput()

    def test_help_should_not_hang_emacsshell(self):
        """'set term-width 0' should not hang the help command."""
        res = lldb.SBCommandReturnObject()
        self.ci.HandleCommand("set term-width 0", res)
        self.assertTrue(res.Succeeded())
        self.ci.HandleCommand("help", res)
        self.assertTrue(res.Succeeded())
        self.assertTrue(res.GetOutput().startswith(
            'The following is a list of built-in, permanent debugger commands'))
        #print res.GetOutput()


if __name__ == '__main__':
    unittest.main()
