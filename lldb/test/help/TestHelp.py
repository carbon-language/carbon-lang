"""
Test lldb help command.

See also CommandInterpreter::OutputFormattedHelpText().
"""

import os, time
import unittest2
import lldb
from lldbtest import *

class TestHelpCommand(TestBase):

    mydir = "help"

    def test_simplehelp(self):
        """A simple test of 'help' command and its output."""
        self.expect("help",
            startstr = 'The following is a list of built-in, permanent debugger commands')

    def test_help_should_not_hang_emacsshell(self):
        """Command 'set term-width 0' should not hang the help command."""
        self.runCmd("set term-width 0")
        self.expect("help",
            startstr = 'The following is a list of built-in, permanent debugger commands')


if __name__ == '__main__':
    import atexit
    lldb.SBDebugger.Initialize()
    atexit.register(lambda: lldb.SBDebugger.Terminate())
    unittest2.main()
