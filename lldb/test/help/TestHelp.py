"""
Test some lldb help commands.

See also CommandInterpreter::OutputFormattedHelpText().
"""

import os, time
import unittest2
import lldb
from lldbtest import *

class HelpCommandTestCase(TestBase):

    mydir = "help"

    def test_simplehelp(self):
        """A simple test of 'help' command and its output."""
        self.expect("help",
            startstr = 'The following is a list of built-in, permanent debugger commands')

    def test_help_should_not_hang_emacsshell(self):
        """Command 'settings set term-width 0' should not hang the help command."""
        self.runCmd("settings set term-width 0")
        self.expect("help",
            startstr = 'The following is a list of built-in, permanent debugger commands')

    def test_help_image_dump_symtab_should_not_crash(self):
        """Command 'help image dump symtab' should not crash lldb."""
        self.expect("help image dump symtab",
            substrs = ['image dump symtab',
                       'sort-order'])


if __name__ == '__main__':
    import atexit
    lldb.SBDebugger.Initialize()
    atexit.register(lambda: lldb.SBDebugger.Terminate())
    unittest2.main()
