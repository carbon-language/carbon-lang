"""
Test _regexp-break command which uses regular expression matching to dispatch to other built in breakpoint commands.
"""

import os, time
import unittest2
import lldb
from lldbtest import *

class RegexpBreakCommandTestCase(TestBase):

    mydir = os.path.join("functionalities", "breakpoint", "breakpoint_command")

    @unittest2.skipUnless(sys.platform.startswith("darwin"), "requires Darwin")
    @dsym_test
    def test_with_dsym(self):
        """Test _regexp-break command."""
        self.buildDsym()
        self.regexp_break_command()

    @dwarf_test
    def test_with_dwarf(self):
        """Test _regexp-break command."""
        self.buildDwarf()
        self.regexp_break_command()

    def setUp(self):
        # Call super's setUp().
        TestBase.setUp(self)
        # Find the line number to break inside main().
        self.source = 'main.c'
        self.line = line_number(self.source, '// Set break point at this line.')

    def regexp_break_command(self):
        """Test the super consie "b" command, which is analias for _regexp-break."""
        exe = os.path.join(os.getcwd(), "a.out")
        self.runCmd("file " + exe, CURRENT_EXECUTABLE_SET)

        self.expect("b %d" % self.line,
                    BREAKPOINT_CREATED,
            substrs = ["Breakpoint created: 1: file ='main.c', line = %d, locations = 1" % self.line])
        self.expect("b %s:%d" % (self.source, self.line),
                    BREAKPOINT_CREATED,
            substrs = ["Breakpoint created: 2: file ='main.c', line = %d, locations = 1" % self.line])

        self.runCmd("run", RUN_SUCCEEDED)

        # The stop reason of the thread should be breakpoint.
        self.expect("thread list", STOPPED_DUE_TO_BREAKPOINT,
            substrs = ['stopped',
                       'stop reason = breakpoint'])


if __name__ == '__main__':
    import atexit
    lldb.SBDebugger.Initialize()
    atexit.register(lambda: lldb.SBDebugger.Terminate())
    unittest2.main()
