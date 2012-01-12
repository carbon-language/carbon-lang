"""Test breaking inside functions defined within a BSD archive file libfoo.a."""

import os, time
import unittest2
import lldb
from lldbtest import *

class BSDArchivesTestCase(TestBase):

    mydir = os.path.join("functionalities", "archives")

    def test_with_dwarf(self):
        """Break inside a() and b() defined within libfoo.a."""
        self.buildDwarf()
        self.break_inside_bsd_archives()

    def setUp(self):
        # Call super's setUp().
        TestBase.setUp(self)
        # Find the line number in a(int) to break at.
        self.line = line_number('a.c', '// Set file and line breakpoint inside a().')

    def break_inside_bsd_archives(self):
        """Break inside a() and b() defined within libfoo.a."""
        exe = os.path.join(os.getcwd(), "a.out")
        self.runCmd("file " + exe, CURRENT_EXECUTABLE_SET)

        # Break inside a() by file and line first.
        self.expect("breakpoint set -f a.c -l %d" % self.line, BREAKPOINT_CREATED,
            startstr = "Breakpoint created: 1: file ='a.c', line = %d" %
                        self.line)
        self.runCmd("run", RUN_SUCCEEDED)

        # The stop reason of the thread should be breakpoint.
        self.expect("thread list", STOPPED_DUE_TO_BREAKPOINT,
            substrs = ['stopped',
                       'stop reason = breakpoint'])

        # Break at a(int) first.
        self.expect("frame variable", VARIABLES_DISPLAYED_CORRECTLY,
            substrs = ['(int) arg = 1'])
        self.expect("frame variable __a_global", VARIABLES_DISPLAYED_CORRECTLY,
            substrs = ['(int) __a_global = 1'])

        # Set breakpoint for b() next.
        self.expect("breakpoint set -n b", BREAKPOINT_CREATED,
            startstr = "Breakpoint created: 2: name = 'b'",
            substrs = ['resolved = 1'])

        # Continue the program, we should break at b(int) next.
        self.runCmd("continue")
        self.expect("thread list", STOPPED_DUE_TO_BREAKPOINT,
            substrs = ['stopped',
                       'stop reason = breakpoint'])
        self.expect("frame variable", VARIABLES_DISPLAYED_CORRECTLY,
            substrs = ['(int) arg = 2'])
        self.expect("frame variable __b_global", VARIABLES_DISPLAYED_CORRECTLY,
            substrs = ['(int) __b_global = 2'])


if __name__ == '__main__':
    import atexit
    lldb.SBDebugger.Initialize()
    atexit.register(lambda: lldb.SBDebugger.Terminate())
    unittest2.main()
