"""
Test example snippets from the lldb 'help expression' output.
"""

import os, time
import unittest2
import lldb
from lldbtest import *

class Radar9673644TestCase(TestBase):

    mydir = os.path.join("expression_command", "radar_9673664")

    def setUp(self):
        # Call super's setUp().
        TestBase.setUp(self)
        # Find the line number to break inside main().
        self.main_source = "main.c"
        self.line = line_number(self.main_source, '// Set breakpoint here.')

    # rdar://problem/9673664
    @unittest2.expectedFailure
    def test_expr_commands(self):
        """The following expression commands should just work."""
        self.buildDefault()

        self.runCmd("file a.out", CURRENT_EXECUTABLE_SET)

        self.expect("breakpoint set -f %s -l %d" % (self.main_source, self.line),
                    BREAKPOINT_CREATED,
            startstr = "Breakpoint created: 1: file ='%s', line = %d, locations = 1" %
                        (self.main_source, self.line))

        self.runCmd("run", RUN_SUCCEEDED)

        # rdar://problem/9673664 lldb expression evaluation problem

        self.runCmd('expr char c[] = "foo"; c[0]')
        # Fill in an example output here.
        # And change self.runCmd() -> self.expect()...


if __name__ == '__main__':
    import atexit
    lldb.SBDebugger.Initialize()
    atexit.register(lambda: lldb.SBDebugger.Terminate())
    unittest2.main()
