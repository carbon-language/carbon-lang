"""
Test the 'memory read' command.
"""

import os, time
import re
import unittest2
import lldb
from lldbtest import *

class MemoryReadTestCase(TestBase):

    mydir = os.path.join("functionalities", "memory", "read")

    @unittest2.skipUnless(os.uname()[4] in ['x86_64'], "requires x86_64")
    def test_register_commands(self):
        """Test commands related to registers, in particular xmm registers."""
        self.buildDefault()
        self.register_commands()

    def register_commands(self):
        """Test commands related to registers, in particular xmm registers."""
        exe = os.path.join(os.getcwd(), "a.out")
        self.runCmd("file " + exe, CURRENT_EXECUTABLE_SET)

        # Break in main().
        self.expect("breakpoint set -n main",
                    BREAKPOINT_CREATED,
            startstr = "Breakpoint created: 1: name = 'main'")

        self.runCmd("run", RUN_SUCCEEDED)

        # The stop reason of the thread should be breakpoint.
        self.expect("thread list", STOPPED_DUE_TO_BREAKPOINT,
            substrs = ['stopped', 'stop reason = breakpoint'])

        # Test some register-related commands.

        self.runCmd("register read -a")
        self.runCmd("register read xmm0")

        # rdar://problem/10611315
        # expression command doesn't handle xmm or stmm registers...
        self.expect("expr $xmm0",
            substrs = ['vector_type'])

        self.expect("expr (unsigned int)$xmm0[0]",
            substrs = ['unsigned int'])


if __name__ == '__main__':
    import atexit
    lldb.SBDebugger.Initialize()
    atexit.register(lambda: lldb.SBDebugger.Terminate())
    unittest2.main()
