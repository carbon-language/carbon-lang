"""
Test the 'register' command.
"""

import os, time
import re
import unittest2
import lldb
from lldbtest import *

class RegisterCommandsTestCase(TestBase):

    mydir = os.path.join("functionalities", "register")

    def test_register_commands(self):
        """Test commands related to registers, in particular xmm registers."""
        if not self.getArchitecture() in ['i386', 'x86_64']:
            self.skipTest("This test requires i386 or x86_64 as the architecture for the inferior")
        self.buildDefault()
        self.register_commands()

    def test_convenience_registers(self):
        """Test convenience registers."""
        if not self.getArchitecture() in ['x86_64']:
            self.skipTest("This test requires x86_64 as the architecture for the inferior")
        self.buildDefault()
        self.convenience_registers()

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

    def convenience_registers(self):
        """Test convenience registers."""
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

        # Test reading of rax and eax.
        self.runCmd("register read rax eax")

        # No write rax with a unique bit pattern and test that eax indeed represents the lower half of rax.
        self.runCmd("register write rax 0x1234567887654321")
        self.expect("expr -- ($rax & 0xffffffff) == $eax",
            substrs = ['true'])


if __name__ == '__main__':
    import atexit
    lldb.SBDebugger.Initialize()
    atexit.register(lambda: lldb.SBDebugger.Terminate())
    unittest2.main()
