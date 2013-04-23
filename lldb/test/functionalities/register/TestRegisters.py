"""
Test the 'register' command.
"""

import os, time
import re
import unittest2
import lldb
from lldbtest import *
import lldbutil

class RegisterCommandsTestCase(TestBase):

    mydir = os.path.join("functionalities", "register")

    def test_register_commands(self):
        """Test commands related to registers, in particular vector registers."""
        if not self.getArchitecture() in ['i386', 'x86_64']:
            self.skipTest("This test requires i386 or x86_64 as the architecture for the inferior")
        self.buildDefault()
        self.register_commands()

    def test_register_expressions(self):
        """Test expression evaluation with commands related to registers."""
        if not self.getArchitecture() in ['i386', 'x86_64']:
            self.skipTest("This test requires i386 or x86_64 as the architecture for the inferior")
        self.buildDefault()
        self.register_expressions()

    @expectedFailureLinux # bugzilla 14600 - Convenience registers not supported on Linux
    def test_convenience_registers(self):
        """Test convenience registers."""
        if not self.getArchitecture() in ['x86_64']:
            self.skipTest("This test requires x86_64 as the architecture for the inferior")
        self.buildDefault()
        self.convenience_registers()

    @expectedFailureLinux # bugzilla 14600 - Convenience registers not supported on Linux
    def test_convenience_registers_with_process_attach(self):
        """Test convenience registers after a 'process attach'."""
        if not self.getArchitecture() in ['x86_64']:
            self.skipTest("This test requires x86_64 as the architecture for the inferior")
        self.buildDefault()
        self.convenience_registers_with_process_attach()

    def common_setup(self):
        exe = os.path.join(os.getcwd(), "a.out")
        self.runCmd("file " + exe, CURRENT_EXECUTABLE_SET)

        # Break in main().
        lldbutil.run_break_set_by_symbol (self, "main", num_expected_locations=-1)

        self.runCmd("run", RUN_SUCCEEDED)

        # The stop reason of the thread should be breakpoint.
        self.expect("thread list", STOPPED_DUE_TO_BREAKPOINT,
            substrs = ['stopped', 'stop reason = breakpoint'])

    def register_commands(self):
        """Test commands related to registers, in particular vector registers."""
        self.common_setup()

        self.expect("register read -a", MISSING_EXPECTED_REGISTERS,
            substrs = ['registers were unavailable'], matching = False)
        self.runCmd("register read xmm0")
        self.runCmd("register read ymm15") # may be available

        self.expect("register read -s 3",
            substrs = ['invalid register set index: 3'], error = True)

    @expectedFailureLinux # bugzilla 14661 - Expressions involving XMM registers fail on Linux
    def register_expressions(self):
        """Test expression evaluation with commands related to registers."""
        self.common_setup()

        # rdar://problem/10611315
        # expression command doesn't handle xmm or stmm registers...
        self.expect("expr $xmm0",
            substrs = ['vector_type'])

        self.expect("expr (unsigned int)$xmm0[0]",
            substrs = ['unsigned int'])

    def convenience_registers(self):
        """Test convenience registers."""
        self.common_setup()

        # The vanilla "register read" command does not output derived register like eax.
        self.expect("register read", matching=False,
            substrs = ['eax'])
        # While "register read -a" does output derived register like eax.
        self.expect("register read -a", matching=True,
            substrs = ['eax'])
        
        # Test reading of rax and eax.
        self.runCmd("register read rax eax")

        # Now write rax with a unique bit pattern and test that eax indeed represents the lower half of rax.
        self.runCmd("register write rax 0x1234567887654321")
        self.expect("expr -- ($rax & 0xffffffff) == $eax",
            substrs = ['true'])
        self.expect("expr -- $ax == (($ah << 8) | $al)",
            substrs = ['true'])

    def convenience_registers_with_process_attach(self):
        """Test convenience registers after a 'process attach'."""
        exe = self.lldbHere
        
        # Spawn a new process
        proc = self.spawnSubprocess(exe, [self.lldbOption])
        self.addTearDownHook(self.cleanupSubprocesses)

        if self.TraceOn():
            print "pid of spawned process: %d" % proc.pid

        self.runCmd("process attach -p %d" % proc.pid)

        # Check that "register read eax" works.
        self.runCmd("register read eax")


if __name__ == '__main__':
    import atexit
    lldb.SBDebugger.Initialize()
    atexit.register(lambda: lldb.SBDebugger.Terminate())
    unittest2.main()
