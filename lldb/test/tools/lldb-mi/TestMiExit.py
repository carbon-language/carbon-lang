"""
Test that the lldb-mi driver works properly with "-gdb-exit".
"""

import lldbmi_testcase
from lldbtest import *
import unittest2

class MiExitTestCase(lldbmi_testcase.MiTestCaseBase):

    @lldbmi_test
    @expectedFailureWindows("llvm.org/pr22274: need a pexpect replacement for windows")
    def test_lldbmi_gdbexit(self):
        """Test that '-gdb-exit' terminates debug session and exits."""

        self.spawnLldbMi(args = None)

        # Load executable
        self.runCmd("-file-exec-and-symbols %s" % self.myexe)
        self.expect("\^done")

        # Run to main
        self.runCmd("-break-insert -f main")
        self.expect("\^done,bkpt={number=\"1\"")
        self.runCmd("-exec-run")
        self.expect("\^running")
        self.expect("\*stopped,reason=\"breakpoint-hit\"")

        # Test -gdb-exit: try to exit and check that program is finished
        self.runCmd("-gdb-exit")
        self.runCmd("") #FIXME hangs here on Linux; extra return is needed
        self.expect("\^exit")
        import pexpect
        self.expect(pexpect.EOF)

    @lldbmi_test
    @expectedFailureWindows("llvm.org/pr22274: need a pexpect replacement for windows")
    def test_lldbmi_quit(self):
        """Test that 'quit' exits immediately."""

        self.spawnLldbMi(args = None)

        # Load executable
        self.runCmd("-file-exec-and-symbols %s" % self.myexe)
        self.expect("\^done")

        # Run to main
        self.runCmd("-break-insert -f main")
        self.expect("\^done,bkpt={number=\"1\"")
        self.runCmd("-exec-run")
        self.expect("\^running")
        self.expect("\*stopped,reason=\"breakpoint-hit\"")

        # Test quit: try to exit and check that program is finished
        self.runCmd("quit")
        import pexpect
        self.expect(pexpect.EOF)

if __name__ == '__main__':
    unittest2.main()
