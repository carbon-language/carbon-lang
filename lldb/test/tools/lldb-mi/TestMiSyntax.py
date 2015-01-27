"""
Test that the lldb-mi driver understands MI command syntax.
"""

import lldbmi_testcase
from lldbtest import *
import unittest2

class MiSyntaxTestCase(lldbmi_testcase.MiTestCaseBase):

    @lldbmi_test
    @expectedFailureWindows("llvm.org/pr22274: need a pexpect replacement for windows")
    def test_lldbmi_tokens(self):
        """Test that 'lldb-mi --interpreter' echos command tokens."""

        self.spawnLldbMi(args = None)

        # Load executable
        self.runCmd("000-file-exec-and-symbols %s" % self.myexe)
        self.expect("000\^done")

        # Run to main
        self.runCmd("100000001-break-insert -f a_MyFunction")
        self.expect("100000001\^done,bkpt={number=\"1\"")
        self.runCmd("2-exec-run")
        self.expect("2\^running")
        self.expect("\*stopped,reason=\"breakpoint-hit\"")

        # Exit
        self.runCmd("0000000000000000000003-exec-continue")
        self.expect("0000000000000000000003\^running")
        self.expect("\*stopped,reason=\"exited-normally\"")

if __name__ == '__main__':
    unittest2.main()
