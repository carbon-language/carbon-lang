"""
Test that the lldb-mi driver understands an MI breakpoint command.
"""

import lldbmi_testcase
from lldbtest import *
import unittest2

class MiBreakpointTestCase(lldbmi_testcase.MiTestCaseBase):

    @lldbmi_test
    @expectedFailureWindows("llvm.org/pr22274: need a pexpect replacement for windows")
    def test_lldbmi_pendbreakonsym(self):
        """Test that 'lldb-mi --interpreter' works for pending symbol breakpoints."""

        self.spawnLldbMi(args = None)

        self.runCmd("-file-exec-and-symbols %s" % self.myexe)
        self.expect("\^done")

        self.runCmd("-break-insert -f a_MyFunction")
        self.expect("\^done,bkpt={number=\"1\"")

        self.runCmd("-exec-run")
        self.expect("\^running")
        self.expect("\*stopped,reason=\"breakpoint-hit\"")

        self.runCmd("-exec-continue")
        self.expect("\^running")
        self.expect("\*stopped,reason=\"exited-normally\"")

    @lldbmi_test
    @expectedFailureWindows("llvm.org/pr22274: need a pexpect replacement for windows")
    def test_lldbmi_pendbreakonsrc(self):
        """Test that 'lldb-mi --interpreter' works for pending source breakpoints."""

        self.spawnLldbMi(args = None)

        self.runCmd("-file-exec-and-symbols %s" % self.myexe)
        self.expect("\^done")

        # Find the line number to break inside main() and set
        # pending BP.
        line = line_number('main.c', '//BP_source')
        self.runCmd("-break-insert -f main.c:%d" % line)
        self.expect("\^done,bkpt={number=\"1\"")

        self.runCmd("-exec-run")
        self.expect("\^running")
        self.expect("\*stopped,reason=\"breakpoint-hit\"")

        self.runCmd("-exec-continue")
        self.expect("\^running")
        self.expect("\*stopped,reason=\"exited-normally\"")

    @lldbmi_test
    @expectedFailureWindows("llvm.org/pr22274: need a pexpect replacement for windows")
    def test_lldbmi_breakpoints(self):
        """Test that 'lldb-mi --interpreter' works for breakpoints."""

        self.spawnLldbMi(args = None)

        self.runCmd("-file-exec-and-symbols %s" % self.myexe)
        self.expect("\^done")

        self.runCmd("-break-insert -f main")
        self.expect("\^done,bkpt={number=\"1\"")

        self.runCmd("-exec-run")
        self.expect("\^running")
        self.expect("\*stopped,reason=\"breakpoint-hit\"")

        #break on symbol
        self.runCmd("-break-insert a_MyFunction")
        self.expect("\^done,bkpt={number=\"2\"")

        self.runCmd("-exec-continue")
        self.expect("\^running")
        self.expect("\*stopped,reason=\"breakpoint-hit\"")

        #break on source
        line = line_number('main.c', '//BP_source')
        self.runCmd("-break-insert main.c:%d" % line)
        self.expect("\^done,bkpt={number=\"3\"")

        self.runCmd("-exec-continue")
        self.expect("\^running")
        self.expect("\*stopped,reason=\"breakpoint-hit\"")

        #run to exit
        self.runCmd("-exec-continue")
        self.expect("\^running")
        self.expect("\*stopped,reason=\"exited-normally\"")

if __name__ == '__main__':
    unittest2.main()
