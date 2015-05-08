"""
Test that the lldb-mi driver prints prompt properly.
"""

import lldbmi_testcase
from lldbtest import *
import unittest2

class MiPromptTestCase(lldbmi_testcase.MiTestCaseBase):

    mydir = TestBase.compute_mydir(__file__)

    @lldbmi_test
    @expectedFailureWindows("llvm.org/pr22274: need a pexpect replacement for windows")
    @skipIfFreeBSD # llvm.org/pr22411: Failure presumably due to known thread races
    def test_lldbmi_prompt(self):
        """Test that 'lldb-mi --interpreter' echos '(gdb)' after commands and events."""

        self.spawnLldbMi(args = None)

        # Test that lldb-mi is ready after unknown command
        self.runCmd("-unknown-command")
        self.expect("\^error,msg=\"Driver\. Received command '-unknown-command'\. It was not handled\. Command 'unknown-command' not in Command Factory\"")
        self.expect(self.child_prompt, exactly = True)

        # Test that lldb-mi is ready after -file-exec-and-symbols
        self.runCmd("-file-exec-and-symbols %s" % self.myexe)
        self.expect("\^done")
        self.expect(self.child_prompt, exactly = True)

        # Test that lldb-mi is ready after -break-insert
        self.runCmd("-break-insert -f main")
        self.expect("\^done,bkpt={number=\"1\"")
        self.expect(self.child_prompt, exactly = True)

        # Test that lldb-mi is ready after -exec-run
        self.runCmd("-exec-run")
        self.expect("\*running")
        self.expect(self.child_prompt, exactly = True)

        # Test that lldb-mi is ready after BP hit
        self.expect("\*stopped,reason=\"breakpoint-hit\"")
        self.expect(self.child_prompt, exactly = True)

        # Test that lldb-mi is ready after -exec-continue
        self.runCmd("-exec-continue")
        self.expect("\^running")
        self.expect(self.child_prompt, exactly = True)

        # Test that lldb-mi is ready after program exited
        self.expect("\*stopped,reason=\"exited-normally\"")
        self.expect(self.child_prompt, exactly = True)

if __name__ == '__main__':
    unittest2.main()
