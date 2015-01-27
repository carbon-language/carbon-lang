"""
Test that the lldb-mi driver can pass arguments to the app.
"""

import lldbmi_testcase
from lldbtest import *
import unittest2

class MiProgramArgsTestCase(lldbmi_testcase.MiTestCaseBase):

    @lldbmi_test
    @unittest2.skip("lldb-mi can't pass params to app.")
    @expectedFailureWindows("llvm.org/pr22274: need a pexpect replacement for windows")
    def test_lldbmi_paramargs(self):
        """Test that 'lldb-mi --interpreter' can pass arguments to the app."""

        self.spawnLldbMi(args = None)

        self.runCmd("-file-exec-and-symbols %s" % self.myexe)
        self.expect("\^done")

        self.runCmd("settings set target.run-args l") #FIXME: args not passed
        #self.runCmd("-exec-arguments l") #FIXME: not recognized and hung lldb-mi

        #run to main
        self.runCmd("-break-insert -f main")
        self.expect("\^done,bkpt={number=\"1\"")
        self.runCmd("-exec-run")
        self.expect("\^running")
        self.expect("\*stopped,reason=\"breakpoint-hit\"")

        #check argc to see if arg passed
        self.runCmd("-data-evaluate-expression argc")
        self.expect("value=\"2\"")

        #set BP on code which is only executed if "l" was passed correctly (marked BP_argtest)
        line = line_number('main.c', '//BP_argtest')
        self.runCmd("-break-insert main.c:%d" % line)
        self.expect("\^done,bkpt={number=\"2\"")
        self.runCmd("-exec-continue")
        self.expect("\^running")
        self.expect("\*stopped,reason=\"breakpoint-hit\"")

if __name__ == '__main__':
    unittest2.main()
