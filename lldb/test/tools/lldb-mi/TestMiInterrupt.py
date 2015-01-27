"""
Test that the lldb-mi driver can interrupt and resume a looping app.
"""

import lldbmi_testcase
from lldbtest import *
import unittest2

class MiInterruptTestCase(lldbmi_testcase.MiTestCaseBase):

    @lldbmi_test
    @unittest2.skipUnless(sys.platform.startswith("darwin"), "requires Darwin")
    @expectedFailureWindows("llvm.org/pr22274: need a pexpect replacement for windows")
    def test_lldbmi_interrupt(self):
        """Test that 'lldb-mi --interpreter' interrupt and resume a looping app."""

        self.spawnLldbMi(args = None)

        self.runCmd("-file-exec-and-symbols %s" % self.myexe)
        self.expect("\^done")

        #run to main
        self.runCmd("-break-insert -f main")
        self.expect("\^done,bkpt={number=\"1\"")
        self.runCmd("-exec-run")
        self.expect("\^running")
        self.expect("\*stopped,reason=\"breakpoint-hit\"")

        #set doloop=1 and run (to loop forever)
        self.runCmd("-data-evaluate-expression \"doloop=1\"")
        self.expect("value=\"1\"")
        self.runCmd("-exec-continue")
        self.expect("\^running")

        #issue interrupt, set BP in loop (marked BP_loop), and resume
        self.runCmd("-exec-interrupt")
        self.expect("\*stopped,reason=\"signal-received\"")
        line = line_number('loop.c', '//BP_loop')
        self.runCmd("-break-insert loop.c:%d" % line)
        self.expect("\^done,bkpt={number=\"2\"")
        #self.runCmd("-exec-resume") #FIXME: command not recognized
        self.runCmd("-exec-continue")
        self.expect("\*stopped,reason=\"breakpoint-hit\"")

        #we should have hit BP
        #set loop=-1 so we'll exit the loop
        self.runCmd("-data-evaluate-expression \"loop=-1\"")
        self.expect("value=\"-1\"")
        self.runCmd("-exec-continue")
        self.expect("\^running")
        self.expect("\*stopped,reason=\"exited-normally\"")

if __name__ == '__main__':
    unittest2.main()
