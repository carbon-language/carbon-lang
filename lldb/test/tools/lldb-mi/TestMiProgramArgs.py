"""
Test that the lldb-mi driver can pass arguments to the app.
"""

import os
import unittest2
import lldb
from lldbtest import *

class MiProgramArgsTestCase(TestBase):

    mydir = TestBase.compute_mydir(__file__)
    myexe = "a.out"

    @classmethod
    def classCleanup(cls):
        """Cleanup the test byproducts."""
        try:
            os.remove("child_send.txt")
            os.remove("child_read.txt")
            os.remove(cls.myexe)
        except:
            pass

    @unittest2.skip("lldb-mi can't pass params to app.")
    @lldbmi_test
    @expectedFailureWindows("llvm.org/pr22274: need a pexpect replacement for windows")
    def test_lldbmi_paramargs(self):
        """Test that 'lldb-mi --interpreter' can pass arguments to the app."""
        import pexpect
        self.buildDefault()

        # So that the child gets torn down after the test.
        self.child = pexpect.spawn('%s --interpreter' % (self.lldbMiExec))
        child = self.child
        child.setecho(True)
        # Turn on logging for input/output to/from the child.
        with open('child_send.txt', 'w') as f_send:
            with open('child_read.txt', 'w') as f_read:
                child.logfile_send = f_send
                child.logfile_read = f_read

                child.sendline("-file-exec-and-symbols " + self.myexe)
                child.expect("\^done")

                child.sendline("settings set target.run-args l") #FIXME: args not passed
                #child.sendline("-exec-arguments l") #FIXME: not recognized and hung lldb-mi

                #run to main
                child.sendline("-break-insert -f main")
                child.expect("\^done,bkpt={number=\"1\"")
                child.sendline("-exec-run")
                child.expect("\^running")
                child.expect("\*stopped,reason=\"breakpoint-hit\"")

                #check argc to see if arg passed
                child.sendline("-data-evaluate-expression argc")
                child.expect("value=\"2\"")

                #set BP on code which is only executed if "l" was passed correctly (marked BP_argtest)
                self.line = line_number('main.c', '//BP_argtest')
                child.sendline("-break-insert main.c:%d" % self.line)
                child.expect("\^done,bkpt={number=\"2\"")
                child.sendline("-exec-continue")
                child.expect("\^running")
                child.expect("\*stopped,reason=\"breakpoint-hit\"")

        # Now that the necessary logging is done, restore logfile to None to
        # stop further logging.
        child.logfile_send = None
        child.logfile_read = None
        
        with open('child_send.txt', 'r') as fs:
            if self.TraceOn():
                print "\n\nContents of child_send.txt:"
                print fs.read()
        with open('child_read.txt', 'r') as fr:
            from_child = fr.read()
            if self.TraceOn():
                print "\n\nContents of child_read.txt:"
                print from_child

if __name__ == '__main__':
    import atexit
    lldb.SBDebugger.Initialize()
    atexit.register(lambda: lldb.SBDebugger.Terminate())
    unittest2.main()
