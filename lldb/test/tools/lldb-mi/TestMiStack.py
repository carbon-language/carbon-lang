"""
Test that the lldb-mi driver works with -stack-xxx commands
"""

import os
import unittest2
import lldb
from lldbtest import *

class MiStackTestCase(TestBase):

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

    @lldbmi_test
    @expectedFailureWindows("llvm.org/pr22274: need a pexpect replacement for windows")
    def test_lldbmi_stackargs(self):
        """Test that 'lldb-mi --interpreter' can shows arguments."""
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

                # Load executable
                child.sendline("-file-exec-and-symbols %s" % (self.myexe))
                child.expect("\^done")

                # Run to main
                child.sendline("-break-insert -f main")
                child.expect("\^done,bkpt={number=\"1\"")
                child.sendline("-exec-run")
                child.expect("\^running")
                child.expect("\*stopped,reason=\"breakpoint-hit\"")

                # Test arguments
                child.sendline("-stack-list-arguments 0")
                child.expect("\^done,stack-args=\[frame={level=\"0\",args=\[{name=\"argc\",value=\"1\"},{name=\"argv\",value=\".*\"}\]}")

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

    @lldbmi_test
    @expectedFailureWindows("llvm.org/pr22274: need a pexpect replacement for windows")
    def test_lldbmi_locals(self):
        """Test that 'lldb-mi --interpreter' can shows local variables."""
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

                # Load executable
                child.sendline("-file-exec-and-symbols %s" % (self.myexe))
                child.expect("\^done")

                # Run to main
                self.line = line_number('main.c', '//BP_localstest')
                child.sendline("-break-insert --file main.c:%d" % (self.line))
                child.expect("\^done,bkpt={number=\"1\"")
                child.sendline("-exec-run")
                child.expect("\^running")
                child.expect("\*stopped,reason=\"breakpoint-hit\"")

                # Test locals
                child.sendline("-stack-list-locals 0")
                child.expect("\^done,locals=\[{name=\"a\",value=\"10\"},{name=\"b\",value=\"20\"}\]")

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
