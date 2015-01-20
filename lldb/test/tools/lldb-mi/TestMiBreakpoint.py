"""
Test that the lldb-mi driver understands an MI breakpoint command.
"""

import os
import unittest2
import lldb
from lldbtest import *

class MiBreakpointTestCase(TestBase):

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
    def test_lldbmi_pendbreakonsym(self):
        """Test that 'lldb-mi --interpreter' works for pending symbol breakpoints."""
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

                child.sendline("-break-insert -f a_MyFunction")
                child.expect("\^done,bkpt={number=\"1\"")

                child.sendline("-exec-run")
                child.expect("\^running")
                child.expect("\*stopped,reason=\"breakpoint-hit\"")

                child.sendline("-exec-continue")
                child.expect("\^running")
                child.expect("\*stopped,reason=\"exited-normally\"")

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

            self.expect(from_child, exe=False,
                substrs = ["breakpoint-hit"])

    @lldbmi_test
    @expectedFailureWindows("llvm.org/pr22274: need a pexpect replacement for windows")
    def test_lldbmi_pendbreakonsrc(self):
        """Test that 'lldb-mi --interpreter' works for pending source breakpoints."""
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

                # Find the line number to break inside main() and set
                # pending BP.
                self.line = line_number('main.c', '//BP_source')
                child.sendline("-break-insert -f main.c:%d" % self.line)
                child.expect("\^done,bkpt={number=\"1\"")

                child.sendline("-exec-run")
                child.expect("\^running")
                child.expect("\*stopped,reason=\"breakpoint-hit\"")

                child.sendline("-exec-continue")
                child.expect("\^running")
                child.expect("\*stopped,reason=\"exited-normally\"")

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

            self.expect(from_child, exe=False,
                substrs = ["breakpoint-hit"])

    @lldbmi_test
    @expectedFailureWindows("llvm.org/pr22274: need a pexpect replacement for windows")
    def test_lldbmi_breakpoints(self):
        """Test that 'lldb-mi --interpreter' works for breakpoints."""
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

                child.sendline("-break-insert -f main")
                child.expect("\^done,bkpt={number=\"1\"")

                child.sendline("-exec-run")
                child.expect("\^running")
                child.expect("\*stopped,reason=\"breakpoint-hit\"")

                #break on symbol
                child.sendline("-break-insert a_MyFunction")
                child.expect("\^done,bkpt={number=\"2\"")

                child.sendline("-exec-continue")
                child.expect("\^running")
                child.expect("\*stopped,reason=\"breakpoint-hit\"")

                #break on source
                self.line = line_number('main.c', '//BP_source')
                child.sendline("-break-insert main.c:%d" % self.line)
                child.expect("\^done,bkpt={number=\"3\"")

                child.sendline("-exec-continue")
                child.expect("\^running")
                child.expect("\*stopped,reason=\"breakpoint-hit\"")

                #run to exit
                child.sendline("-exec-continue")
                child.expect("\^running")
                child.expect("\*stopped,reason=\"exited-normally\"")

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

            self.expect(from_child, exe=False,
                substrs = ["breakpoint-hit"])


if __name__ == '__main__':
    import atexit
    lldb.SBDebugger.Initialize()
    atexit.register(lambda: lldb.SBDebugger.Terminate())
    unittest2.main()
