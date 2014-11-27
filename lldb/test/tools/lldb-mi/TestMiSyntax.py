"""
Test that the lldb-mi driver understands MI command syntax.
"""

import os
import unittest2
import lldb
from lldbtest import *

class MiSyntaxTestCase(TestBase):

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
    def test_lldbmi_tokens(self):
        """Test that 'lldb-mi --interpreter' echos command tokens."""
        import pexpect
        self.buildDefault()

        # The default lldb-mi prompt (seriously?!).
        prompt = "(gdb)"

        # So that the child gets torn down after the test.
        self.child = pexpect.spawn('%s --interpreter' % (self.lldbMiExec))
        child = self.child
        child.setecho(True)
        # Turn on logging for input/output to/from the child.
        with open('child_send.txt', 'w') as f_send:
            with open('child_read.txt', 'w') as f_read:
                child.logfile_send = f_send
                child.logfile_read = f_read

                child.sendline("000-file-exec-and-symbols " + self.myexe)
                child.expect("000\^done")

                child.sendline("100000001-break-insert -f a_MyFunction")
                child.expect("100000001\^done,bkpt={number=\"1\"")

                child.sendline("2-exec-run")
                child.sendline("") # FIXME: lldb-mi hangs here, so extra return is needed
                child.expect("2\^running")
                child.expect("\*stopped,reason=\"breakpoint-hit\"")

                child.sendline("0000000000000000000003-exec-continue")
                child.expect("0000000000000000000003\^running")
                child.expect("\*stopped,reason=\"exited-normally\"")
                child.expect_exact(prompt)

                child.sendline("quit")

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
