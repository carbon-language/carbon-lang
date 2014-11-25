"""
Test various ways the lldb-mi driver can launch a program.
"""

import os
import unittest2
import lldb
from lldbtest import *

class MiLaunchTestCase(TestBase):

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
    def test_lldbmi_exe(self):
        """Test that 'lldb-mi --interpreter' works for -file-exec-and-symbols exe."""
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

                #use no path
                child.send("-file-exec-and-symbols " + self.myexe)
                child.sendline('')
                child.expect("\^done")

                child.send("-exec-run")
                child.sendline('') # FIXME: lldb-mi hangs here, so the extra return below is needed
                child.send("")
                child.sendline('')
                child.expect("\^running")
                child.expect("\*stopped,reason=\"exited-normally\"")
                child.expect_exact(prompt)

                child.send("quit")
                child.sendline('')

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
    def test_lldbmi_abspathexe(self):
        """Test that 'lldb-mi --interpreter' works for -file-exec-and-symbols fullpath/exe."""
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

                #use full path
                exe = os.path.join(os.getcwd(), "a.out")
                child.send("-file-exec-and-symbols " + exe)
                child.sendline('')
                child.expect("\^done")

                child.send("-exec-run")
                child.sendline('') # FIXME: lldb-mi hangs here, so the extra return below is needed
                child.send("")
                child.sendline('')
                child.expect("\^running")
                child.expect("\*stopped,reason=\"exited-normally\"")
                child.expect_exact(prompt)

                child.send("quit")
                child.sendline('')

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
    def test_lldbmi_relpathexe(self):
        """Test that 'lldb-mi --interpreter' works for -file-exec-and-symbols relpath/exe."""
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

                #use relative path
                exe = "../../" + self.mydir + "/" + self.myexe
                child.send("-file-exec-and-symbols " + exe)
                child.sendline('')
                child.expect("\^done")

                child.send("-exec-run")
                child.sendline('') # FIXME: lldb-mi hangs here, so the extra return below is needed
                child.send("")
                child.sendline('')
                child.expect("\^running")
                child.expect("\*stopped,reason=\"exited-normally\"")
                child.expect_exact(prompt)

                child.send("quit")
                child.sendline('')

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

    @unittest2.skip("lldb-mi badpath hang")
    @lldbmi_test
    def test_lldbmi_badpathexe(self):
        """Test that 'lldb-mi --interpreter' works for -file-exec-and-symbols badpath/exe."""
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

                #use relative path
                exe = "badpath/" + self.myexe
                #print ("-file-exec-and-symbols " + exe)
                child.send("-file-exec-and-symbols " + exe)
                child.sendline('') #FIXME: non-existant directory caused hang
                child.expect("\^error")

                child.expect_exact(prompt)

                child.send("quit")
                child.sendline('')

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
