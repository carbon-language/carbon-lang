"""
Test that the lldb-mi driver can evaluate expressions.
"""

import os
import unittest2
import lldb
from lldbtest import *

class MiEvaluateTestCase(TestBase):

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
    def test_lldbmi_eval(self):
        """Test that 'lldb-mi --interpreter' works for evaluating."""
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

                child.send("-file-exec-and-symbols " + self.myexe)
                child.sendline('')
                child.expect("\^done")

                #run to main
                child.send("-break-insert -f main")
                child.sendline('')
                child.expect("\^done,bkpt={number=\"1\"")

                child.send("-exec-run")
                child.sendline('') #FIXME: hangs here; extra return below is needed
                child.send("")
                child.sendline('')
                child.expect("\^running")
                child.expect("\*stopped,reason=\"breakpoint-hit\"")

                #run to program return
                child.send("-break-insert main.c:30") #BP_source
                child.sendline('')
                child.expect("\^done,bkpt={number=\"2\"")

                child.send("-exec-continue")
                child.sendline('')
                child.expect("\^running")
                child.expect("\*stopped,reason=\"breakpoint-hit\"")

                #print non-existant variable
                #child.send("-var-create var1 --thread 1 --frame 0 * undef")
                #child.sendline('') #FIXME: shows undef as {...}
                #child.expect("error")
                #child.send("-data-evaluate-expression undef")
                #child.sendline('') #FIXME: gets value="undef"
                #child.expect("error")

                #print global "g_MyVar"
                child.send("-var-create var1 --thread 1 --frame 0 * g_MyVar")
                child.sendline('') #FIXME: shows name=<unnamedvariable>"
                child.expect("value=\"3\",type=\"int\"")
                #child.send("-var-evaluate-expression var1")
                #child.sendline('') #FIXME: gets var1 does not exist
                child.send("-var-show-attributes var1")
                child.sendline('')
                child.expect("status=\"editable\"")
                child.send("-var-delete var1")
                child.sendline('')
                child.expect("\^done")
                child.send("-var-create var1 --thread 1 --frame 0 * g_MyVar")
                child.sendline('')
                child.expect("value=\"3\",type=\"int\"")

                #print static "s_MyVar" and modify
                child.send("-data-evaluate-expression s_MyVar")
                child.sendline('')
                child.expect("value=\"30\"")
                child.send("-var-create var3 --thread 1 --frame 0 * \"s_MyVar=3\"")
                child.sendline('')
                child.expect("value=\"3\",type=\"int\"")
                child.send("-data-evaluate-expression \"s_MyVar=30\"")
                child.sendline('')
                child.expect("value=\"30\"")

                #print local "b" and modify
                child.send("-data-evaluate-expression b")
                child.sendline('')
                child.expect("value=\"20\"")
                child.send("-var-create var3 --thread 1 --frame 0 * \"b=3\"")
                child.sendline('')
                child.expect("value=\"3\",type=\"int\"")
                child.send("-data-evaluate-expression \"b=20\"")
                child.sendline('')
                child.expect("value=\"20\"")

                #print "a + b"
                child.send("-data-evaluate-expression \"a + b\"")
                child.sendline('')
                child.expect("value=\"30\"")
                child.send("-var-create var3 --thread 1 --frame 0 * \"a + b\"")
                child.sendline('')
                child.expect("value=\"30\",type=\"int\"")

                #print "argv[0]"
                child.send("-data-evaluate-expression \"argv[0]\"")
                child.sendline('')
                child.expect("value=\"0x")
                child.send("-var-create var3 --thread 1 --frame 0 * \"argv[0]\"")
                child.sendline('')
                child.expect("numchild=\"1\",value=\"0x.*\",type=\"const char \*\"")

                #run to exit
                child.send("-exec-continue")
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


if __name__ == '__main__':
    import atexit
    lldb.SBDebugger.Initialize()
    atexit.register(lambda: lldb.SBDebugger.Terminate())
    unittest2.main()
