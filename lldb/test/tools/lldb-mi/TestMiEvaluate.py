"""
Test that the lldb-mi driver can evaluate expressions.
"""

import lldbmi_testcase
from lldbtest import *
import unittest2

class MiEvaluateTestCase(lldbmi_testcase.MiTestCaseBase):

    @lldbmi_test
    @expectedFailureWindows("llvm.org/pr22274: need a pexpect replacement for windows")
    def test_lldbmi_eval(self):
        """Test that 'lldb-mi --interpreter' works for evaluating."""

        self.spawnLldbMi(args = None)

        self.runCmd("-file-exec-and-symbols %s" % self.myexe)
        self.expect("\^done")

        #run to main
        self.runCmd("-break-insert -f main")
        self.expect("\^done,bkpt={number=\"1\"")

        self.runCmd("-exec-run")
        self.expect("\^running")
        self.expect("\*stopped,reason=\"breakpoint-hit\"")

        #run to program return (marked BP_source)
        line = line_number('main.c', '//BP_source')
        self.runCmd("-break-insert main.c:%d" % line)
        self.expect("\^done,bkpt={number=\"2\"")

        self.runCmd("-exec-continue")
        self.expect("\^running")
        self.expect("\*stopped,reason=\"breakpoint-hit\"")

        #print non-existant variable
        #self.runCmd("-var-create var1 --thread 1 --frame 0 * undef") #FIXME: shows undef as {...}
        #self.expect("error")
        #self.runCmd("-data-evaluate-expression undef") #FIXME: gets value="undef"
        #self.expect("error")

        #print global "g_MyVar"
        self.runCmd("-var-create var1 --thread 1 --frame 0 * g_MyVar") #FIXME: shows name=<unnamedvariable>"
        self.expect("value=\"3\",type=\"int\"")
        #self.runCmd("-var-evaluate-expression var1") #FIXME: gets var1 does not exist
        self.runCmd("-var-show-attributes var1")
        self.expect("status=\"editable\"")
        self.runCmd("-var-delete var1")
        self.expect("\^done")
        self.runCmd("-var-create var1 --thread 1 --frame 0 * g_MyVar")
        self.expect("value=\"3\",type=\"int\"")

        #print static "s_MyVar" and modify
        self.runCmd("-data-evaluate-expression s_MyVar")
        self.expect("value=\"30\"")
        self.runCmd("-var-create var3 --thread 1 --frame 0 * \"s_MyVar=3\"")
        self.expect("value=\"3\",type=\"int\"")
        self.runCmd("-data-evaluate-expression \"s_MyVar=30\"")
        self.expect("value=\"30\"")

        #print local "b" and modify
        self.runCmd("-data-evaluate-expression b")
        self.expect("value=\"20\"")
        self.runCmd("-var-create var3 --thread 1 --frame 0 * \"b=3\"")
        self.expect("value=\"3\",type=\"int\"")
        self.runCmd("-data-evaluate-expression \"b=20\"")
        self.expect("value=\"20\"")

        #print "a + b"
        self.runCmd("-data-evaluate-expression \"a + b\"")
        self.expect("value=\"30\"")
        self.runCmd("-var-create var3 --thread 1 --frame 0 * \"a + b\"")
        self.expect("value=\"30\",type=\"int\"")

        #print "argv[0]"
        self.runCmd("-data-evaluate-expression \"argv[0]\"")
        self.expect("value=\"0x")
        self.runCmd("-var-create var3 --thread 1 --frame 0 * \"argv[0]\"")
        self.expect("numchild=\"1\",value=\"0x.*\",type=\"const char \*\"")

        #run to exit
        self.runCmd("-exec-continue")
        self.expect("\^running")
        self.expect("\*stopped,reason=\"exited-normally\"")

if __name__ == '__main__':
    unittest2.main()
