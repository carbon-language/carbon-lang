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

        # Load executable
        self.runCmd("-file-exec-and-symbols %s" % self.myexe)
        self.expect("\^done")

        # Run to program return (marked BP_source)
        line = line_number('main.c', '//BP_source')
        self.runCmd("-break-insert main.c:%d" % line)
        self.expect("\^done,bkpt={number=\"1\"")
        self.runCmd("-exec-run")
        self.expect("\^running")
        self.expect("\*stopped,reason=\"breakpoint-hit\"")

        # Print non-existant variable
        self.runCmd("-var-create var1 * undef")
        #self.expect("\^error") #FIXME: shows undef as {...}
        self.runCmd("-data-evaluate-expression undef")
        self.expect("\^error,msg=\"Could not evaluate expression\"")

        # Print global "g_MyVar", modify, delete and create again
        self.runCmd("-data-evaluate-expression g_MyVar")
        self.expect("\^done,value=\"3\"")
        self.runCmd("-var-create var2 * g_MyVar")
        self.expect("\^done,name=\"var2\",numchild=\"0\",value=\"3\",type=\"int\",thread-id=\"1\",has_more=\"0\"")
        self.runCmd("-var-evaluate-expression var2")
        self.expect("\^done,value=\"3\"")
        self.runCmd("-var-show-attributes var2")
        self.expect("\^done,status=\"editable\"")
        self.runCmd("-data-evaluate-expression \"g_MyVar=30\"")
        self.expect("\^done,value=\"30\"")
        self.runCmd("-var-update var2")
        #self.expect("name=\"var2\",value=\"30\"") #FIXME -var-update doesn't work
        self.runCmd("-var-delete var2")
        self.expect("\^done")
        self.runCmd("-var-create var2 * g_MyVar")
        self.expect("\^done,name=\"var2\",numchild=\"0\",value=\"30\",type=\"int\",thread-id=\"1\",has_more=\"0\"")

        # Print static "s_MyVar", modify, delete and create again
        self.runCmd("-data-evaluate-expression s_MyVar")
        self.expect("\^done,value=\"30\"")
        self.runCmd("-var-create var3 * s_MyVar")
        self.expect("\^done,name=\"var3\",numchild=\"0\",value=\"30\",type=\"int\",thread-id=\"1\",has_more=\"0\"")
        self.runCmd("-var-evaluate-expression var3")
        self.expect("\^done,value=\"30\"")
        self.runCmd("-var-show-attributes var3")
        self.expect("\^done,status=\"editable\"")
        self.runCmd("-data-evaluate-expression \"s_MyVar=3\"")
        self.expect("\^done,value=\"3\"")
        self.runCmd("-var-update var3")
        #self.expect("name=\"var3\",value=\"3\"") #FIXME -var-update doesn't work
        self.runCmd("-var-delete var3")
        self.expect("\^done")
        self.runCmd("-var-create var3 * s_MyVar")
        self.expect("\^done,name=\"var3\",numchild=\"0\",value=\"3\",type=\"int\",thread-id=\"1\",has_more=\"0\"")

        # Print local "b", modify, delete and create again
        self.runCmd("-data-evaluate-expression b")
        self.expect("\^done,value=\"20\"")
        self.runCmd("-var-create var4 * b")
        self.expect("\^done,name=\"var4\",numchild=\"0\",value=\"20\",type=\"int\",thread-id=\"1\",has_more=\"0\"")
        self.runCmd("-var-evaluate-expression var4")
        self.expect("\^done,value=\"20\"")
        self.runCmd("-var-show-attributes var4")
        self.expect("\^done,status=\"editable\"")
        self.runCmd("-data-evaluate-expression \"b=2\"")
        self.expect("\^done,value=\"2\"")
        self.runCmd("-var-update var4")
        #self.expect("name=\"var4\",value=\"2\"") #FIXME -var-update doesn't work
        self.runCmd("-var-delete var4")
        self.expect("\^done")
        self.runCmd("-var-create var4 * b")
        self.expect("\^done,name=\"var4\",numchild=\"0\",value=\"2\",type=\"int\",thread-id=\"1\",has_more=\"0\"")

        # Print temp "a + b"
        self.runCmd("-data-evaluate-expression \"a + b\"")
        self.expect("\^done,value=\"12\"")
        self.runCmd("-var-create var5 * \"a + b\"")
        self.expect("\^done,name=\"var5\",numchild=\"0\",value=\"12\",type=\"int\",thread-id=\"1\",has_more=\"0\"")
        self.runCmd("-var-evaluate-expression var5")
        self.expect("\^done,value=\"12\"")
        self.runCmd("-var-show-attributes var5")
        self.expect("\^done,status=\"editable\"") #FIXME editable or not?

        # Print argument "argv[0]"
        self.runCmd("-data-evaluate-expression \"argv[0]\"")
        self.expect("\^done,value=\"0x[0-9a-f]+\"")
        self.runCmd("-var-create var6 * \"argv[0]\"")
        self.expect("\^done,name=\"var6\",numchild=\"1\",value=\"0x[0-9a-f]+\",type=\"const char \*\",thread-id=\"1\",has_more=\"0\"")
        self.runCmd("-var-evaluate-expression var6")
        self.expect("\^done,value=\"0x[0-9a-f]+\"")
        self.runCmd("-var-show-attributes var6")
        self.expect("\^done,status=\"editable\"")

if __name__ == '__main__':
    unittest2.main()
