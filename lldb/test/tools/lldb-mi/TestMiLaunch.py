"""
Test various ways the lldb-mi driver can launch a program.
"""

import lldbmi_testcase
from lldbtest import *
import unittest2

class MiLaunchTestCase(lldbmi_testcase.MiTestCaseBase):

    @lldbmi_test
    @expectedFailureWindows("llvm.org/pr22274: need a pexpect replacement for windows")
    def test_lldbmi_exe(self):
        """Test that 'lldb-mi --interpreter' works for -file-exec-and-symbols exe."""

        self.spawnLldbMi(args = None)

        #use no path
        self.runCmd("-file-exec-and-symbols %s" % self.myexe)
        self.expect("\^done")

        self.runCmd("-exec-run")
        self.expect("\^running")
        self.expect("\*stopped,reason=\"exited-normally\"")

    @lldbmi_test
    @expectedFailureWindows("llvm.org/pr22274: need a pexpect replacement for windows")
    def test_lldbmi_abspathexe(self):
        """Test that 'lldb-mi --interpreter' works for -file-exec-and-symbols fullpath/exe."""

        self.spawnLldbMi(args = None)

        #use full path
        import os
        exe = os.path.join(os.getcwd(), self.myexe)
        self.runCmd("-file-exec-and-symbols %s" % exe)
        self.expect("\^done")

        self.runCmd("-exec-run")
        self.expect("\^running")
        self.expect("\*stopped,reason=\"exited-normally\"")

    @lldbmi_test
    @expectedFailureWindows("llvm.org/pr22274: need a pexpect replacement for windows")
    def test_lldbmi_relpathexe(self):
        """Test that 'lldb-mi --interpreter' works for -file-exec-and-symbols relpath/exe."""

        self.spawnLldbMi(args = None)

        #use relative path
        exe = "../../" + self.mydir + "/" + self.myexe
        self.runCmd("-file-exec-and-symbols %s" % exe)
        self.expect("\^done")

        self.runCmd("-exec-run")
        self.expect("\^running")
        self.expect("\*stopped,reason=\"exited-normally\"")

    @lldbmi_test
    @expectedFailureWindows("llvm.org/pr22274: need a pexpect replacement for windows")
    def test_lldbmi_badpathexe(self):
        """Test that 'lldb-mi --interpreter' works for -file-exec-and-symbols badpath/exe."""

        self.spawnLldbMi(args = None)

        #use non-existant path
        exe = "badpath/" + self.myexe
        self.runCmd("-file-exec-and-symbols %s" % exe)
        self.expect("\^error")

if __name__ == '__main__':
    unittest2.main()
