"""
Test that the lldb-mi driver works with -stack-xxx commands
"""

import lldbmi_testcase
from lldbtest import *
import unittest2

class MiStackTestCase(lldbmi_testcase.MiTestCaseBase):

    @lldbmi_test
    @expectedFailureWindows("llvm.org/pr22274: need a pexpect replacement for windows")
    def test_lldbmi_stackargs(self):
        """Test that 'lldb-mi --interpreter' can shows arguments."""

        self.spawnLldbMi(args = None)

        # Load executable
        self.runCmd("-file-exec-and-symbols %s" % self.myexe)
        self.expect("\^done")

        # Run to main
        self.runCmd("-break-insert -f main")
        self.expect("\^done,bkpt={number=\"1\"")
        self.runCmd("-exec-run")
        self.expect("\^running")
        self.expect("\*stopped,reason=\"breakpoint-hit\"")

        # Test -stack-list-arguments: use 0 or --no-values
        self.runCmd("-stack-list-arguments 0")
        self.expect("\^done,stack-args=\[frame={level=\"0\",args=\[name=\"argc\",name=\"argv\"\]}")
        self.runCmd("-stack-list-arguments --no-values")
        self.expect("\^done,stack-args=\[frame={level=\"0\",args=\[name=\"argc\",name=\"argv\"\]}")

        # Test -stack-list-arguments: use 1 or --all-values
        self.runCmd("-stack-list-arguments 1")
        self.expect("\^done,stack-args=\[frame={level=\"0\",args=\[{name=\"argc\",value=\"1\"},{name=\"argv\",value=\".*\"}\]}")
        self.runCmd("-stack-list-arguments --all-values")
        self.expect("\^done,stack-args=\[frame={level=\"0\",args=\[{name=\"argc\",value=\"1\"},{name=\"argv\",value=\".*\"}\]}")

        # Test -stack-list-arguments: use 2 or --simple-values
        self.runCmd("-stack-list-arguments 2")
        self.expect("\^done,stack-args=\[frame={level=\"0\",args=\[{name=\"argc\",value=\"1\"},{name=\"argv\",value=\".*\"}\]}")
        self.runCmd("-stack-list-arguments --simple-values")
        self.expect("\^done,stack-args=\[frame={level=\"0\",args=\[{name=\"argc\",value=\"1\"},{name=\"argv\",value=\".*\"}\]}")

    @lldbmi_test
    @expectedFailureWindows("llvm.org/pr22274: need a pexpect replacement for windows")
    def test_lldbmi_locals(self):
        """Test that 'lldb-mi --interpreter' can shows local variables."""

        self.spawnLldbMi(args = None)

        # Load executable
        self.runCmd("-file-exec-and-symbols %s" % self.myexe)
        self.expect("\^done")

        # Run to main
        line = line_number('main.c', '//BP_localstest')
        self.runCmd("-break-insert --file main.c:%d" % line)
        self.expect("\^done,bkpt={number=\"1\"")
        self.runCmd("-exec-run")
        self.expect("\^running")
        self.expect("\*stopped,reason=\"breakpoint-hit\"")

        # Test -stack-list-locals: use 0 or --no-values
        self.runCmd("-stack-list-locals 0")
        self.expect("\^done,locals=\[name=\"a\",name=\"b\"\]")
        self.runCmd("-stack-list-locals --no-values")
        self.expect("\^done,locals=\[name=\"a\",name=\"b\"\]")

        # Test -stack-list-locals: use 1 or --all-values
        self.runCmd("-stack-list-locals 1")
        self.expect("\^done,locals=\[{name=\"a\",value=\"10\"},{name=\"b\",value=\"20\"}\]")
        self.runCmd("-stack-list-locals --all-values")
        self.expect("\^done,locals=\[{name=\"a\",value=\"10\"},{name=\"b\",value=\"20\"}\]")

        # Test -stack-list-locals: use 2 or --simple-values
        self.runCmd("-stack-list-locals 2")
        self.expect("\^done,locals=\[{name=\"a\",value=\"10\"},{name=\"b\",value=\"20\"}\]")
        self.runCmd("-stack-list-locals --simple-values")
        self.expect("\^done,locals=\[{name=\"a\",value=\"10\"},{name=\"b\",value=\"20\"}\]")

    @lldbmi_test
    @expectedFailureWindows("llvm.org/pr22274: need a pexpect replacement for windows")
    def test_lldbmi_stackdepth(self):
        """Test that 'lldb-mi --interpreter' can shows depth of the stack."""

        self.spawnLldbMi(args = None)

        # Load executable
        self.runCmd("-file-exec-and-symbols %s" % self.myexe)
        self.expect("\^done")

        # Run to main
        self.runCmd("-break-insert -f main")
        self.expect("\^done,bkpt={number=\"1\"")
        self.runCmd("-exec-run")
        self.expect("\^running")
        self.expect("\*stopped,reason=\"breakpoint-hit\"")

        # Test stack depth
        self.runCmd("-stack-info-depth")
        self.expect("\^done,depth=\"[1-9]\"")

    @lldbmi_test
    @expectedFailureWindows("llvm.org/pr22274: need a pexpect replacement for windows")
    def test_lldbmi_stackframes(self):
        """Test that 'lldb-mi --interpreter' can lists the frames on the stack."""

        self.spawnLldbMi(args = None)

        # Load executable
        self.runCmd("-file-exec-and-symbols %s" % self.myexe)
        self.expect("\^done")

        # Run to main
        self.runCmd("-break-insert -f main")
        self.expect("\^done,bkpt={number=\"1\"")
        self.runCmd("-exec-run")
        self.expect("\^running")
        self.expect("\*stopped,reason=\"breakpoint-hit\"")

        # Test stack frame: get frame #0 info
        self.runCmd("-stack-list-frames 0 0")
        self.expect("\^done,stack=\[frame=\{level=\"0\",addr=\".+\",func=\"main\",file=\"main\.c\",fullname=\".*main\.c\",line=\".+\"\}\]")

if __name__ == '__main__':
    unittest2.main()
