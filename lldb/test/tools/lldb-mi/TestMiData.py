"""
Test that the lldb-mi driver works with -data-xxx commands
"""

import lldbmi_testcase
from lldbtest import *
import unittest2

class MiDataTestCase(lldbmi_testcase.MiTestCaseBase):

    @lldbmi_test
    @expectedFailureWindows("llvm.org/pr22274: need a pexpect replacement for windows")
    def test_lldbmi_data_disassemble(self):
        """Test that 'lldb-mi --interpreter' works for -data-disassemble."""

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

        # Get an address for disassembling: use main
        self.runCmd("-data-evaluate-expression main")
        self.expect("\^done,value=\"0x[0-9a-f]+\"")
        addr = int(self.child.after.split("\"")[1], 16)

        # Test -data-disassemble: try to disassemble some address
        self.runCmd("-data-disassemble -s %#x -e %#x -- 0" % (addr, addr + 0x10))
        self.expect("\^done,asm_insns=\[{address=\"%#x\",func-name=\"main\",offset=\"0x0\",size=\"[1-9]\",inst=\".+\"}," % addr)

    @lldbmi_test
    @expectedFailureWindows("llvm.org/pr22274: need a pexpect replacement for windows")
    def test_lldbmi_data_list_register_names(self):
        """Test that 'lldb-mi --interpreter' works for -data-list-register-names."""

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

        # Test -data-list-register-names: try to get all registers
        self.runCmd("-data-list-register-names")
        self.expect("\^done,register-names=\[\".+\",")

        # Test -data-list-register-names: try to get specified registers
        self.runCmd("-data-list-register-names 0")
        self.expect("\^done,register-names=\[\".+\"\]")

    @lldbmi_test
    @expectedFailureWindows("llvm.org/pr22274: need a pexpect replacement for windows")
    def test_lldbmi_data_list_register_values(self):
        """Test that 'lldb-mi --interpreter' works for -data-list-register-values."""

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

        # Test -data-list-register-values: try to get all registers
        self.runCmd("-data-list-register-values x")
        self.expect("\^done,register-values=\[{number=\"0\",value=\"0x[0-9a-f]+\"")

        # Test -data-list-register-values: try to get specified registers
        self.runCmd("-data-list-register-values x 0")
        self.expect("\^done,register-values=\[{number=\"0\",value=\"0x[0-9a-f]+\"}\]")

if __name__ == '__main__':
    unittest2.main()
