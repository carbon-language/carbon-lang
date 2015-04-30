"""
Test lldb-mi -data-xxx commands.
"""

import lldbmi_testcase
from lldbtest import *
import unittest2

class MiDataTestCase(lldbmi_testcase.MiTestCaseBase):

    mydir = TestBase.compute_mydir(__file__)

    @lldbmi_test
    @expectedFailureWindows("llvm.org/pr22274: need a pexpect replacement for windows")
    @skipIfFreeBSD # llvm.org/pr22411: Failure presumably due to known thread races
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
        self.expect("\^done,asm_insns=\[{address=\"0x0*%x\",func-name=\"main\",offset=\"0\",size=\"[1-9]+\",inst=\".+?\"}," % addr)

    @lldbmi_test
    @expectedFailureWindows("llvm.org/pr22274: need a pexpect replacement for windows")
    @skipIfFreeBSD # llvm.org/pr22411: Failure presumably due to known thread races
    @unittest2.skip("-data-evaluate-expression doesn't work") #FIXME: the global case worked before refactoring
    def test_lldbmi_data_read_memory_bytes(self):
        """Test that 'lldb-mi --interpreter' works for -data-read-memory-bytes."""

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

        # Get address of char[] (global)
        self.runCmd("-data-evaluate-expression &g_CharArray")
        self.expect("\^done,value=\"0x[0-9a-f]+\"")
        addr = int(self.child.after.split("\"")[1], 16)
        size = 5

        # Test that -data-read-memory-bytes works for char[] type (global)
        self.runCmd("-data-read-memory-bytes %#x %d" % (addr, size))
        self.expect("\^done,memory=\[{begin=\"0x0*%x\",offset=\"0x0+\",end=\"0x0*%x\",contents=\"1112131400\"}\]" % (addr, addr + size))

        # Get address of static char[]
        self.runCmd("-data-evaluate-expression &s_CharArray")
        self.expect("\^done,value=\"0x[0-9a-f]+\"")
        addr = int(self.child.after.split("\"")[1], 16)
        size = 5

        # Test that -data-read-memory-bytes works for static char[] type
        self.runCmd("-data-read-memory-bytes %#x %d" % (addr, size))
        self.expect("\^done,memory=\[{begin=\"0x0*%x\",offset=\"0x0+\",end=\"0x0*%x\",contents=\"1112131400\"}\]" % (addr, addr + size))

    @lldbmi_test
    @expectedFailureWindows("llvm.org/pr22274: need a pexpect replacement for windows")
    @skipIfFreeBSD # llvm.org/pr22411: Failure presumably due to known thread races
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
        self.expect("\^done,register-names=\[\".+?\",")

        # Test -data-list-register-names: try to get specified registers
        self.runCmd("-data-list-register-names 0")
        self.expect("\^done,register-names=\[\".+?\"\]")

    @lldbmi_test
    @expectedFailureWindows("llvm.org/pr22274: need a pexpect replacement for windows")
    @skipIfFreeBSD # llvm.org/pr22411: Failure presumably due to known thread races
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

    @lldbmi_test
    @expectedFailureWindows("llvm.org/pr22274: need a pexpect replacement for windows")
    @skipIfFreeBSD # llvm.org/pr22411: Failure presumably due to known thread races
    def test_lldbmi_data_info_line(self):
        """Test that 'lldb-mi --interpreter' works for -data-info-line."""

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

        # Get the address of main and its line
        self.runCmd("-data-evaluate-expression main")
        self.expect("\^done,value=\"0x[0-9a-f]+\"")
        addr = int(self.child.after.split("\"")[1], 16)
        line = line_number('main.cpp', '// FUNC_main')

        # Test that -data-info-line works for address
        self.runCmd("-data-info-line *%#x" % addr)
        self.expect("\^done,start=\"0x0*%x\",end=\"0x[0-9a-f]+\",file=\".+?main.cpp\",line=\"%d\"" % (addr, line))

        # Test that -data-info-line works for file:line
        self.runCmd("-data-info-line main.cpp:%d" % line)
        self.expect("\^done,start=\"0x0*%x\",end=\"0x[0-9a-f]+\",file=\".+?main.cpp\",line=\"%d\"" % (addr, line))

        # Test that -data-info-line fails when invalid address is specified
        self.runCmd("-data-info-line *0x0")
        self.expect("\^error,msg=\"Command 'data-info-line'\. Error: The LineEntry is absent or has an unknown format\.\"")

        # Test that -data-info-line fails when file is unknown
        self.runCmd("-data-info-line unknown_file:1")
        self.expect("\^error,msg=\"Command 'data-info-line'\. Error: The LineEntry is absent or has an unknown format\.\"")

        # Test that -data-info-line fails when line has invalid format
        self.runCmd("-data-info-line main.cpp:bad_line")
        self.expect("\^error,msg=\"error: invalid line number string 'bad_line'")
        self.runCmd("-data-info-line main.cpp:0")
        self.expect("\^error,msg=\"error: zero is an invalid line number")

if __name__ == '__main__':
    unittest2.main()
