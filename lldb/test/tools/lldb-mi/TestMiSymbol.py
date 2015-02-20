"""
Test that the lldb-mi driver works with -symbol-xxx commands
"""

import lldbmi_testcase
from lldbtest import *
import unittest2

class MiSymbolTestCase(lldbmi_testcase.MiTestCaseBase):

    @lldbmi_test
    @expectedFailureWindows("llvm.org/pr22274: need a pexpect replacement for windows")
    @skipIfFreeBSD # llvm.org/pr22411: Failure presumably due to known thread races
    @skipIfLinux # llvm.org/pr22411: Failure presumably due to known thread races
    def test_lldbmi_symbol_list_lines_file(self):
        """Test that 'lldb-mi --interpreter' works for -symbol-list-lines when file exists."""

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

        # Get address of main
        self.runCmd("-data-evaluate-expression main")
        self.expect("\^done,value=\"0x[0-9a-f]+\"")
        main_addr = int(self.child.after.split("\"")[1], 16)
        main_line = line_number('main.c', '//FUNC_main')

        # Test that -symbol-list-lines works on valid data
        self.runCmd("-symbol-list-lines main.c")
        self.expect("\^done,lines=\[\{pc=\"0x0*%x\",line=\"%d\"\}(,\{pc=\"0x[0-9a-f]+\",line=\"\d+\"\})+\]" % (main_addr, main_line))

        # Test that -symbol-list-lines fails when file doesn't exist
        self.runCmd("-symbol-list-lines unknown_file")
        self.expect("\^error,message=\"warning: No source filenames matched 'unknown_file'. error: no source filenames matched any command arguments \"")

        # Test that -symbol-list-lines fails when file is specified using relative path
        self.runCmd("-symbol-list-lines ./main.c")
        self.expect("\^error,message=\"warning: No source filenames matched './main.c'. error: no source filenames matched any command arguments \"")

        # Test that -symbol-list-lines works when file is specified using absolute path
        import os
        main_file = os.path.join(os.getcwd(), "main.c")
        self.runCmd("-symbol-list-lines \"%s\"" % main_file)
        self.expect("\^done,lines=\[\{pc=\"0x0*%x\",line=\"%d\"\}(,\{pc=\"0x[0-9a-f]+\",line=\"\d+\"\})+\]" % (main_addr, main_line))

        # Test that -symbol-list-lines fails when file doesn't exist
        self.runCmd("-symbol-list-lines unknown_dir/main.c")
        self.expect("\^error,message=\"warning: No source filenames matched 'unknown_dir/main.c'. error: no source filenames matched any command arguments \"")

if __name__ == '__main__':
    unittest2.main()
