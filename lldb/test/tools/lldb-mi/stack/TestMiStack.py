"""
Test lldb-mi -stack-xxx commands.
"""

# adjust path for lldbmi_testcase.py
import sys, os.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import lldbmi_testcase
from lldbtest import *
import unittest2

class MiStackTestCase(lldbmi_testcase.MiTestCaseBase):

    mydir = TestBase.compute_mydir(__file__)

    @lldbmi_test
    @expectedFailureWindows("llvm.org/pr22274: need a pexpect replacement for windows")
    @skipIfFreeBSD # llvm.org/pr22411: Failure presumably due to known thread races
    def test_lldbmi_stack_list_arguments(self):
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
    @skipIfFreeBSD # llvm.org/pr22411: Failure presumably due to known thread races
    def test_lldbmi_stack_list_locals(self):
        """Test that 'lldb-mi --interpreter' can shows local variables."""

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

        # Test int local variables:
        # Run to BP_local_int_test
        line = line_number('main.cpp', '// BP_local_int_test')
        self.runCmd("-break-insert --file main.cpp:%d" % line)
        self.expect("\^done,bkpt={number=\"2\"")
        self.runCmd("-exec-continue")
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
        
        # Test struct local variable:
        # Run to BP_local_struct_test
        line = line_number('main.cpp', '// BP_local_struct_test')
        self.runCmd("-break-insert --file main.cpp:%d" % line)
        self.expect("\^done,bkpt={number=\"3\"")
        self.runCmd("-exec-continue")
        self.expect("\^running")
        self.expect("\*stopped,reason=\"breakpoint-hit\"")
        
        # Test -stack-list-locals: use 0 or --no-values
        self.runCmd("-stack-list-locals 0")
        self.expect("\^done,locals=\[name=\"var_c\"\]")
        self.runCmd("-stack-list-locals --no-values")
        self.expect("\^done,locals=\[name=\"var_c\"\]")

        # Test -stack-list-locals: use 1 or --all-values
        self.runCmd("-stack-list-locals 1")
        self.expect("\^done,locals=\[{name=\"var_c\",value=\"{var_a = 10,var_b = 97 'a',inner_ = { var_d = 30 }}\"}\]")
        self.runCmd("-stack-list-locals --all-values")
        self.expect("\^done,locals=\[{name=\"var_c\",value=\"{var_a = 10,var_b = 97 'a',inner_ = { var_d = 30 }}\"}\]")

        # Test -stack-list-locals: use 2 or --simple-values
        self.runCmd("-stack-list-locals 2")
        self.expect("\^done,locals=\[name=\"var_c\"\]")
        self.runCmd("-stack-list-locals --simple-values")
        self.expect("\^done,locals=\[name=\"var_c\"\]")
        
        # Test array local variable:
        # Run to BP_local_array_test
        line = line_number('main.cpp', '// BP_local_array_test')
        self.runCmd("-break-insert --file main.cpp:%d" % line)
        self.expect("\^done,bkpt={number=\"4\"")
        self.runCmd("-exec-continue")
        self.expect("\^running")
        self.expect("\*stopped,reason=\"breakpoint-hit\"")
        
        # Test -stack-list-locals: use 0 or --no-values
        self.runCmd("-stack-list-locals 0")
        self.expect("\^done,locals=\[name=\"array\"\]")
        self.runCmd("-stack-list-locals --no-values")
        self.expect("\^done,locals=\[name=\"array\"\]")

        # Test -stack-list-locals: use 1 or --all-values
        self.runCmd("-stack-list-locals 1")
        self.expect("\^done,locals=\[{name=\"array\",value=\"{\[0\] = 100,\[1\] = 200,\[2\] = 300}\"}\]")
        self.runCmd("-stack-list-locals --all-values")
        self.expect("\^done,locals=\[{name=\"array\",value=\"{\[0\] = 100,\[1\] = 200,\[2\] = 300}\"}\]")

        # Test -stack-list-locals: use 2 or --simple-values
        self.runCmd("-stack-list-locals 2")
        self.expect("\^done,locals=\[name=\"array\"\]")
        self.runCmd("-stack-list-locals --simple-values")
        self.expect("\^done,locals=\[name=\"array\"\]")
        
        # Test pointers as local variable:
        # Run to BP_local_pointer_test
        line = line_number('main.cpp', '// BP_local_pointer_test')
        self.runCmd("-break-insert --file main.cpp:%d" % line)
        self.expect("\^done,bkpt={number=\"5\"")
        self.runCmd("-exec-continue")
        self.expect("\^running")
        self.expect("\*stopped,reason=\"breakpoint-hit\"")
        
        # Test -stack-list-locals: use 0 or --no-values
        self.runCmd("-stack-list-locals 0")
        self.expect("\^done,locals=\[name=\"test_str\",name=\"var_e\",name=\"ptr\"\]")
        self.runCmd("-stack-list-locals --no-values")
        self.expect("\^done,locals=\[name=\"test_str\",name=\"var_e\",name=\"ptr\"\]")

        # Test -stack-list-locals: use 1 or --all-values
        self.runCmd("-stack-list-locals 1")
        self.expect("\^done,locals=\[{name=\"test_str\",value=\".*Rakaposhi.*\"},{name=\"var_e\",value=\"24\"},{name=\"ptr\",value=\".*\"}\]")
        self.runCmd("-stack-list-locals --all-values")
        self.expect("\^done,locals=\[{name=\"test_str\",value=\".*Rakaposhi.*\"},{name=\"var_e\",value=\"24\"},{name=\"ptr\",value=\".*\"}\]")

        # Test -stack-list-locals: use 2 or --simple-values
        self.runCmd("-stack-list-locals 2")
        self.expect("\^done,locals=\[{name=\"test_str\",value=\".*Rakaposhi.*\"},{name=\"var_e\",value=\"24\"},{name=\"ptr\",value=\".*\"}\]")
        self.runCmd("-stack-list-locals --simple-values")
        self.expect("\^done,locals=\[{name=\"test_str\",value=\".*Rakaposhi.*\"},{name=\"var_e\",value=\"24\"},{name=\"ptr\",value=\".*\"}\]")

    @lldbmi_test
    @expectedFailureWindows("llvm.org/pr22274: need a pexpect replacement for windows")
    @skipIfFreeBSD # llvm.org/pr22411: Failure presumably due to known thread races
    def test_lldbmi_stack_info_depth(self):
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
    @skipIfFreeBSD # llvm.org/pr22411: Failure presumably due to known thread races
    def test_lldbmi_stack_info_frame(self):
        """Test that 'lldb-mi --interpreter' can show information about current frame."""

        self.spawnLldbMi(args = None)

        # Test that -stack-info-frame fails when program isn't running
        self.runCmd("-stack-info-frame")
        self.expect("\^error,msg=\"Command 'stack-info-frame'. Invalid process during debug session\"")

        # Load executable
        self.runCmd("-file-exec-and-symbols %s" % self.myexe)
        self.expect("\^done")

        # Run to main
        self.runCmd("-break-insert -f main")
        self.expect("\^done,bkpt={number=\"1\"")
        self.runCmd("-exec-run")
        self.expect("\^running")
        self.expect("\*stopped,reason=\"breakpoint-hit\"")

        # Test that -stack-info-frame works when program is running
        self.runCmd("-stack-info-frame")
        self.expect("\^done,frame=\{level=\"0\",addr=\".+\",func=\"main\",file=\"main\.cpp\",fullname=\".*main\.cpp\",line=\"\d+\"\}")

    @lldbmi_test
    @expectedFailureWindows("llvm.org/pr22274: need a pexpect replacement for windows")
    @skipIfFreeBSD # llvm.org/pr22411: Failure presumably due to known thread races
    def test_lldbmi_stack_list_frames(self):
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
        self.expect("\^done,stack=\[frame=\{level=\"0\",addr=\".+\",func=\"main\",file=\"main\.cpp\",fullname=\".*main\.cpp\",line=\".+\"\}\]")

    @lldbmi_test
    @expectedFailureWindows("llvm.org/pr22274: need a pexpect replacement for windows")
    @skipIfFreeBSD # llvm.org/pr22411: Failure presumably due to known thread races
    def test_lldbmi_stack_select_frame(self):
        """Test that 'lldb-mi --interpreter' can choose current frame."""

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

        # Test that -stack-select-frame requires 1 mandatory argument
        self.runCmd("-stack-select-frame")
        self.expect("\^error,msg=\"Command 'stack-select-frame'. Command Args. Missing options, 1 or more required\"")

        # Test that -stack-select-frame fails on invalid frame number
        self.runCmd("-stack-select-frame 99")
        self.expect("\^error,msg=\"Command 'stack-select-frame'. Frame ID invalid\"")

        # Test that current frame is #0
        self.runCmd("-stack-info-frame")
        self.expect("\^done,frame=\{level=\"0\",addr=\".+\",func=\"main\",file=\"main\.cpp\",fullname=\".*main\.cpp\",line=\"\d+\"\}")

        # Test that -stack-select-frame can select the selected frame
        self.runCmd("-stack-select-frame 0")
        self.expect("\^done")

        # Test that current frame is still #0
        self.runCmd("-stack-info-frame")
        self.expect("\^done,frame=\{level=\"0\",addr=\".+\",func=\"main\",file=\"main\.cpp\",fullname=\".*main\.cpp\",line=\"\d+\"\}")

        # Test that -stack-select-frame can select frame #1 (parent frame)
        self.runCmd("-stack-select-frame 1")
        self.expect("\^done")

        # Test that current frame is #1
        self.runCmd("-stack-info-frame")
        self.expect("\^done,frame=\{level=\"1\",addr=\".+\",func=\".+\",file=\"\?\?\",fullname=\"\?\?\",line=\"-1\"\}")

        # Test that -stack-select-frame can select frame #0 (child frame)
        self.runCmd("-stack-select-frame 0")
        self.expect("\^done")

        # Test that current frame is #0 and it has the same information
        self.runCmd("-stack-info-frame")
        self.expect("\^done,frame=\{level=\"0\",addr=\".+\",func=\"main\",file=\"main\.cpp\",fullname=\".*main\.cpp\",line=\"\d+\"\}")

if __name__ == '__main__':
    unittest2.main()
