"""
Test lldb-mi -break-xxx commands.
"""

import lldbmi_testcase
from lldbtest import *
import unittest2

class MiBreakTestCase(lldbmi_testcase.MiTestCaseBase):

    mydir = TestBase.compute_mydir(__file__)

    @lldbmi_test
    @expectedFailureWindows("llvm.org/pr22274: need a pexpect replacement for windows")
    @skipIfFreeBSD # llvm.org/pr22411: Failure presumably due to known thread races
    def test_lldbmi_break_insert_function_pending(self):
        """Test that 'lldb-mi --interpreter' works for pending function breakpoints."""

        self.spawnLldbMi(args = None)

        self.runCmd("-file-exec-and-symbols %s" % self.myexe)
        self.expect("\^done")

        self.runCmd("-break-insert -f printf")
        #FIXME function name is unknown on Darwin, fullname should be ??, line is -1
        #self.expect("\^done,bkpt={number=\"1\",type=\"breakpoint\",disp=\"keep\",enabled=\"y\",addr=\"0xffffffffffffffff\",func=\"printf\",file=\"\?\?\",fullname=\"\?\?\",line=\"-1\",pending=\[\"printf\"\],times=\"0\",original-location=\"printf\"}")
        self.expect("\^done,bkpt={number=\"1\",type=\"breakpoint\",disp=\"keep\",enabled=\"y\",addr=\"0xffffffffffffffff\",func=\"\?\?\",file=\"\?\?\",fullname=\"\?\?/\?\?\",line=\"0\",pending=\[\"printf\"\],times=\"0\",original-location=\"printf\"}")
        #FIXME function name is unknown on Darwin, =breakpoint-created is treated as =breakpoint-modified, fullname should be ??, line -1
        #self.expect("=breakpoint-created,bkpt={number=\"1\",type=\"breakpoint\",disp=\"keep\",enabled=\"y\",addr=\"0xffffffffffffffff\",func=\"printf\",file=\"\?\?\",fullname=\"\?\?\",line=\"-1\",pending=\[\"printf\"\],times=\"0\",original-location=\"printf\"}")
        self.expect("=breakpoint-modified,bkpt={number=\"1\",type=\"breakpoint\",disp=\"keep\",enabled=\"y\",addr=\"0xffffffffffffffff\",func=\"\?\?\",file=\"\?\?\",fullname=\"\?\?/\?\?\",line=\"0\",pending=\[\"printf\"\],times=\"0\",original-location=\"printf\"}")

        self.runCmd("-exec-run")
        self.expect("\^running")
        self.expect("=breakpoint-modified,bkpt={number=\"1\",type=\"breakpoint\",disp=\"keep\",enabled=\"y\",addr=\"(?!0xffffffffffffffff)0x[0-9a-f]+\",func=\".+?\",file=\".+?\",fullname=\".+?\",line=\"(-1|\d+)\",pending=\[\"printf\"\],times=\"0\",original-location=\"printf\"}")
        self.expect("\*stopped,reason=\"breakpoint-hit\"")

    @lldbmi_test
    @expectedFailureWindows("llvm.org/pr22274: need a pexpect replacement for windows")
    @skipIfFreeBSD # llvm.org/pr22411: Failure presumably due to known thread races
    def test_lldbmi_break_insert_function(self):
        """Test that 'lldb-mi --interpreter' works for function breakpoints."""

        self.spawnLldbMi(args = None)

        self.runCmd("-file-exec-and-symbols %s" % self.myexe)
        self.expect("\^done")

        self.runCmd("-break-insert -f main")
        self.expect("\^done,bkpt={number=\"1\",type=\"breakpoint\",disp=\"keep\",enabled=\"y\",addr=\"(?!0xffffffffffffffff)0x[0-9a-f]+\",func=\"main\",file=\"main\.cpp\",fullname=\".+?main\.cpp\",line=\"\d+\",pending=\[\"main\"\],times=\"0\",original-location=\"main\"}")
        #FIXME =breakpoint-created is treated as =breakpoint-modified
        #self.expect("=breakpoint-created,bkpt={number=\"1\",type=\"breakpoint\",disp=\"keep\",enabled=\"y\",addr=\"(?!0xffffffffffffffff)0x[0-9a-f]+\",func=\"main\",file=\"main\.cpp\",fullname=\".+?main\.cpp\",line=\"\d+\",pending=\[\"main\"\],times=\"0\",original-location=\"main\"}")
        self.expect("=breakpoint-modified,bkpt={number=\"1\",type=\"breakpoint\",disp=\"keep\",enabled=\"y\",addr=\"(?!0xffffffffffffffff)0x[0-9a-f]+\",func=\"main\",file=\"main\.cpp\",fullname=\".+?main\.cpp\",line=\"\d+\",pending=\[\"main\"\],times=\"0\",original-location=\"main\"}")

        self.runCmd("-exec-run")
        self.expect("\^running")
        self.expect("=breakpoint-modified,bkpt={number=\"1\",type=\"breakpoint\",disp=\"keep\",enabled=\"y\",addr=\"(?!0xffffffffffffffff)0x[0-9a-f]+\",func=\"main\",file=\"main\.cpp\",fullname=\".+?main\.cpp\",line=\"\d+\",pending=\[\"main\"\],times=\"0\",original-location=\"main\"}")
        self.expect("\*stopped,reason=\"breakpoint-hit\"")

        self.runCmd("-break-insert printf")
        #FIXME function name is unknown on Darwin
        #self.expect("\^done,bkpt={number=\"2\",type=\"breakpoint\",disp=\"keep\",enabled=\"y\",addr=\"(?!0xffffffffffffffff)0x[0-9a-f]+\",func=\"printf\",file=\".+?\",fullname=\".+?\",line=\"(-1|\d+)\",times=\"0\",original-location=\"printf\"}")
        self.expect("\^done,bkpt={number=\"2\",type=\"breakpoint\",disp=\"keep\",enabled=\"y\",addr=\"(?!0xffffffffffffffff)0x[0-9a-f]+\",func=\".+?\",file=\".+?\",fullname=\".+?\",line=\"(-1|\d+)\",times=\"0\",original-location=\"printf\"}")
        #FIXME function name is unknown on Darwin, =breakpoint-created is treated as =breakpoint-modified
        #self.expect("=breakpoint-created,bkpt={number=\"2\",type=\"breakpoint\",disp=\"keep\",enabled=\"y\",addr=\"(?!0xffffffffffffffff)0x[0-9a-f]+\",func=\"printf\",file=\".+?\",fullname=\".+?\",line=\"(-1|\d+)\",times=\"0\",original-location=\"printf\"}")
        self.expect("=breakpoint-modified,bkpt={number=\"2\",type=\"breakpoint\",disp=\"keep\",enabled=\"y\",addr=\"(?!0xffffffffffffffff)0x[0-9a-f]+\",func=\".+?\",file=\".+?\",fullname=\".+?\",line=\"(-1|\d+)\",times=\"0\",original-location=\"printf\"}")
        # FIXME function name is unknown on Darwin
        #self.expect("=breakpoint-modified,bkpt={number=\"2\",type=\"breakpoint\",disp=\"keep\",enabled=\"y\",addr=\"(?!0xffffffffffffffff)0x[0-9a-f]+\",func=\"printf\",file=\".+?\",fullname=\".+?\",line=\"(-1|\d+)\",times=\"0\",original-location=\"printf\"}")
        self.expect("=breakpoint-modified,bkpt={number=\"2\",type=\"breakpoint\",disp=\"keep\",enabled=\"y\",addr=\"(?!0xffffffffffffffff)0x[0-9a-f]+\",func=\".+?\",file=\".+?\",fullname=\".+?\",line=\"(-1|\d+)\",times=\"0\",original-location=\"printf\"}")

        self.runCmd("-exec-continue")
        self.expect("\^running")
        self.expect("\*stopped,reason=\"breakpoint-hit\"")

    @lldbmi_test
    @expectedFailureWindows("llvm.org/pr22274: need a pexpect replacement for windows")
    @skipIfFreeBSD # llvm.org/pr22411: Failure presumably due to known thread races
    def test_lldbmi_break_insert_file_line_pending(self):
        """Test that 'lldb-mi --interpreter' works for pending file:line breakpoints."""

        self.spawnLldbMi(args = None)

        self.runCmd("-file-exec-and-symbols %s" % self.myexe)
        self.expect("\^done")

        # Find the line number to break inside main() and set
        # pending BP
        line = line_number('main.cpp', '// BP_return')
        self.runCmd("-break-insert -f main.cpp:%d" % line)
        self.expect("\^done,bkpt={number=\"1\"")

        self.runCmd("-exec-run")
        self.expect("\^running")
        self.expect("\*stopped,reason=\"breakpoint-hit\"")

    @lldbmi_test
    @expectedFailureWindows("llvm.org/pr22274: need a pexpect replacement for windows")
    @skipIfFreeBSD # llvm.org/pr22411: Failure presumably due to known thread races
    def test_lldbmi_break_insert_file_line(self):
        """Test that 'lldb-mi --interpreter' works for file:line breakpoints."""

        self.spawnLldbMi(args = None)

        self.runCmd("-file-exec-and-symbols %s" % self.myexe)
        self.expect("\^done")

        self.runCmd("-break-insert -f main")
        self.expect("\^done,bkpt={number=\"1\"")

        self.runCmd("-exec-run")
        self.expect("\^running")
        self.expect("\*stopped,reason=\"breakpoint-hit\"")

        line = line_number('main.cpp', '// BP_return')
        self.runCmd("-break-insert main.cpp:%d" % line)
        self.expect("\^done,bkpt={number=\"2\"")

        self.runCmd("-exec-continue")
        self.expect("\^running")
        self.expect("\*stopped,reason=\"breakpoint-hit\"")

    @lldbmi_test
    @expectedFailureWindows("llvm.org/pr22274: need a pexpect replacement for windows")
    @skipIfFreeBSD # llvm.org/pr22411: Failure presumably due to known thread races
    @unittest2.expectedFailure("-break-insert doesn't work for absolute path")
    def test_lldbmi_break_insert_file_line_absolute_path(self):
        """Test that 'lldb-mi --interpreter' works for file:line breakpoints."""

        self.spawnLldbMi(args = None)

        self.runCmd("-file-exec-and-symbols %s" % self.myexe)
        self.expect("\^done")

        self.runCmd("-break-insert -f main")
        self.expect("\^done,bkpt={number=\"1\"")

        self.runCmd("-exec-run")
        self.expect("\^running")
        self.expect("\*stopped,reason=\"breakpoint-hit\"")

        import os
        path = os.path.join(os.getcwd(), "main.cpp")
        line = line_number('main.cpp', '// BP_return')
        self.runCmd("-break-insert %s:%d" % (path, line))
        self.expect("\^done,bkpt={number=\"2\"")

        self.runCmd("-exec-continue")
        self.expect("\^running")
        self.expect("\*stopped,reason=\"breakpoint-hit\"")

if __name__ == '__main__':
    unittest2.main()
