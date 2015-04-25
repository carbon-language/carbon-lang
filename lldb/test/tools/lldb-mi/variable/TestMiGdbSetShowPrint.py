"""
Test lldb-mi -gdb-set and -gdb-show commands for 'print option-name'.
"""

import lldbmi_testcase
from lldbtest import *
import unittest2

class MiGdbSetShowTestCase(lldbmi_testcase.MiTestCaseBase):

    mydir = TestBase.compute_mydir(__file__)

    @lldbmi_test
    @expectedFailureWindows("llvm.org/pr22274: need a pexpect replacement for windows")
    def test_lldbmi_gdb_set_show_print_char_array_as_string(self):
        """Test that 'lldb-mi --interpreter' can print array of chars as string."""

        self.spawnLldbMi(args = None)

        # Load executable
        self.runCmd("-file-exec-and-symbols %s" % self.myexe)
        self.expect("\^done")

        # Run to BP_gdb_set_show_print_char_array_as_string_test
        line = line_number('main.cpp', '// BP_gdb_set_show_print_char_array_as_string_test')
        self.runCmd("-break-insert main.cpp:%d" % line)
        self.expect("\^done,bkpt={number=\"1\"")
        self.runCmd("-exec-run")
        self.expect("\^running")
        self.expect("\*stopped,reason=\"breakpoint-hit\"")

        # Test that default print char-array-as-string value is "off"
        self.runCmd("-gdb-show print char-array-as-string")
        self.expect("\^done,value=\"off\"")

        # Test that an char* is expanded to string when print char-array-as-string is "off"
        self.runCmd("-var-create var1 * string_ptr")
        self.expect("\^done,name=\"var1\",numchild=\"1\",value=\"0x[0-9a-f]+ \\\\\\\"string - const char \*\\\\\\\"\",type=\"const char \*\",thread-id=\"1\",has_more=\"0\"")

        # Test that an char[] isn't expanded to string when print char-array-as-string is "off"
        self.runCmd("-var-create var2 * string_arr")
        self.expect("\^done,name=\"var2\",numchild=\"17\",value=\"\[17\]\",type=\"const char \[17\]\",thread-id=\"1\",has_more=\"0\"")

        # Test that -gdb-set can set print char-array-as-string flag
        self.runCmd("-gdb-set print char-array-as-string on")
        self.expect("\^done")
        self.runCmd("-gdb-set print char-array-as-string 1")
        self.expect("\^done")
        self.runCmd("-gdb-show print char-array-as-string")
        self.expect("\^done,value=\"on\"")

        # Test that an char* is expanded to string when print char-array-as-string is "on"
        self.runCmd("-var-create var1 * string_ptr")
        self.expect("\^done,name=\"var1\",numchild=\"1\",value=\"0x[0-9a-f]+ \\\\\\\"string - const char \*\\\\\\\"\",type=\"const char \*\",thread-id=\"1\",has_more=\"0\"")

        # Test that an char[] isn't expanded to string when print char-array-as-string is "on"
        self.runCmd("-var-create var2 * string_arr")
        self.expect("\^done,name=\"var2\",numchild=\"17\",value=\"\\\\\\\"string - char \[\]\\\\\\\"\",type=\"const char \[17\]\",thread-id=\"1\",has_more=\"0\"")

        # Test that -gdb-set print char-array-as-string fails if "on"/"off" isn't specified
        self.runCmd("-gdb-set print char-array-as-string")
        self.expect("\^error,msg=\"The request ''print' expects option-name and \"on\" or \"off\"' failed.\"")

        # Test that -gdb-set print char-array-as-string fails when option is unknown
        self.runCmd("-gdb-set print char-array-as-string unknown")
        self.expect("\^error,msg=\"The request ''print' expects option-name and \"on\" or \"off\"' failed.\"")

if __name__ == '__main__':
    unittest2.main()
