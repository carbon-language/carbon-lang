"""
Check that vector types format properly
"""

import os, time
import unittest2
import lldb
from lldbtest import *
import lldbutil

class VectorTypesFormattingTestCase(TestBase):

    mydir = TestBase.compute_mydir(__file__)

    # rdar://problem/14035604
    @skipUnlessDarwin
    @dsym_test
    def test_with_dsym_and_run_command(self):
        """Check that vector types format properly"""
        self.buildDsym()
        self.propagate_test_commands()

    # rdar://problem/14035604
    @dwarf_test
    @skipIf(compiler='gcc') # gcc don't have ext_vector_type extension
    def test_with_dwarf_and_run_command(self):
        """Check that vector types format properly"""
        self.buildDwarf()
        self.propagate_test_commands()

    def setUp(self):
        # Call super's setUp().
        TestBase.setUp(self)
        # Find the line number to break at.
        self.line = line_number('main.cpp', '// break here')

    def propagate_test_commands(self):
        """Check that vector types format properly"""
        self.runCmd("file a.out", CURRENT_EXECUTABLE_SET)

        lldbutil.run_break_set_by_file_and_line (self, "main.cpp", self.line, num_expected_locations=1, loc_exact=True)

        self.runCmd("run", RUN_SUCCEEDED)

        # The stop reason of the thread should be breakpoint.
        self.expect("thread list", STOPPED_DUE_TO_BREAKPOINT,
            substrs = ['stopped',
                       'stop reason = breakpoint'])

        # This is the function to remove the custom formats in order to have a
        # clean slate for the next test case.
        def cleanup():
            pass

        # Execute the cleanup function during test case tear down.
        self.addTearDownHook(cleanup)

        pass # my code never fails
        
        v = self.frame().FindVariable("v")
        v.SetPreferSyntheticValue(True)
        v.SetFormat(lldb.eFormatVectorOfFloat32)
        
        if self.TraceOn(): print v
        
        self.assertTrue(v.GetNumChildren() == 4, "v as float32[] has 4 children")
        self.assertTrue(v.GetChildAtIndex(0).GetData().float[0] == 1.25, "child 0 == 1.25")
        self.assertTrue(v.GetChildAtIndex(1).GetData().float[0] == 1.25, "child 1 == 1.25")
        self.assertTrue(v.GetChildAtIndex(2).GetData().float[0] == 2.50, "child 2 == 2.50")
        self.assertTrue(v.GetChildAtIndex(3).GetData().float[0] == 2.50, "child 3 == 2.50")
        
        self.expect("expr -f int16_t[] -- v", substrs=['[0] = 0', '[1] = 16288', '[2] = 0', '[3] = 16288', '[4] = 0', '[5] = 16416', '[6] = 0', '[7] = 16416'])
        self.expect("expr -f uint128_t[] -- v", substrs=['[0] = 85236745249553456609335044694184296448'])
        
        oldValue = v.GetChildAtIndex(0).GetValue()
        v.SetFormat(lldb.eFormatHex)
        newValue = v.GetChildAtIndex(0).GetValue()
        self.assertFalse(oldValue == newValue, "values did not change along with format")
        
        v.SetFormat(lldb.eFormatVectorOfFloat32)
        oldValueAgain = v.GetChildAtIndex(0).GetValue()
        self.assertTrue(oldValue == oldValueAgain, "same format but different values")

if __name__ == '__main__':
    import atexit
    lldb.SBDebugger.Initialize()
    atexit.register(lambda: lldb.SBDebugger.Terminate())
    unittest2.main()
