"""
Test calling a function, stopping in the call, continue and gather the result on stop.
"""

import unittest2
import lldb
import lldbutil
from lldbtest import *

class ExprCommandCallStopContinueTestCase(TestBase):

    mydir = TestBase.compute_mydir(__file__)

    def setUp(self):
        # Call super's setUp().
        TestBase.setUp(self)
        # Find the line number to break for main.c.
        self.line = line_number('main.cpp',
                                '// Please test these expressions while stopped at this line:')
        self.func_line = line_number ('main.cpp', 
                                '{ 5, "five" }')

    @unittest2.skipUnless(sys.platform.startswith("darwin"), "requires Darwin")
    @dsym_test
    @expectedFailureDarwin("llvm.org/pr20274") # intermittent failure on MacOSX
    def test_with_dsym(self):
        """Test gathering result from interrupted function call."""
        self.buildDsym()
        self.call_function()

    @dwarf_test
    @expectedFailureDarwin("llvm.org/pr20274") # intermittent failure on MacOSX
    @expectedFailureFreeBSD("llvm.org/pr20274") # intermittent failure
    @expectedFailureLinux("llvm.org/pr20274") # intermittent failure on Linux
    def test_with_dwarf(self):
        """Test gathering result from interrupted function call."""
        self.buildDwarf()
        self.call_function()

    def call_function(self):
        """Test gathering result from interrupted function call."""
        self.runCmd("file a.out", CURRENT_EXECUTABLE_SET)

        # Some versions of GCC encode two locations for the 'return' statement in main.cpp
        lldbutil.run_break_set_by_file_and_line (self, "main.cpp", self.line, num_expected_locations=-1, loc_exact=True)

        self.runCmd("run", RUN_SUCCEEDED)

        lldbutil.run_break_set_by_file_and_line (self, "main.cpp", self.func_line, num_expected_locations=-1, loc_exact=True)
        
        self.expect("expr -i false -- returnsFive()", error=True,
            substrs = ['Execution was interrupted, reason: breakpoint'])

        self.runCmd("continue", "Continue completed")
        self.expect ("thread list",
                     substrs = ['stop reason = User Expression thread plan',
                                r'Completed expression: (Five) $0 = (number = 5, name = "five")'])

if __name__ == '__main__':
    import atexit
    lldb.SBDebugger.Initialize()
    atexit.register(lambda: lldb.SBDebugger.Terminate())
    unittest2.main()
