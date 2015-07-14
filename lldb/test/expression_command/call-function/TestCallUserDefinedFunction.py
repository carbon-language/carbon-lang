"""
Test calling user defined functions using expression evaluation.

Note:
  LLDBs current first choice of evaluating functions is using the IR interpreter,
  which is only supported on Hexagon. Otherwise JIT is used for the evaluation.

"""

import unittest2
import lldb
import lldbutil
from lldbtest import *

class ExprCommandCallUserDefinedFunction(TestBase):

    mydir = TestBase.compute_mydir(__file__)

    def setUp(self):
        # Call super's setUp().
        TestBase.setUp(self)
        # Find the line number to break for main.c.
        self.line = line_number('main.cpp',
                                '// Please test these expressions while stopped at this line:')
    @skipUnlessDarwin
    @dsym_test
    @expectedFailureDarwin("llvm.org/pr20274") # intermittent failure on MacOSX
    def test_with_dsym(self):
        """Test return values of user defined function calls."""
        self.buildDsym()
        self.call_function()

    @dwarf_test
    @expectedFailureFreeBSD("llvm.org/pr20274") # intermittent failure
    def test_with_dwarf(self):
        """Test return values of user defined function calls."""
        self.buildDwarf()
        self.call_functions()

    def call_functions(self):
        """Test return values of user defined function calls."""

        # Set breakpoint in main and run exe
        self.runCmd("file a.out", CURRENT_EXECUTABLE_SET)
        lldbutil.run_break_set_by_file_and_line (self, "main.cpp", self.line, num_expected_locations=-1, loc_exact=True)

        self.runCmd("run", RUN_SUCCEEDED)

        # Test recursive function call.
        self.expect("expr fib(5)", substrs = ['$0 = 5'])

        # Test function with more than one paramter
        self.expect("expr add(4,8)", substrs = ['$1 = 12'])

        # Test nesting function calls in function paramters
        self.expect("expr add(add(5,2),add(3,4))", substrs = ['$2 = 14'])
        self.expect("expr add(add(5,2),fib(5))", substrs = ['$3 = 12'])

        # Test function with pointer paramter
        self.expect("exp stringCompare((const char*) \"Hello world\")", substrs = ['$4 = true'])
        self.expect("exp stringCompare((const char*) \"Hellworld\")", substrs = ['$5 = false'])


if __name__ == '__main__':
    import atexit
    lldb.SBDebugger.Initialize()
    atexit.register(lambda: lldb.SBDebugger.Terminate())
    unittest2.main()
