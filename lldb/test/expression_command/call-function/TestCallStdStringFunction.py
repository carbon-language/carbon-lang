"""
Test calling std::String member functions.
"""

import unittest2
import lldb
import lldbutil
from lldbtest import *

class ExprCommandCallFunctionTestCase(TestBase):

    mydir = os.path.join("expression_command", "call-function")

    def setUp(self):
        # Call super's setUp().
        TestBase.setUp(self)
        # Find the line number to break for main.c.
        self.line = line_number('main.cpp',
                                '// Please test these expressions while stopped at this line:')

    @unittest2.skipUnless(sys.platform.startswith("darwin"), "requires Darwin")
    def test_with_dsym(self):
        """Test calling std::String member function."""
        self.buildDsym()
        self.call_function()

    @unittest2.skipUnless(sys.platform.startswith("darwin"), "requires Darwin")
    @python_api_test
    def test_with_dwarf_(self):
        """Test calling std::String member function."""
        self.buildDsym()
        self.call_function()

    def call_function(self):
        """Test calling std::String member function."""
        self.runCmd("file a.out", CURRENT_EXECUTABLE_SET)

        self.expect("breakpoint set -f main.cpp -l %d" % self.line,
                    BREAKPOINT_CREATED,
            startstr = "Breakpoint created: 1: file ='main.cpp', line = %d" %
                        self.line)

        self.runCmd("run", RUN_SUCCEEDED)

        self.expect("print str",
            substrs = ['Hello world'])

        # rdar://problem/9471744 test failure: ./dotest.py -C clang -v -w -t -p CallStdString
        # runCmd: print str.c_str()
        # runCmd failed!
        # error: Couldn't convert the expression to DWARF
        self.expect("print str.c_str()",
            substrs = ['Hello world'])

if __name__ == '__main__':
    import atexit
    lldb.SBDebugger.Initialize()
    atexit.register(lambda: lldb.SBDebugger.Terminate())
    unittest2.main()
