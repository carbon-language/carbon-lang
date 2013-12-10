"""
Test that the po command acts correctly.
"""

import unittest2
import lldb
import lldbutil
from lldbtest import *

class PoVerbosityTestCase(TestBase):

    mydir = TestBase.compute_mydir(__file__)

    def setUp(self):
        # Call super's setUp().
        TestBase.setUp(self)
        # Find the line number to break for main.cpp.
        self.line = line_number('main.m',
                                '// Stop here')

    @unittest2.skipUnless(sys.platform.startswith("darwin"), "requires Darwin")
    @dsym_test
    def test_with_dsym(self):
        """Test that the po command acts correctly."""
        self.buildDsym()
        self.do_my_test()

    @unittest2.skipUnless(sys.platform.startswith("darwin"), "requires Darwin due to ObjC test case")
    @dwarf_test
    def test_with_dwarf(self):
        """Test that the po command acts correctly."""
        self.buildDwarf()
        self.do_my_test()

    def do_my_test(self):
        
        # This is the function to remove the custom formats in order to have a
        # clean slate for the next test case.
        def cleanup():
            self.runCmd('type summary clear', check=False)
            self.runCmd('type synthetic clear', check=False)
        
        # Execute the cleanup function during test case tear down.
        self.addTearDownHook(cleanup)

        """Test expr + formatters for good interoperability."""
        self.runCmd("file a.out", CURRENT_EXECUTABLE_SET)

        lldbutil.run_break_set_by_file_and_line (self, "main.m", self.line, loc_exact=True)

        self.runCmd("run", RUN_SUCCEEDED)
        
        self.expect("expr -O -v -- foo",
            substrs = ['(id) $',' = 0x', '1 = 2','2 = 3;'])
        self.expect("expr -O -vfull -- foo",
            substrs = ['(id) $',' = 0x', '1 = 2','2 = 3;'])
        self.expect("expr -O -- foo",matching=False,
            substrs = ['(id) $'])

        self.expect("expr -O -- 5",matching=False,
            substrs = ['(int) $'])
        self.expect("expr -O -- 5",
            substrs = ['5'])

        self.expect("expr -O -vfull -- 5",
            substrs = ['(int) $', ' = 5'])

        self.expect("expr -O -v -- 5",
            substrs = ['(int) $', ' = 5'])


if __name__ == '__main__':
    import atexit
    lldb.SBDebugger.Initialize()
    atexit.register(lambda: lldb.SBDebugger.Terminate())
    unittest2.main()
