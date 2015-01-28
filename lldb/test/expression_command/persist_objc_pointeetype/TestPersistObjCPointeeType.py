"""
Test that we can p *objcObject
"""

import unittest2
import lldb
import lldbutil
from lldbtest import *

class PersistObjCPointeeType(TestBase):

    mydir = TestBase.compute_mydir(__file__)

    def setUp(self):
        # Call super's setUp().
        TestBase.setUp(self)
        # Find the line number to break for main.cpp.
        self.line = line_number('main.m','// break here')

    @unittest2.skipUnless(sys.platform.startswith("darwin"), "requires Darwin")
    @dsym_test
    def test_with_dsym(self):
        """Test that we can p *objcObject"""
        self.buildDsym()
        self.do_my_test()

    @unittest2.skipUnless(sys.platform.startswith("darwin"), "requires Darwin due to ObjC test case")
    @dwarf_test
    def test_with_dwarf(self):
        """Test that we can p *objcObject"""
        self.buildDwarf()
        self.do_my_test()

    def do_my_test(self):
        def cleanup():
            pass
        
        # Execute the cleanup function during test case tear down.
        self.addTearDownHook(cleanup)

        self.runCmd("file a.out", CURRENT_EXECUTABLE_SET)

        lldbutil.run_break_set_by_file_and_line (self, "main.m", self.line, loc_exact=True)

        self.runCmd("run", RUN_SUCCEEDED)
        
        self.expect("p *self", substrs=['_sc_name = nil',
        '_sc_name2 = nil',
        '_sc_name3 = nil',
        '_sc_name4 = nil',
        '_sc_name5 = nil',
        '_sc_name6 = nil',
        '_sc_name7 = nil',
        '_sc_name8 = nil'])

if __name__ == '__main__':
    import atexit
    lldb.SBDebugger.Initialize()
    atexit.register(lambda: lldb.SBDebugger.Terminate())
    unittest2.main()
