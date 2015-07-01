"""
Test that we can have persistent pointer variables
"""

import unittest2
import lldb
import lldbutil
from lldbtest import *

class PersistentPtrUpdateTestCase(TestBase):

    mydir = TestBase.compute_mydir(__file__)

    def setUp(self):
        # Call super's setUp().
        TestBase.setUp(self)

    @skipUnlessDarwin
    @dsym_test
    def test_with_dsym(self):
        """Test that we can have persistent pointer variables"""
        self.buildDsym()
        self.do_my_test()

    @skipUnlessDarwin
    @dwarf_test
    def test_with_dwarf(self):
        """Test that we can have persistent pointer variables"""
        self.buildDwarf()
        self.do_my_test()

    def do_my_test(self):
        def cleanup():
            pass
        
        # Execute the cleanup function during test case tear down.
        self.addTearDownHook(cleanup)

        self.runCmd("file a.out", CURRENT_EXECUTABLE_SET)
        
        self.runCmd('break set -p here')

        self.runCmd("run", RUN_SUCCEEDED)
        
        self.runCmd("expr void* $foo = nullptr")
        
        self.runCmd("continue")
        
        self.expect("expr $foo", substrs=['$foo','0x0'])

if __name__ == '__main__':
    import atexit
    lldb.SBDebugger.Initialize()
    atexit.register(lambda: lldb.SBDebugger.Terminate())
    unittest2.main()
