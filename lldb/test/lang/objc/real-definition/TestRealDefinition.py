"""Test that types defined in shared libraries work correctly."""

import os, time
import unittest2
import lldb
from lldbtest import *

class TestRealDefinition(TestBase):

    mydir = os.path.join("lang", "objc", "real-definition")

    def test_expr_with_dsym(self):
        """Test that we can find the implementation for an objective C type"""
        self.buildDsym()
        self.stop_at_interface()

    def test_expr_with_dwarf(self):
        """Test that we can find the implementation for an objective C type"""
        self.buildDwarf()
        self.stop_at_interface()

    def test_frame_variable_with_dsym(self):
        """Test that we can find the implementation for an objective C type"""
        self.buildDsym()
        self.stop_at_implementation()

    def test_frame_variable_with_dwarf(self):
        """Test that we can find the implementation for an objective C type"""
        self.buildDwarf()
        self.stop_at_implementation()

    def setUp(self):
        # Call super's setUp().
        TestBase.setUp(self)

    def common_setup(self):
        exe = os.path.join(os.getcwd(), "a.out")
        self.runCmd("file " + exe, CURRENT_EXECUTABLE_SET)

        # Break inside the foo function which takes a bar_ptr argument.
        self.expect("breakpoint set -f main.m -l %d" % line_number('main.m', '// Set breakpoint in main'), BREAKPOINT_CREATED, startstr = "Breakpoint created")

    def stop_at_interface(self):
        """Test that we can find the implementation for an objective C type when we stop in the interface"""
        self.common_setup()

        self.expect("breakpoint set -f Foo.m -l %d" % line_number('Foo.m', '// Set breakpoint where Bar is an interface'), BREAKPOINT_CREATED, startstr = "Breakpoint created")

        self.runCmd("run", RUN_SUCCEEDED)

        # The stop reason of the thread should be breakpoint.
        self.expect("thread list", STOPPED_DUE_TO_BREAKPOINT,
            substrs = ['stopped',
                       'stop reason = breakpoint'])

        # Run and stop at Foo
        self.expect("breakpoint list -f", BREAKPOINT_HIT_ONCE,
            substrs = [' resolved, hit count = 1'])

        self.runCmd("continue", RUN_SUCCEEDED)

        # Run at stop at main
        self.expect("breakpoint list -f", BREAKPOINT_HIT_ONCE,
            substrs = [' resolved, hit count = 1'])
            
        # This should display correctly.
        self.expect("frame variable foo->_bar->_hidden_ivar", VARIABLES_DISPLAYED_CORRECTLY,
            substrs = ["(NSString *)", "foo->_bar->_hidden_ivar = 0x"])

    def stop_at_implementation(self):
        """Test that we can find the implementation for an objective C type when we stop in the implementation"""
        self.common_setup()

        self.expect("breakpoint set -f Bar.m -l %d" % line_number('Bar.m', '// Set breakpoint where Bar is an implementation'), BREAKPOINT_CREATED, startstr = "Breakpoint created")

        self.runCmd("run", RUN_SUCCEEDED)

        # The stop reason of the thread should be breakpoint.
        self.expect("thread list", STOPPED_DUE_TO_BREAKPOINT,
            substrs = ['stopped',
                       'stop reason = breakpoint'])

        # Run and stop at Foo
        self.expect("breakpoint list -f", BREAKPOINT_HIT_ONCE,
            substrs = [' resolved, hit count = 1'])

        self.runCmd("continue", RUN_SUCCEEDED)

        # Run at stop at main
        self.expect("breakpoint list -f", BREAKPOINT_HIT_ONCE,
            substrs = [' resolved, hit count = 1'])

        # This should display correctly.
        self.expect("frame variable foo->_bar->_hidden_ivar", VARIABLES_DISPLAYED_CORRECTLY,
            substrs = ["(NSString *)", "foo->_bar->_hidden_ivar = 0x"])

                       
if __name__ == '__main__':
    import atexit
    lldb.SBDebugger.Initialize()
    atexit.register(lambda: lldb.SBDebugger.Terminate())
    unittest2.main()
