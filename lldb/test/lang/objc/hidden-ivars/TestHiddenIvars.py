"""Test that hidden ivars in a shared library are visible from the main executable."""

import os, time
import unittest2
import lldb
from lldbtest import *
import lldbutil

class HiddenIvarsTestCase(TestBase):

    mydir = TestBase.compute_mydir(__file__)

    @unittest2.skipUnless(sys.platform.startswith("darwin"), "requires Darwin")
    @dsym_test
    def test_expr_with_dsym(self):
        if self.getArchitecture() == 'i386':
            self.skipTest("requires modern objc runtime")
        self.buildDsym()
        self.expr()

    @unittest2.skipUnless(sys.platform.startswith("darwin"), "requires Darwin")
    @dwarf_test
    def test_expr_with_dwarf(self):
        if self.getArchitecture() == 'i386':
            self.skipTest("requires modern objc runtime")
        self.buildDwarf()
        self.expr()

    @unittest2.expectedFailure("rdar://18683637")
    @unittest2.skipUnless(sys.platform.startswith("darwin"), "requires Darwin")
    @dsym_test
    def test_frame_variable_with_dsym(self):
        if self.getArchitecture() == 'i386':
            self.skipTest("requires modern objc runtime")
        self.buildDsym()
        self.frame_var()

    @unittest2.expectedFailure("rdar://18683637")
    @unittest2.skipUnless(sys.platform.startswith("darwin"), "requires Darwin")
    @dwarf_test
    def test_frame_variable_with_dwarf(self):
        if self.getArchitecture() == 'i386':
            self.skipTest("requires modern objc runtime")
        self.buildDwarf()
        self.frame_var()

    def setUp(self):
        # Call super's setUp().
        TestBase.setUp(self)
        # Find the line number to break inside main().
        self.source = 'main.m'
        self.line = line_number(self.source, '// breakpoint1')
        # The makefile names of the shared libraries as they appear in DYLIB_NAME.
        # The names should have no loading "lib" or extension as they will be localized
        self.shlib_names = ["InternalDefiner"]
        
    def common_setup(self):
        
        # Create a target by the debugger.
        target = self.dbg.CreateTarget("a.out")
        self.assertTrue(target, VALID_TARGET)

        # Create the breakpoint inside function 'main'.
        breakpoint = target.BreakpointCreateByLocation(self.source, self.line)
        self.assertTrue(breakpoint, VALID_BREAKPOINT)
        
        # Register our shared libraries for remote targets so they get automatically uploaded
        environment = self.registerSharedLibrariesWithTarget(target, self.shlib_names)

        # Now launch the process, and do not stop at entry point.
        process = target.LaunchSimple (None, environment, self.get_process_working_directory())
        self.assertTrue(process, PROCESS_IS_VALID)
        
        exe = os.path.join(os.getcwd(), "a.out")
        self.runCmd("file " + exe, CURRENT_EXECUTABLE_SET)

        # Break inside the foo function which takes a bar_ptr argument.
        lldbutil.run_break_set_by_file_and_line (self, "main.m", self.line, num_expected_locations=1, loc_exact=True)

        self.runCmd("run", RUN_SUCCEEDED)

        # The stop reason of the thread should be breakpoint.
        self.expect("thread list", STOPPED_DUE_TO_BREAKPOINT,
            substrs = ['stopped',
                       'stop reason = breakpoint'])

        # The breakpoint should have a hit count of 1.
        self.expect("breakpoint list -f", BREAKPOINT_HIT_ONCE,
            substrs = [' resolved, hit count = 1'])

    def expr(self):
        self.common_setup()

        # This should display correctly.
        self.expect("expression (j->_definer->foo)", VARIABLES_DISPLAYED_CORRECTLY,
            substrs = ["= 4"])

        self.expect("expression (j->_definer->bar)", VARIABLES_DISPLAYED_CORRECTLY,
            substrs = ["= 5"])
            
        self.expect("expression *(j->_definer)", VARIABLES_DISPLAYED_CORRECTLY,
            substrs = ["foo = 4", "bar = 5"])

        self.expect("expression (k->foo)", VARIABLES_DISPLAYED_CORRECTLY,
            substrs = ["= 2"])

        self.expect("expression (k->bar)", VARIABLES_DISPLAYED_CORRECTLY,
            substrs = ["= 3"])

        self.expect("expression *(k)", VARIABLES_DISPLAYED_CORRECTLY,
            substrs = ["foo = 2", "bar = 3"])

    def frame_var(self):
        self.common_setup()

        # This should display correctly.
        self.expect("frame variable j->_definer->foo", VARIABLES_DISPLAYED_CORRECTLY,
            substrs = ["= 4"])

        self.expect("frame variable j->_definer->bar", VARIABLES_DISPLAYED_CORRECTLY,
            substrs = ["= 5"])
            
        self.expect("frame variable *j->_definer", VARIABLES_DISPLAYED_CORRECTLY,
            substrs = ["foo = 4", "bar = 5"])

        self.expect("frame variable k->foo", VARIABLES_DISPLAYED_CORRECTLY,
            substrs = ["= 2"])

        self.expect("frame variable k->bar", VARIABLES_DISPLAYED_CORRECTLY,
            substrs = ["= 3"])

        self.expect("frame variable *k", VARIABLES_DISPLAYED_CORRECTLY,
            substrs = ["foo = 2", "bar = 3"])
                       
if __name__ == '__main__':
    import atexit
    lldb.SBDebugger.Initialize()
    atexit.register(lambda: lldb.SBDebugger.Terminate())
    unittest2.main()
