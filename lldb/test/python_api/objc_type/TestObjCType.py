"""
Test SBType for ObjC classes.
"""

import os, time
import re
import unittest2
import lldb, lldbutil
from lldbtest import *

class ObjCSBTypeTestCase(TestBase):

    mydir = TestBase.compute_mydir(__file__)

    @unittest2.skipUnless(sys.platform.startswith("darwin"), "requires Darwin")
    @python_api_test
    @dsym_test
    def test_with_dsym(self):
        """Test SBType for ObjC classes."""
        self.buildDsym()
        self.objc_sbtype_test()

    @unittest2.skipUnless(sys.platform.startswith("darwin"), "requires Darwin")
    @python_api_test
    @dwarf_test
    def test_with_dwarf(self):
        """Test SBType for ObjC classes."""
        self.buildDwarf()
        self.objc_sbtype_test()

    def setUp(self):
        # Call super's setUp().
        TestBase.setUp(self)
        self.line = line_number("main.m", '// Break at this line')

    def objc_sbtype_test(self):
        """Exercise SBType and SBTypeList API."""
        exe = os.path.join(os.getcwd(), "a.out")

        # Create a target by the debugger.
        target = self.dbg.CreateTarget(exe)
        self.assertTrue(target, VALID_TARGET)

        # Create the breakpoint inside function 'main'.
        breakpoint = target.BreakpointCreateByLocation("main.m", self.line)
        self.assertTrue(breakpoint, VALID_BREAKPOINT)

        # Now launch the process, and do not stop at entry point.
        process = target.LaunchSimple (None, None, self.get_process_working_directory())
        self.assertTrue(process, PROCESS_IS_VALID)



        # Get Frame #0.
        self.assertTrue(process.GetState() == lldb.eStateStopped)
        thread = lldbutil.get_stopped_thread(process, lldb.eStopReasonBreakpoint)
        self.assertTrue(thread.IsValid(), "There should be a thread stopped due to breakpoint condition")

        aBar = self.frame().FindVariable("aBar")
        aBarType = aBar.GetType()
        self.assertTrue(aBarType.IsValid(), "Bar should be a valid data type")
        self.assertTrue(aBarType.GetName() == "Bar *", "Bar has the right name")

        self.assertTrue(aBarType.GetNumberOfDirectBaseClasses() == 1, "Bar has a superclass")
        aFooType = aBarType.GetDirectBaseClassAtIndex(0)

        self.assertTrue(aFooType.IsValid(), "Foo should be a valid data type")
        self.assertTrue(aFooType.GetName() == "Foo", "Foo has the right name")

        self.assertTrue(aBarType.GetNumberOfFields() == 1, "Bar has a field")
        aBarField = aBarType.GetFieldAtIndex(0)

        self.assertTrue(aBarField.GetName() == "_iVar", "The field has the right name")

if __name__ == '__main__':
    import atexit
    lldb.SBDebugger.Initialize()
    atexit.register(lambda: lldb.SBDebugger.Terminate())
    unittest2.main()
