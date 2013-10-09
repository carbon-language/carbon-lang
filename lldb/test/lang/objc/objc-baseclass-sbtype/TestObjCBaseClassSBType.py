"""
Use lldb Python API to test base class resolution for ObjC classes
"""

import os, time
import re
import unittest2
import lldb, lldbutil
from lldbtest import *

class ObjCDynamicValueTestCase(TestBase):

    mydir = os.path.join("lang", "objc", "objc-baseclass-sbtype")

    @unittest2.skipUnless(sys.platform.startswith("darwin"), "requires Darwin")
    @python_api_test
    @dsym_test
    def test_get_baseclass_with_dsym(self):
        """Test fetching ObjC base class info."""
        if self.getArchitecture() == 'i386':
            # rdar://problem/9946499
            self.skipTest("Dynamic types for ObjC V1 runtime not implemented")
        self.buildDsym()
        self.do_get_baseclass_info()

    @unittest2.skipUnless(sys.platform.startswith("darwin"), "requires Darwin")
    @python_api_test
    @dwarf_test
    def test_get_baseclass_with_dwarf(self):
        """Test fetching ObjC dynamic values."""
        if self.getArchitecture() == 'i386':
            # rdar://problem/9946499
            self.skipTest("Dynamic types for ObjC V1 runtime not implemented")
        self.buildDwarf()
        self.do_get_baseclass_info()

    def setUp(self):
        # Call super's setUp().                                                                                                           
        TestBase.setUp(self)

        self.line = line_number('main.m', '// Set breakpoint here.')

    def do_get_baseclass_info(self):
        """Make sure we get dynamic values correctly both for compiled in classes and dynamic ones"""
        exe = os.path.join(os.getcwd(), "a.out")

        # Create a target from the debugger.

        target = self.dbg.CreateTarget (exe)
        self.assertTrue(target, VALID_TARGET)

        # Set up our breakpoints:

        target.BreakpointCreateByLocation('main.m', self.line)
        process = target.LaunchSimple (None, None, os.getcwd())

        self.assertTrue(process.GetState() == lldb.eStateStopped,
                        PROCESS_STOPPED)

        button = self.frame().FindVariable("button")
        button_ptr_type = button.GetType()
        button_pte_type = button_ptr_type.GetPointeeType()
        self.assertTrue(button_ptr_type.GetNumberOfDirectBaseClasses() == 1, "NSButton * has one base class")
        self.assertTrue(button_pte_type.GetNumberOfDirectBaseClasses() == 1, "NSButton has one base class")

        self.assertTrue(button_ptr_type.GetDirectBaseClassAtIndex(0).IsValid(), "NSButton * has a valid base class")
        self.assertTrue(button_pte_type.GetDirectBaseClassAtIndex(0).IsValid(), "NSButton * has a valid base class")

        self.assertTrue(button_ptr_type.GetDirectBaseClassAtIndex(0).GetName() == button_pte_type.GetDirectBaseClassAtIndex(0).GetName(), "NSButton and its pointer type don't agree on their base class")

if __name__ == '__main__':
    import atexit
    lldb.SBDebugger.Initialize()
    atexit.register(lambda: lldb.SBDebugger.Terminate())
    unittest2.main()
