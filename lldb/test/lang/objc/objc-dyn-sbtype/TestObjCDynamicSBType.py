"""
Test that we are able to properly report a usable dynamic type for NSImage
"""

import os, time
import unittest2
import lldb
from lldbtest import *
import lldbutil

@unittest2.skipUnless(sys.platform.startswith("darwin"), "requires Darwin")
class ObjCDynamicSBTypeTestCase(TestBase):

    mydir = os.path.join("lang", "objc", "objc-dyn-sbtype")

    @dsym_test
    @skipIfi386
    def test_nsimage_dyn_with_dsym(self):
        """Test that we are able to properly report a usable dynamic type for NSImage."""
        d = {'EXE': self.exe_name}
        self.buildDsym(dictionary=d)
        self.setTearDownCleanup(dictionary=d)
        self.nsimage_dyn(self.exe_name)

    @dwarf_test
    @skipIfi386
    def test_nsimage_dyn_with_dwarf(self):
        """Test that we are able to properly report a usable dynamic type for NSImage."""
        d = {'EXE': self.exe_name}
        self.buildDwarf(dictionary=d)
        self.setTearDownCleanup(dictionary=d)
        self.nsimage_dyn(self.exe_name)

    def setUp(self):
        # Call super's setUp().
        TestBase.setUp(self)
        # We'll use the test method name as the exe_name.
        self.exe_name = self.testMethodName
        # Find the line number to break inside main().
        self.main_source = "main.m"
        self.line = line_number(self.main_source, '// Set breakpoint here.')

    def nsimage_dyn(self, exe_name):
        """Test that we are able to properly report a usable dynamic type for NSImage."""
        exe = os.path.join(os.getcwd(), exe_name)
        self.runCmd("file " + exe, CURRENT_EXECUTABLE_SET)

        lldbutil.run_break_set_by_file_and_line (self, self.main_source, self.line, num_expected_locations=1, loc_exact=True)

        self.runCmd("run", RUN_SUCCEEDED)

        image = self.frame().EvaluateExpression("(id)image",lldb.eDynamicCanRunTarget)
        self.assertTrue(image.GetTypeName() == "NSImage *", "The SBValue is properly type-named")
        image_type = image.GetType()
        self.assertTrue(image_type.GetName() == "NSImage *", "The dynamic SBType is for the correct type")
        image_pointee_type = image_type.GetPointeeType()
        self.assertTrue(image_pointee_type.GetName() == "NSImage", "The dynamic type figures out its pointee type just fine")
        self.assertTrue(image_pointee_type.GetDirectBaseClassAtIndex(0).GetName() == "NSObject", "The dynamic type can go back to its base class")


if __name__ == '__main__':
    import atexit
    lldb.SBDebugger.Initialize()
    atexit.register(lambda: lldb.SBDebugger.Terminate())
    unittest2.main()
