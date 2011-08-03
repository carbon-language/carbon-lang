"""Test printing ObjC objects that use unbacked properties - so that the static ivar offsets are incorrect."""

import os, time
import unittest2
import lldb
from lldbtest import *
import lldbutil

class TestObjCIvarOffsets(TestBase):

    mydir = os.path.join("lang", "objc", "objc-ivar-offsets")

    @unittest2.skipUnless(sys.platform.startswith("darwin"), "requires Darwin")
    @python_api_test
    def test_with_dsym_and_python_api(self):
        """Test printing ObjC objects that use unbacked properties"""
        self.buildDsym()
        self.objc_ivar_offsets()

    @python_api_test
    def test_with_dwarf_and_python_api(self):
        """Test printing ObjC objects that use unbacked properties"""
        self.buildDwarf()
        self.objc_ivar_offsets()

    def setUp(self):
        # Call super's setUp().
        TestBase.setUp(self)
        # Find the line numbers to break inside main().
        self.main_source = "main.m"
        self.stop_line = line_number(self.main_source, '// Set breakpoint here.')

    def objc_ivar_offsets(self):
        """Use Python APIs to test stepping into ObjC methods."""
        exe = os.path.join(os.getcwd(), "a.out")

        target = self.dbg.CreateTarget(exe)
        self.assertTrue(target, VALID_TARGET)

        breakpoint = target.BreakpointCreateByLocation(self.main_source, self.stop_line)
        self.assertTrue(breakpoint, VALID_BREAKPOINT)

        process = target.LaunchSimple (None, None, os.getcwd())
        self.assertTrue (process, "Created a process.")
        self.assertTrue (process.GetState() == lldb.eStateStopped, "Stopped it too.")

        thread_list = lldbutil.get_threads_stopped_at_breakpoint (process, breakpoint)
        self.assertTrue (len(thread_list) == 1)
        thread = thread_list[0]
        
        frame = thread.GetFrameAtIndex(0)
        self.assertTrue (frame, "frame 0 is valid")
        
        mine = thread.GetFrameAtIndex(0).FindVariable("mine")
        self.assertTrue(mine, "Found local variable mine.")
        
        # Test the value object value for BaseClass->_backed_int

        mine_backed_int = mine.GetChildMemberWithName ("_backed_int")
        self.assertTrue(mine_backed_int, "Found mine->backed_int local variable.")
        backed_value = int (mine_backed_int.GetValue (), 0)
        self.assertTrue (backed_value == 1111)
        
        # Test the value object value for DerivedClass->_derived_backed_int

        mine_derived_backed_int = mine.GetChildMemberWithName ("_derived_backed_int")
        self.assertTrue(mine_derived_backed_int, "Found mine->derived_backed_int local variable.")
        derived_backed_value = int (mine_derived_backed_int.GetValue (), 0)
        self.assertTrue (derived_backed_value == 3333)
                                    
if __name__ == '__main__':
    import atexit
    lldb.SBDebugger.Initialize()
    atexit.register(lambda: lldb.SBDebugger.Terminate())
    unittest2.main()
