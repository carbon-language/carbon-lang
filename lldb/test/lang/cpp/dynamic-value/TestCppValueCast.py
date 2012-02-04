"""
Test lldb Python API SBValue::Cast(SBType) for C++ types.
"""

import os, time
import re
import unittest2
import lldb, lldbutil
from lldbtest import *

class CppValueCastTestCase(TestBase):

    mydir = os.path.join("lang", "cpp", "dynamic-value")

    @unittest2.skipUnless(sys.platform.startswith("darwin"), "requires Darwin")
    @python_api_test
    def test_value_cast_with_dsym(self):
        """Test SBValue::Cast(SBType) API for C++ types."""
        self.buildDsym(dictionary=self.d)
        self.do_sbvalue_cast(self.exe_name)

    @python_api_test
    def test_get_dynamic_vals_with_dwarf(self):
        """Test SBValue::Cast(SBType) API for C++ types."""
        self.buildDwarf(dictionary=self.d)
        self.do_sbvalue_cast(self.exe_name)

    def setUp(self):
        # Call super's setUp().                                                                                                           
        TestBase.setUp(self)

        # Find the line number to break for main.c.                                                                                       
        self.source = 'sbvalue-cast.cpp';
        self.line = line_number(self.source, '// Set breakpoint here.')
        self.exe_name = self.testMethodName
        self.d = {'CXX_SOURCES': self.source, 'EXE': self.exe_name}

    def do_sbvalue_cast (self, exe_name):
        """Test SBValue::Cast(SBType) API for C++ types."""
        exe = os.path.join(os.getcwd(), exe_name)

        # Create a target from the debugger.

        target = self.dbg.CreateTarget (exe)
        self.assertTrue(target, VALID_TARGET)

        # Set up our breakpoints:

        breakpoint = target.BreakpointCreateByLocation(self.source, self.line)
        self.assertTrue(breakpoint, VALID_BREAKPOINT)

        # Now launch the process, and do not stop at the entry point.
        process = target.LaunchSimple (None, None, os.getcwd())

        self.assertTrue(process.GetState() == lldb.eStateStopped,
                        PROCESS_STOPPED)

        # Find DerivedA and DerivedB types.
        typeA = target.FindFirstType('DerivedA')
        typeB = target.FindFirstType('DerivedB')
        self.DebugSBType(typeA)
        self.DebugSBType(typeB)
        self.assertTrue(typeA)
        self.assertTrue(typeB)
        error = lldb.SBError()

        # First stop is for DerivedA instance.
        threads = lldbutil.get_threads_stopped_at_breakpoint (process, breakpoint)
        self.assertTrue (len(threads) == 1)
        thread = threads[0]
        frame0 = thread.GetFrameAtIndex(0)

        tellerA = frame0.FindVariable('teller', lldb.eNoDynamicValues)
        self.DebugSBValue(tellerA)
        self.assertTrue(tellerA.GetChildMemberWithName('m_base_val').GetValueAsUnsigned(error, 0) == 20)

        if self.TraceOn():
            for child in tellerA:
                print "child name:", child.GetName()
                print child

        # Call SBValue.Cast() to obtain instanceA.
        instanceA = tellerA.Cast(typeA.GetPointerType())
        self.DebugSBValue(instanceA)

        # These outputs don't look correct?
        if self.TraceOn():
            for child in instanceA:
                print "child name:", child.GetName()
                print child
        a_member_val = instanceA.GetChildMemberWithName('m_a_val')
        self.DebugSBValue(a_member_val)

        # Second stop is for DerivedB instance.
        threads = lldbutil.continue_to_breakpoint (process, breakpoint)
        self.assertTrue (len(threads) == 1)
        thread = threads[0]
        frame0 = thread.GetFrameAtIndex(0)

        tellerB = frame0.FindVariable('teller', lldb.eNoDynamicValues)
        self.DebugSBValue(tellerB)
        self.assertTrue(tellerB.GetChildMemberWithName('m_base_val').GetValueAsUnsigned(error, 0) == 12)

        if self.TraceOn():
            for child in tellerB:
                print "child name:", child.GetName()
                print child

        # Call SBValue.Cast() to obtain instanceB.
        instanceB = tellerB.Cast(typeB.GetPointerType())
        self.DebugSBValue(instanceB)

        # These outputs don't look correct?
        if self.TraceOn():
            for child in instanceB:
                print "child name:", child.GetName()
                print child
        b_member_val = instanceB.GetChildMemberWithName('m_b_val')
        self.DebugSBValue(b_member_val)


if __name__ == '__main__':
    import atexit
    lldb.SBDebugger.Initialize()
    atexit.register(lambda: lldb.SBDebugger.Terminate())
    unittest2.main()
