"""
Test SBType APIs to fetch member function types.
"""

import os, time
import re
import unittest2
import lldb, lldbutil
from lldbtest import *

class SBTypeMemberFunctionsTest(TestBase):

    mydir = TestBase.compute_mydir(__file__)

    @unittest2.skipUnless(sys.platform.startswith("darwin"), "requires Darwin")
    @python_api_test
    @dsym_test
    def test_with_dsym(self):
        """Test SBType APIs to fetch member function types."""
        d = {'EXE': self.exe_name}
        self.buildDsym(dictionary=d)
        self.setTearDownCleanup(dictionary=d)
        self.type_api(self.exe_name)

    @python_api_test
    @dwarf_test
    def test_with_dwarf(self):
        """Test SBType APIs to fetch member function types."""
        d = {'EXE': self.exe_name}
        self.buildDwarf(dictionary=d)
        self.setTearDownCleanup(dictionary=d)
        self.type_api(self.exe_name)

    def setUp(self):
        # Call super's setUp().
        TestBase.setUp(self)
        # We'll use the test method name as the exe_name.
        self.exe_name = self.testMethodName
        # Find the line number to break at.
        self.source = 'main.cpp'
        self.line = line_number(self.source, '// set breakpoint here')

    def type_api(self, exe_name):
        """Test SBType APIs to fetch member function types."""
        exe = os.path.join(os.getcwd(), exe_name)

        # Create a target by the debugger.
        target = self.dbg.CreateTarget(exe)
        self.assertTrue(target, VALID_TARGET)

        # Create the breakpoint inside function 'main'.
        breakpoint = target.BreakpointCreateByLocation(self.source, self.line)
        self.assertTrue(breakpoint, VALID_BREAKPOINT)

        # Now launch the process, and do not stop at entry point.
        process = target.LaunchSimple (None, None, self.get_process_working_directory())
        self.assertTrue(process, PROCESS_IS_VALID)

        # Get Frame #0.
        self.assertTrue(process.GetState() == lldb.eStateStopped)
        thread = lldbutil.get_stopped_thread(process, lldb.eStopReasonBreakpoint)
        self.assertTrue(thread.IsValid(), "There should be a thread stopped due to breakpoint condition")
        frame0 = thread.GetFrameAtIndex(0)
        
        variable = frame0.FindVariable("d")
        Derived = variable.GetType()
        Base = Derived.GetDirectBaseClassAtIndex(0).GetType()

        self.assertTrue(Derived.GetNumberOfMemberFunctions() == 2, "Derived declares two methods")
        self.assertTrue(Derived.GetMemberFunctionAtIndex(0).GetType().GetFunctionReturnType().GetName() == "int", "Derived::dImpl returns int")
        
        self.assertTrue(Base.GetNumberOfMemberFunctions() == 4, "Base declares three methods")
        self.assertTrue(Base.GetMemberFunctionAtIndex(3).GetType().GetFunctionArgumentTypes().GetSize() == 3, "Base::sfunc takes three arguments")
        self.assertTrue(Base.GetMemberFunctionAtIndex(3).GetName() == "sfunc", "Base::sfunc not found")
        self.assertTrue(Base.GetMemberFunctionAtIndex(3).GetKind() == lldb.eMemberFunctionKindStaticMethod, "Base::sfunc is a static")
        self.assertTrue(Base.GetMemberFunctionAtIndex(2).GetType().GetFunctionArgumentTypes().GetSize() == 0, "Base::dat takes no arguments")
        self.assertTrue(Base.GetMemberFunctionAtIndex(1).GetType().GetFunctionArgumentTypes().GetTypeAtIndex(1).GetName() == "char", "Base::bar takes a second 'char' argument")
        self.assertTrue(Base.GetMemberFunctionAtIndex(1).GetName() == "bar", "Base::bar not found")

if __name__ == '__main__':
    import atexit
    lldb.SBDebugger.Initialize()
    atexit.register(lambda: lldb.SBDebugger.Terminate())
    unittest2.main()
