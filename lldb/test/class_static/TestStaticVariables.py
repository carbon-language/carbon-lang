"""
Test display and Python APIs on file and class static variables.
"""

import os, time
import unittest2
import lldb
from lldbtest import *

class StaticVariableTestCase(TestBase):

    mydir = "class_static"

    @unittest2.skipUnless(sys.platform.startswith("darwin"), "requires Darwin")
    def test_with_dsym_and_run_command(self):
        """Test that file and class static variables display correctly."""
        self.buildDsym()
        self.static_variable_commands()

    def test_with_dwarf_and_run_command(self):
        """Test that file and class static variables display correctly."""
        self.buildDwarf()
        self.static_variable_commands()

    @unittest2.skipUnless(sys.platform.startswith("darwin"), "requires Darwin")
    def test_with_dsym_and_python_api(self):
        """Test Python APIs on file and class static variables."""
        self.buildDsym()
        self.static_variable_python()

    def test_with_dwarf_and_python_api(self):
        """Test Python APIs on file and class static variables."""
        self.buildDwarf()
        self.static_variable_python()

    def setUp(self):
        # Call super's setUp().
        TestBase.setUp(self)
        # Find the line number to break at.
        self.line = line_number('main.cpp', '// Set break point at this line.')

    def static_variable_commands(self):
        """Test that that file and class static variables display correctly."""
        self.runCmd("file a.out", CURRENT_EXECUTABLE_SET)

        self.expect("breakpoint set -f main.cpp -l %d" % self.line,
                    BREAKPOINT_CREATED,
            startstr = "Breakpoint created: 1: file ='main.cpp', line = %d, locations = 1" %
                        self.line)

        self.runCmd("run", RUN_SUCCEEDED)

        # The stop reason of the thread should be breakpoint.
        self.expect("thread list", STOPPED_DUE_TO_BREAKPOINT,
            substrs = ['state is stopped',
                       'stop reason = breakpoint'])

        # On Mac OS X, gcc 4.2 emits the wrong debug info for A::g_points.
        slist = ['(PointType [2]) g_points', 'A::g_points']

        # 'frame variable -G' finds and displays global variable(s) by name.
        self.expect('frame variable -G g_points', VARIABLES_DISPLAYED_CORRECTLY,
            substrs = slist)

        # A::g_points is an array of two elements.
        if sys.platform.startswith("darwin") and self.getCompiler() in ['clang', 'llvm-gcc']:
            self.expect("frame variable A::g_points[1].x", VARIABLES_DISPLAYED_CORRECTLY,
                startstr = "(int) A::g_points[1].x = 11")

    def static_variable_python(self):
        """Test Python APIs on file and class static variables."""
        exe = os.path.join(os.getcwd(), "a.out")

        target = self.dbg.CreateTarget(exe)
        self.assertTrue(target.IsValid(), VALID_TARGET)

        breakpoint = target.BreakpointCreateByLocation("main.cpp", self.line)
        self.assertTrue(breakpoint.IsValid(), VALID_BREAKPOINT)

        # Now launch the process, and do not stop at entry point.
        self.process = target.LaunchProcess([], [], os.ctermid(), 0, False)

        self.process = target.GetProcess()
        self.assertTrue(self.process.IsValid(), PROCESS_IS_VALID)

        # The stop reason of the thread should be breakpoint.
        thread = self.process.GetThreadAtIndex(0)
        if thread.GetStopReason() != lldb.eStopReasonBreakpoint:
            from lldbutil import StopReasonString
            self.fail(STOPPED_DUE_TO_BREAKPOINT_WITH_STOP_REASON_AS %
                      StopReasonString(thread.GetStopReason()))

        # Get the SBValue of 'A::g_points' and 'g_points'.
        frame = thread.GetFrameAtIndex(0)

        # arguments =>     False
        # locals =>        False
        # statics =>       True
        # in_scope_only => False
        valList = frame.GetVariables(False, False, True, False)

        from lldbutil import lldb_iter
        for val in lldb_iter(valList, 'GetSize', 'GetValueAtIndex'):
            self.DebugSBValue(frame, val)
            self.assertTrue(val.GetValueType() == lldb.eValueTypeVariableGlobal)
            name = val.GetName()
            self.assertTrue(name in ['g_points', 'A::g_points'])
            if name == 'g_points':
                self.assertTrue(val.GetNumChildren() == 2)
            elif name == 'A::g_points' and self.getCompiler() in ['clang', 'llvm-gcc']:
                # On Mac OS X, gcc 4.2 emits the wrong debug info for A::g_points.        
                self.assertTrue(val.GetNumChildren() == 2)
                child1 = val.GetChildAtIndex(1)
                self.DebugSBValue(frame, child1)
                child1_x = child1.GetChildAtIndex(0)
                self.DebugSBValue(frame, child1_x)
                self.assertTrue(child1_x.GetTypeName() == 'int' and
                                child1_x.GetValue(frame) == '11')

        # SBFrame.LookupVarInScope() should also work.
        val = frame.LookupVarInScope("A::g_points", "global")
        self.DebugSBValue(frame, val)
        self.assertTrue(val.GetName() == 'A::g_points')

        # Also exercise the "parameter" and "local" scopes while we are at it.
        val = frame.LookupVarInScope("argc", "parameter")
        self.DebugSBValue(frame, val)
        self.assertTrue(val.GetName() == 'argc')

        val = frame.LookupVarInScope("argv", "parameter")
        self.DebugSBValue(frame, val)
        self.assertTrue(val.GetName() == 'argv')

        val = frame.LookupVarInScope("hello_world", "local")
        self.DebugSBValue(frame, val)
        self.assertTrue(val.GetName() == 'hello_world')


if __name__ == '__main__':
    import atexit
    lldb.SBDebugger.Initialize()
    atexit.register(lambda: lldb.SBDebugger.Terminate())
    unittest2.main()
