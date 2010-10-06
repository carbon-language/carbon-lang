"""Test breakpoint by file/line number; and list variables with array types."""

import os, time
import unittest2
import lldb
from lldbtest import *

class ArrayTypesTestCase(TestBase):

    mydir = "array_types"

    @unittest2.skipUnless(sys.platform.startswith("darwin"), "requires Darwin")
    def test_with_dsym_and_run_command(self):
        """Test 'frame variable var_name' on some variables with array types."""
        self.buildDsym()
        self.array_types()

    @unittest2.skipUnless(sys.platform.startswith("darwin"), "requires Darwin")
    def test_with_dsym_and_python_api(self):
        """Use Python APIs to inspect variables with array types."""
        self.buildDsym()
        self.array_types_python()

    def test_with_dwarf_and_run_command(self):
        """Test 'frame variable var_name' on some variables with array types."""
        self.buildDwarf()
        self.array_types()

    def test_with_dwarf_and_python_api(self):
        """Use Python APIs to inspect variables with array types."""
        self.buildDwarf()
        self.array_types_python()

    def array_types(self):
        """Test 'frame variable var_name' on some variables with array types."""
        exe = os.path.join(os.getcwd(), "a.out")
        self.runCmd("file " + exe, CURRENT_EXECUTABLE_SET)

        # Break on line 42 inside main().
        self.expect("breakpoint set -f main.c -l 42", BREAKPOINT_CREATED,
            startstr = "Breakpoint created: 1: file ='main.c', line = 42, locations = 1")

        self.runCmd("run", RUN_SUCCEEDED)

        # The stop reason of the thread should be breakpoint.
        self.expect("thread list", STOPPED_DUE_TO_BREAKPOINT,
            substrs = ['state is Stopped',
                       'stop reason = breakpoint'])

        # The breakpoint should have a hit count of 1.
        self.expect("breakpoint list", BREAKPOINT_HIT_ONCE,
            substrs = ['resolved, hit count = 1'])

        # Issue 'variable list' command on several array-type variables.

        self.expect("frame variable strings", VARIABLES_DISPLAYED_CORRECTLY,
            startstr = '(char *[4])',
            substrs = ['(char *) strings[0]',
                       '(char *) strings[1]',
                       '(char *) strings[2]',
                       '(char *) strings[3]',
                       'Hello',
                       'Hola',
                       'Bonjour',
                       'Guten Tag'])

        self.expect("frame variable char_16", VARIABLES_DISPLAYED_CORRECTLY,
            substrs = ['(char) char_16[0]',
                       '(char) char_16[15]'])

        self.expect("frame variable ushort_matrix", VARIABLES_DISPLAYED_CORRECTLY,
            startstr = '(unsigned short [2][3])')

        self.expect("frame variable long_6", VARIABLES_DISPLAYED_CORRECTLY,
            startstr = '(long [6])')

    def array_types_python(self):
        """Use Python APIs to inspect variables with array types."""
        exe = os.path.join(os.getcwd(), "a.out")

        target = self.dbg.CreateTarget(exe)
        self.assertTrue(target.IsValid(), VALID_TARGET)

        breakpoint = target.BreakpointCreateByLocation("main.c", 42)
        self.assertTrue(breakpoint.IsValid(), VALID_BREAKPOINT)

        # Sanity check the print representation of breakpoint.
        bp = repr(breakpoint)
        self.expect(bp, msg="Breakpoint looks good", exe=False,
            substrs = ["file ='main.c'",
                       "line = 42",
                       "locations = 1"])
        self.expect(bp, msg="Breakpoint is not resolved as yet", exe=False, matching=False,
            substrs = ["resolved = 1"])

        self.runCmd("run", RUN_SUCCEEDED, setCookie=False)
        # This does not work, and results in the process stopped at dyld_start?
        #process = target.LaunchProcess([''], [''], os.ctermid(), False)

        self.process = target.GetProcess()
        self.assertTrue(self.process.IsValid(), PROCESS_IS_VALID)

        # Sanity check the print representation of process.
        proc = repr(self.process)
        self.expect(proc, msg="Process looks good", exe=False,
            substrs = ["state: Stopped",
                       "executable: a.out"])

        # The stop reason of the thread should be breakpoint.
        thread = self.process.GetThreadAtIndex(0)
        self.assertTrue(thread.GetStopReason() == StopReasonEnum("Breakpoint"),
                        STOPPED_DUE_TO_BREAKPOINT)

        # Sanity check the print representation of thread.
        thr = repr(thread)
        self.expect(thr, "Thread looks good with stop reason = breakpoint", exe=False,
            substrs = ["thread #%d: tid = 0x%4.4x" % (thread.GetIndexID(), thread.GetThreadID()),
                       "stop reason = breakpoint",
                       "queue = %s" % thread.GetQueueName() if thread.GetQueueName() else ""])

        # The breakpoint should have a hit count of 1.
        self.assertTrue(breakpoint.GetHitCount() == 1, BREAKPOINT_HIT_ONCE)

        # The breakpoint should be resolved by now.
        bp = repr(breakpoint)
        self.expect(bp, "Breakpoint looks good and is resolved", exe=False,
            substrs = ["file ='main.c'",
                       "line = 42",
                       "locations = 1",
                       "resolved = 1"])

        # Sanity check the print representation of frame.
        frame = thread.GetFrameAtIndex(0)
        frm = repr(frame)
        self.expect(frm, "Thread looks good with correct frame function", exe=False,
            substrs = ["frame #%d" % frame.GetFrameID(),
                       "a.out`%s" % frame.GetFunction().GetName()])

        # Lookup the "strings" string array variable and sanity check its print
        # representation.
        variable = frame.LookupVar("strings")
        var = repr(variable)
        self.expect(var, "Variable for 'strings' looks good with correct type and size", exe=False,
            substrs = ["name: '%s'" % variable.GetName(),
                       "type: %s" % variable.GetTypeName(),
                       "size: %d" % variable.GetByteSize()])
        self.DebugSBValue(frame, variable)
        self.assertTrue(variable.GetNumChildren() == 4,
                        "Variable 'strings' should have 4 children")

        child3 = variable.GetChildAtIndex(3)
        self.DebugSBValue(frame, child3)
        self.assertTrue(child3.GetSummary(frame) == '"Guten Tag"',
                        'strings[3] == "Guten Tag"')

        # Lookup the "char_16" char array variable.
        variable = frame.LookupVar("char_16")
        self.DebugSBValue(frame, variable)
        self.assertTrue(variable.GetNumChildren() == 16,
                        "Variable 'char_16' should have 16 children")

        # Lookup the "ushort_matrix" ushort[] array variable.
        # Notice the pattern of int(child0_2.GetValue(frame), 0).  We pass a
        # base of 0 so that the proper radix is determined based on the contents
        # of the string.  Same applies to long().
        variable = frame.LookupVar("ushort_matrix")
        self.DebugSBValue(frame, variable)
        self.assertTrue(variable.GetNumChildren() == 2,
                        "Variable 'ushort_matrix' should have 2 children")
        child0 = variable.GetChildAtIndex(0)
        self.DebugSBValue(frame, child0)
        self.assertTrue(child0.GetNumChildren() == 3,
                        "Variable 'ushort_matrix[0]' should have 3 children")
        child0_2 = child0.GetChildAtIndex(2)
        self.DebugSBValue(frame, child0_2)
        self.assertTrue(int(child0_2.GetValue(frame), 0) == 3,
                        "ushort_matrix[0][2] == 3")

        # Lookup the "long_6" char array variable.
        variable = frame.LookupVar("long_6")
        self.DebugSBValue(frame, variable)
        self.assertTrue(variable.GetNumChildren() == 6,
                        "Variable 'long_6' should have 6 children")
        child5 = variable.GetChildAtIndex(5)
        self.DebugSBValue(frame, child5)
        self.assertTrue(long(child5.GetValue(frame), 0) == 6,
                        "long_6[5] == 6")


if __name__ == '__main__':
    import atexit
    lldb.SBDebugger.Initialize()
    atexit.register(lambda: lldb.SBDebugger.Terminate())
    unittest2.main()
