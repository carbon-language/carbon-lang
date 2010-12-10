"""
Test many basic expression commands and SBFrame.EvaluateExpression() API.

Test cases:

o test_many_expr_commands:
  Test many basic expression commands.
o test_evaluate_expression_python:
  Use Python APIs (SBFrame.EvaluateExpression()) to evaluate expressions.
o test_expr_commands_can_handle_quotes:
  Throw some expression commands with quotes at lldb.
"""

import os, time
import unittest2
import lldb
import lldbutil
from lldbtest import *

class BasicExprCommandsTestCase(TestBase):

    mydir = os.path.join("expression_command", "test")

    def setUp(self):
        # Call super's setUp().
        TestBase.setUp(self)
        # Find the line number to break for main.c.
        self.line = line_number('main.cpp',
                                '// Please test many expressions while stopped at this line:')

    def test_many_expr_commands(self):
        """These basic expression commands should work as expected."""
        self.buildDefault()

        self.runCmd("file a.out", CURRENT_EXECUTABLE_SET)

        self.expect("breakpoint set -f main.cpp -l %d" % self.line,
                    BREAKPOINT_CREATED,
            startstr = "Breakpoint created: 1: file ='main.cpp', line = %d" %
                        self.line)

        self.runCmd("run", RUN_SUCCEEDED)

        self.expect("expression 2",
            patterns = ["\(int\) \$.* = 2"])
        # (int) $0 = 1

        self.expect("expression 2ull",
            patterns = ["\(unsigned long long\) \$.* = 2"])
        # (unsigned long long) $1 = 2

        self.expect("expression 2.234f",
            patterns = ["\(float\) \$.* = 2\.234"])
        # (float) $2 = 2.234

        self.expect("expression 2.234",
            patterns = ["\(double\) \$.* = 2\.234"])
        # (double) $3 = 2.234

        self.expect("expression 2+3",
            patterns = ["\(int\) \$.* = 5"])
        # (int) $4 = 5

        self.expect("expression argc",
            patterns = ["\(int\) \$.* = 1"])
        # (int) $5 = 1

        self.expect("expression argc + 22",
            patterns = ["\(int\) \$.* = 23"])
        # (int) $6 = 23

        self.expect("expression argv",
            patterns = ["\(const char \*\*\) \$.* = 0x"])
        # (const char *) $7 = ...

        self.expect("expression argv[0]",
            substrs = ["(const char *)",
                       os.path.join(self.mydir, "a.out")])
        # (const char *) $8 = 0x... "/Volumes/data/lldb/svn/trunk/test/expression_command/test/a.out"

    @python_api_test
    def test_evaluate_expression_python(self):
        """Test SBFrame.EvaluateExpression() API for evaluating an expression."""
        self.buildDefault()

        exe = os.path.join(os.getcwd(), "a.out")

        target = self.dbg.CreateTarget(exe)
        self.assertTrue(target.IsValid(), VALID_TARGET)

        # Create the breakpoint.
        filespec = lldb.SBFileSpec("main.cpp", False)
        breakpoint = target.BreakpointCreateByLocation(filespec, self.line)
        self.assertTrue(breakpoint.IsValid(), VALID_BREAKPOINT)

        # Verify the breakpoint just created.
        self.expect(repr(breakpoint), BREAKPOINT_CREATED, exe=False,
            substrs = ['main.cpp',
                       str(self.line)])

        # Launch the process, and do not stop at the entry point.
        # Pass 'X Y Z' as the args, which makes argc == 4.
        rc = lldb.SBError()
        self.process = target.Launch(['X', 'Y', 'Z'], [], os.ctermid(), 0, False, rc)

        if not rc.Success() or not self.process.IsValid():
            self.fail("SBTarget.LaunchProcess() failed")

        if self.process.GetState() != lldb.eStateStopped:
            self.fail("Process should be in the 'stopped' state, "
                      "instead the actual state is: '%s'" %
                      lldbutil.StateTypeString(self.process.GetState()))

        # The stop reason of the thread should be breakpoint.
        thread = self.process.GetThreadAtIndex(0)
        if thread.GetStopReason() != lldb.eStopReasonBreakpoint:
            from lldbutil import StopReasonString
            self.fail(STOPPED_DUE_TO_BREAKPOINT_WITH_STOP_REASON_AS %
                      StopReasonString(thread.GetStopReason()))

        # The filename of frame #0 should be 'main.cpp' and function is main.
        self.expect(lldbutil.GetFilenames(thread)[0],
                    "Break correctly at main.cpp", exe=False,
            startstr = "main.cpp")
        self.expect(lldbutil.GetFunctionNames(thread)[0],
                    "Break correctly at main()", exe=False,
            startstr = "main")

        # We should be stopped on the breakpoint with a hit count of 1.
        self.assertTrue(breakpoint.GetHitCount() == 1, BREAKPOINT_HIT_ONCE)

        #
        # Use Python API to evaluate expressions while stopped in a stack frame.
        #
        frame = thread.GetFrameAtIndex(0)

        val = frame.EvaluateExpression("2.234")
        self.expect(val.GetValue(frame), "2.345 evaluated correctly", exe=False,
            startstr = "2.234")
        self.expect(val.GetTypeName(), "2.345 evaluated correctly", exe=False,
            startstr = "double")
        self.DebugSBValue(frame, val)

        val = frame.EvaluateExpression("argc")
        self.expect(val.GetValue(frame), "Argc evaluated correctly", exe=False,
            startstr = "4")
        self.DebugSBValue(frame, val)

        val = frame.EvaluateExpression("*argv[1]")
        self.expect(val.GetValue(frame), "Argv[1] evaluated correctly", exe=False,
            startstr = "'X'")
        self.DebugSBValue(frame, val)

        val = frame.EvaluateExpression("*argv[2]")
        self.expect(val.GetValue(frame), "Argv[2] evaluated correctly", exe=False,
            startstr = "'Y'")
        self.DebugSBValue(frame, val)

        val = frame.EvaluateExpression("*argv[3]")
        self.expect(val.GetValue(frame), "Argv[3] evaluated correctly", exe=False,
            startstr = "'Z'")
        self.DebugSBValue(frame, val)

    # rdar://problem/8686536
    # CommandInterpreter::HandleCommand is stripping \'s from input for WantsRawCommand commands
    def test_expr_commands_can_handle_quotes(self):
        """Throw some expression commands with quotes at lldb."""
        self.buildDefault()

        self.runCmd("file a.out", CURRENT_EXECUTABLE_SET)

        self.expect("breakpoint set -f main.cpp -l %d" % self.line,
                    BREAKPOINT_CREATED,
            startstr = "Breakpoint created: 1: file ='main.cpp', line = %d" %
                        self.line)

        self.runCmd("run", RUN_SUCCEEDED)

        # runCmd: expression 'a'
        # output: (char) $0 = 'a'
        self.expect("expression 'a'",
            substrs = ['(char) $',
                       "'a'"])

        # runCmd: expression printf ("\n\n\tHello there!\n")
        # output: (unsigned long) $1 = 16
        self.expect(r'''expression printf ("\n\n\tHello there!\n")''',
            substrs = ['(unsigned long) $',
                       '16'])

        # runCmd: expression printf("\t\x68\n")
        # output: (unsigned long) $2 = 3
        self.expect(r'''expression printf("\t\x68\n")''',
            substrs = ['(unsigned long) $',
                       '3'])

        # runCmd: expression printf("\"\n")
        # output: (unsigned long) $3 = 2
        self.expect(r'''expression printf("\"\n")''',
            substrs = ['(unsigned long) $',
                       '2'])

        # runCmd: expression printf("'\n")
        # output: (unsigned long) $4 = 2
        self.expect(r'''expression printf("'\n")''',
            substrs = ['(unsigned long) $',
                       '2'])

        # runCmd: command alias print_hi expression printf ("\n\tHi!\n")
        # output: 
        self.runCmd(r'''command alias print_hi expression printf ("\n\tHi!\n")''')
        # This fails currently.
        self.expect('print_hi',
            substrs = ['(unsigned long) $',
                       '6'])


if __name__ == '__main__':
    import atexit
    lldb.SBDebugger.Initialize()
    atexit.register(lambda: lldb.SBDebugger.Terminate())
    unittest2.main()
