"""
Test breakpoint conditions with 'breakpoint modify -c <expr> id'.
"""

import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class BreakpointConditionsTestCase(TestBase):

    mydir = TestBase.compute_mydir(__file__)

    def test_breakpoint_condition_and_run_command(self):
        """Exercise breakpoint condition with 'breakpoint modify -c <expr> id'."""
        self.build()
        self.breakpoint_conditions()

    def test_breakpoint_condition_inline_and_run_command(self):
        """Exercise breakpoint condition inline with 'breakpoint set'."""
        self.build()
        self.breakpoint_conditions(inline=True)

    @add_test_categories(['pyapi'])
    def test_breakpoint_condition_and_python_api(self):
        """Use Python APIs to set breakpoint conditions."""
        self.build()
        self.breakpoint_conditions_python()

    @add_test_categories(['pyapi'])
    def test_breakpoint_invalid_condition_and_python_api(self):
        """Use Python APIs to set breakpoint conditions."""
        self.build()
        self.breakpoint_invalid_conditions_python()

    def setUp(self):
        # Call super's setUp().
        TestBase.setUp(self)
        # Find the line number to of function 'c'.
        self.line1 = line_number(
            'main.c', '// Find the line number of function "c" here.')
        self.line2 = line_number(
            'main.c', "// Find the line number of c's parent call here.")

    def breakpoint_conditions(self, inline=False):
        """Exercise breakpoint condition with 'breakpoint modify -c <expr> id'."""
        exe = self.getBuildArtifact("a.out")
        self.runCmd("file " + exe, CURRENT_EXECUTABLE_SET)

        if inline:
            # Create a breakpoint by function name 'c' and set the condition.
            lldbutil.run_break_set_by_symbol(
                self,
                "c",
                extra_options="-c 'val == 3'",
                num_expected_locations=1,
                sym_exact=True)
        else:
            # Create a breakpoint by function name 'c'.
            lldbutil.run_break_set_by_symbol(
                self, "c", num_expected_locations=1, sym_exact=True)

            # And set a condition on the breakpoint to stop on when 'val == 3'.
            self.runCmd("breakpoint modify -c 'val == 3' 1")

        # Now run the program.
        self.runCmd("run", RUN_SUCCEEDED)

        # The process should be stopped at this point.
        self.expect("process status", PROCESS_STOPPED,
                    patterns=['Process .* stopped'])

        # 'frame variable --show-types val' should return 3 due to breakpoint condition.
        self.expect(
            "frame variable --show-types val",
            VARIABLES_DISPLAYED_CORRECTLY,
            startstr='(int) val = 3')

        # Also check the hit count, which should be 3, by design.
        self.expect("breakpoint list -f", BREAKPOINT_HIT_ONCE,
                    substrs=["resolved = 1",
                             "Condition: val == 3",
                             "hit count = 1"])

        # The frame #0 should correspond to main.c:36, the executable statement
        # in function name 'c'.  And the parent frame should point to
        # main.c:24.
        self.expect("thread backtrace", STOPPED_DUE_TO_BREAKPOINT_CONDITION,
                    #substrs = ["stop reason = breakpoint"],
                    patterns=["frame #0.*main.c:%d" % self.line1,
                              "frame #1.*main.c:%d" % self.line2])

        # Test that "breakpoint modify -c ''" clears the condition for the last
        # created breakpoint, so that when the breakpoint hits, val == 1.
        self.runCmd("process kill")
        self.runCmd("breakpoint modify -c ''")
        self.expect(
            "breakpoint list -f",
            BREAKPOINT_STATE_CORRECT,
            matching=False,
            substrs=["Condition:"])

        # Now run the program again.
        self.runCmd("run", RUN_SUCCEEDED)

        # The process should be stopped at this point.
        self.expect("process status", PROCESS_STOPPED,
                    patterns=['Process .* stopped'])

        # 'frame variable --show-types val' should return 1 since it is the first breakpoint hit.
        self.expect(
            "frame variable --show-types val",
            VARIABLES_DISPLAYED_CORRECTLY,
            startstr='(int) val = 1')

        self.runCmd("process kill")

    def breakpoint_conditions_python(self):
        """Use Python APIs to set breakpoint conditions."""
        target = self.createTestTarget()

        # Now create a breakpoint on main.c by name 'c'.
        breakpoint = target.BreakpointCreateByName('c', 'a.out')
        self.trace("breakpoint:", breakpoint)
        self.assertTrue(breakpoint and
                        breakpoint.GetNumLocations() == 1,
                        VALID_BREAKPOINT)

        # We didn't associate a thread index with the breakpoint, so it should
        # be invalid.
        self.assertEqual(breakpoint.GetThreadIndex(), lldb.UINT32_MAX,
                        "The thread index should be invalid")
        # The thread name should be invalid, too.
        self.assertTrue(breakpoint.GetThreadName() is None,
                        "The thread name should be invalid")

        # Let's set the thread index for this breakpoint and verify that it is,
        # indeed, being set correctly.
        # There's only one thread for the process.
        breakpoint.SetThreadIndex(1)
        self.assertEqual(breakpoint.GetThreadIndex(), 1,
                        "The thread index has been set correctly")

        # Get the breakpoint location from breakpoint after we verified that,
        # indeed, it has one location.
        location = breakpoint.GetLocationAtIndex(0)
        self.assertTrue(location and
                        location.IsEnabled(),
                        VALID_BREAKPOINT_LOCATION)

        # Set the condition on the breakpoint location.
        location.SetCondition('val == 3')
        self.expect(location.GetCondition(), exe=False,
                    startstr='val == 3')

        # Now launch the process, and do not stop at entry point.
        process = target.LaunchSimple(
            None, None, self.get_process_working_directory())
        self.assertTrue(process, PROCESS_IS_VALID)

        # Frame #0 should be on self.line1 and the break condition should hold.
        from lldbsuite.test.lldbutil import get_stopped_thread
        thread = get_stopped_thread(process, lldb.eStopReasonBreakpoint)
        self.assertTrue(
            thread.IsValid(),
            "There should be a thread stopped due to breakpoint condition")
        frame0 = thread.GetFrameAtIndex(0)
        var = frame0.FindValue('val', lldb.eValueTypeVariableArgument)
        self.assertTrue(frame0.GetLineEntry().GetLine() == self.line1 and
                        var.GetValue() == '3')

        # The hit count for the breakpoint should be 1.
        self.assertEqual(breakpoint.GetHitCount(), 1)

        # Test that the condition expression didn't create a result variable:
        options = lldb.SBExpressionOptions()
        value = frame0.EvaluateExpression("$0", options)
        self.assertTrue(value.GetError().Fail(),
                        "Conditions should not make result variables.")
        process.Continue()

    def breakpoint_invalid_conditions_python(self):
        """Use Python APIs to set breakpoint conditions."""
        exe = self.getBuildArtifact("a.out")

        # Create a target by the debugger.
        target = self.dbg.CreateTarget(exe)
        self.assertTrue(target, VALID_TARGET)

        # Now create a breakpoint on main.c by name 'c'.
        breakpoint = target.BreakpointCreateByName('c', 'a.out')
        self.trace("breakpoint:", breakpoint)
        self.assertTrue(breakpoint and
                        breakpoint.GetNumLocations() == 1,
                        VALID_BREAKPOINT)

        # Set the condition on the breakpoint.
        breakpoint.SetCondition('no_such_variable == not_this_one_either')
        self.expect(breakpoint.GetCondition(), exe=False,
                    startstr='no_such_variable == not_this_one_either')

        # Now launch the process, and do not stop at entry point.
        process = target.LaunchSimple(
            None, None, self.get_process_working_directory())
        self.assertTrue(process, PROCESS_IS_VALID)

        # Frame #0 should be on self.line1 and the break condition should hold.
        from lldbsuite.test.lldbutil import get_stopped_thread
        thread = get_stopped_thread(process, lldb.eStopReasonBreakpoint)
        self.assertTrue(
            thread.IsValid(),
            "There should be a thread stopped due to breakpoint condition")
        frame0 = thread.GetFrameAtIndex(0)
        var = frame0.FindValue('val', lldb.eValueTypeVariableArgument)
        self.assertEqual(frame0.GetLineEntry().GetLine(), self.line1)

        # The hit count for the breakpoint should be 1.
        self.assertEqual(breakpoint.GetHitCount(), 1)
