"""
Test that step-inst over a crash behaves correctly.
"""



import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class CrashDuringStepTestCase(TestBase):

    mydir = TestBase.compute_mydir(__file__)

    def setUp(self):
        TestBase.setUp(self)
        self.breakpoint = line_number('main.cpp', '// Set breakpoint here')

    # IO error due to breakpoint at invalid address
    @expectedFailureAll(triple=re.compile('^mips'))
    @skipIf(oslist=['windows'], archs=['aarch64'])
    def test_step_inst_with(self):
        """Test thread creation during step-inst handling."""
        self.build()
        exe = self.getBuildArtifact("a.out")

        target = self.dbg.CreateTarget(exe)
        self.assertTrue(target and target.IsValid(), "Target is valid")

        self.bp_num = lldbutil.run_break_set_by_file_and_line(
            self, "main.cpp", self.breakpoint, num_expected_locations=1)

        # Run the program.
        process = target.LaunchSimple(
            None, None, self.get_process_working_directory())
        self.assertTrue(process and process.IsValid(), PROCESS_IS_VALID)

        # The stop reason should be breakpoint.
        self.assertEqual(
            process.GetState(),
            lldb.eStateStopped,
            PROCESS_STOPPED)
        thread = lldbutil.get_stopped_thread(
            process, lldb.eStopReasonBreakpoint)
        self.assertTrue(thread.IsValid(), STOPPED_DUE_TO_BREAKPOINT)

        # Keep stepping until the inferior crashes
        while process.GetState() == lldb.eStateStopped and not lldbutil.is_thread_crashed(self, thread):
            thread.StepInstruction(False)

        self.assertEqual(
            process.GetState(),
            lldb.eStateStopped,
            PROCESS_STOPPED)
        self.assertTrue(
            lldbutil.is_thread_crashed(
                self,
                thread),
            "Thread has crashed")
        process.Kill()
