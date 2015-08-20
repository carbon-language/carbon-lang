"""
This tests that we do not lose control of the inferior, while doing an instruction-level step
over a thread creation instruction.
"""

import os
import unittest2
import lldb
from lldbtest import *
import lldbutil

class CreateDuringInstructionStepTestCase(TestBase):

    mydir = TestBase.compute_mydir(__file__)

    def setUp(self):
        # Call super's setUp().
        TestBase.setUp(self)
        # Find the line numbers to break and continue.
        self.breakpoint = line_number('main.cpp', '// Set breakpoint here')

    @dsym_test
    def test_step_inst_with_dsym(self):
        self.buildDsym(dictionary=self.getBuildFlags())
        self.create_during_step_inst_test()

    @dwarf_test
    def test_step_inst_with_dwarf(self):
        self.buildDwarf(dictionary=self.getBuildFlags())
        self.create_during_step_inst_test()

    def create_during_step_inst_test(self):
        exe = os.path.join(os.getcwd(), "a.out")
        target = self.dbg.CreateTarget(exe)
        self.assertTrue(target and target.IsValid(), "Target is valid")

        # This should create a breakpoint in the stepping thread.
        self.bp_num = lldbutil.run_break_set_by_file_and_line (self, "main.cpp", self.breakpoint, num_expected_locations=-1)

        # Run the program.
        process = target.LaunchSimple(None, None, self.get_process_working_directory())
        self.assertTrue(process and process.IsValid(), PROCESS_IS_VALID)

        # The stop reason of the thread should be breakpoint.
        self.assertEqual(process.GetState(), lldb.eStateStopped, PROCESS_STOPPED)
        self.assertEqual(lldbutil.get_stopped_thread(process, lldb.eStopReasonBreakpoint).IsValid(), 1,
                STOPPED_DUE_TO_BREAKPOINT)

        # Get the number of threads
        num_threads = process.GetNumThreads()

        # Make sure we see only one threads
        self.assertEqual(num_threads, 1, 'Number of expected threads and actual threads do not match.')

        thread = process.GetThreadAtIndex(0)
        self.assertTrue(thread and thread.IsValid(), "Thread is valid")

        # Keep stepping until we see the thread creation
        while process.GetNumThreads() < 2:
            # This skips some functions we have trouble stepping into. Testing stepping
            # through these is not the purpose of this test. We just want to find the
            # instruction, which creates the thread.
            if thread.GetFrameAtIndex(0).GetFunctionName() in [
                    '__sync_fetch_and_add_4', # Android arm: unable to set a breakpoint for software single-step
                    'pthread_mutex_lock'      # Android arm: function contains atomic instruction sequences
                    ]:
                thread.StepOut()
            else:
                thread.StepInstruction(False)
            self.assertEqual(process.GetState(), lldb.eStateStopped, PROCESS_STOPPED)
            self.assertEqual(thread.GetStopReason(), lldb.eStopReasonPlanComplete, "Step operation succeeded")
            if self.TraceOn():
                self.runCmd("disassemble --pc")

        if self.TraceOn():
            self.runCmd("thread list")

        # We have successfully caught thread creation. Now just run to completion
        process.Continue()

        # At this point, the inferior process should have exited.
        self.assertEqual(process.GetState(), lldb.eStateExited, PROCESS_EXITED)

if __name__ == '__main__':
    import atexit
    lldb.SBDebugger.Initialize()
    atexit.register(lambda: lldb.SBDebugger.Terminate())
    unittest2.main()
