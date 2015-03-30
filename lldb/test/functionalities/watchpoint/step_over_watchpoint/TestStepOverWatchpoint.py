"""Test stepping over watchpoints."""

import unittest2
import lldb
import lldbutil
from lldbtest import *


class TestStepOverWatchpoint(TestBase):

    mydir = TestBase.compute_mydir(__file__)

    def getCategories(self):
        return ['basic_process']

    @skipUnlessDarwin
    @dsym_test
    def test_with_dsym(self):
        """Test stepping over watchpoints."""
        self.buildDsym()
        self.step_over_watchpoint()

    @dwarf_test
    def test_with_dwarf(self):
        """Test stepping over watchpoints."""
        self.buildDwarf()
        self.step_over_watchpoint()

    def setUp(self):
        TestBase.setUp(self)

    def step_inst_for_watchpoint(self, wp_id):
        watchpoint_hit = False
        current_line = self.frame().GetLineEntry().GetLine()
        while self.frame().GetLineEntry().GetLine() == current_line:
            self.thread().StepInstruction(False)  # step_over=False
            stop_reason = self.thread().GetStopReason()
            if stop_reason == lldb.eStopReasonWatchpoint:
                self.assertFalse(watchpoint_hit, "Watchpoint already hit.")
                expected_stop_desc = "watchpoint %d" % wp_id
                actual_stop_desc = self.thread().GetStopDescription(20)
                self.assertTrue(actual_stop_desc == expected_stop_desc,
                                "Watchpoint ID didn't match.")
                watchpoint_hit = True
            else:
                self.assertTrue(stop_reason == lldb.eStopReasonPlanComplete,
                                STOPPED_DUE_TO_STEP_IN)
        self.assertTrue(watchpoint_hit, "Watchpoint never hit.")

    def step_over_watchpoint(self):
        """Test stepping over watchpoints."""
        exe = os.path.join(os.getcwd(), 'a.out')

        target = self.dbg.CreateTarget(exe)
        self.assertTrue(self.target, VALID_TARGET)

        lldbutil.run_break_set_by_symbol(self, 'main')

        process = target.LaunchSimple(None, None,
                                      self.get_process_working_directory())
        self.assertTrue(process.IsValid(), PROCESS_IS_VALID)
        self.assertTrue(process.GetState() == lldb.eStateStopped,
                        PROCESS_STOPPED)

        thread = lldbutil.get_stopped_thread(process,
                                             lldb.eStopReasonBreakpoint)
        self.assertTrue(thread.IsValid(), "Failed to get thread.")

        frame = thread.GetFrameAtIndex(0)
        self.assertTrue(frame.IsValid(), "Failed to get frame.")

        read_value = frame.FindValue('g_watch_me_read',
                                     lldb.eValueTypeVariableGlobal)
        self.assertTrue(read_value.IsValid(), "Failed to find read value.")

        error = lldb.SBError()

        # resolve_location=True, read=True, write=False
        read_watchpoint = read_value.Watch(True, True, False, error)
        self.assertTrue(error.Success(),
                        "Error while setting watchpoint: %s" %
                        error.GetCString())
        self.assertTrue(read_watchpoint, "Failed to set read watchpoint.")

        write_value = frame.FindValue('g_watch_me_write',
                                      lldb.eValueTypeVariableGlobal)
        self.assertTrue(write_value, "Failed to find write value.")

        # resolve_location=True, read=False, write=True
        write_watchpoint = write_value.Watch(True, False, True, error)
        self.assertTrue(read_watchpoint, "Failed to set write watchpoint.")
        self.assertTrue(error.Success(),
                        "Error while setting watchpoint: %s" %
                        error.GetCString())

        thread.StepOver()
        self.assertTrue(thread.GetStopReason() == lldb.eStopReasonWatchpoint,
                        STOPPED_DUE_TO_WATCHPOINT)
        self.assertTrue(thread.GetStopDescription(20) == 'watchpoint 1')

        process.Continue()
        self.assertTrue(process.GetState() == lldb.eStateStopped,
                        PROCESS_STOPPED)
        self.assertTrue(thread.GetStopDescription(20) == 'step over')

        self.step_inst_for_watchpoint(1)

        thread.StepOver()
        self.assertTrue(thread.GetStopReason() == lldb.eStopReasonWatchpoint,
                        STOPPED_DUE_TO_WATCHPOINT)
        self.assertTrue(thread.GetStopDescription(20) == 'watchpoint 2')

        process.Continue()
        self.assertTrue(process.GetState() == lldb.eStateStopped,
                        PROCESS_STOPPED)
        self.assertTrue(thread.GetStopDescription(20) == 'step over')

        self.step_inst_for_watchpoint(2)

if __name__ == '__main__':
    import atexit
    lldb.SBDebugger.Initialize()
    atexit.register(lambda: lldb.SBDebugger.Terminate())
    unittest2.main()
