"""Test that we handle inferiors that send signals to themselves"""

import os
import unittest2
import lldb
from lldbtest import *
import lldbutil


class RaiseTestCase(TestBase):

    mydir = TestBase.compute_mydir(__file__)

    @skipIfWindows # signals do not exist on Windows
    @skipUnlessDarwin
    @dsym_test
    def test_sigstop_with_dsym(self):
        self.buildDsym()
        self.sigstop()

    @skipIfWindows # signals do not exist on Windows
    @dwarf_test
    def test_sigstop_with_dwarf(self):
        self.buildDwarf()
        self.sigstop()

    def launch(self, target):
        # launch the process, do not stop at entry point.
        process = target.LaunchSimple(['SIGSTOP'], None, self.get_process_working_directory())
        self.assertTrue(process, PROCESS_IS_VALID)
        self.assertEqual(process.GetState(), lldb.eStateStopped)
        thread = lldbutil.get_stopped_thread(process, lldb.eStopReasonBreakpoint)
        self.assertTrue(thread.IsValid(), "Thread should be stopped due to a breakpoint")
        return process

    def set_handle(self, signal, stop_at_signal, pass_signal, notify_signal):
        return_obj = lldb.SBCommandReturnObject()
        self.dbg.GetCommandInterpreter().HandleCommand(
                "process handle %s -s %d -p %d -n %d" % (signal, stop_at_signal, pass_signal, notify_signal),
                return_obj)
        self.assertTrue (return_obj.Succeeded() == True, "Setting signal handling failed")


    def sigstop(self):
        """Test that we handle inferior raising SIGSTOP"""
        exe = os.path.join(os.getcwd(), "a.out")

        # Create a target by the debugger.
        target = self.dbg.CreateTarget(exe)
        self.assertTrue(target, VALID_TARGET)
        lldbutil.run_break_set_by_symbol(self, "main")

        # launch
        process = self.launch(target)

        # Make sure we stop at the signal
        self.set_handle("SIGSTOP", 1, 0, 1)
        process.Continue()
        thread = lldbutil.get_stopped_thread(process, lldb.eStopReasonSignal)
        self.assertTrue(thread.IsValid(), "Thread should be stopped due to a signal")
        self.assertTrue(thread.GetStopReasonDataCount() >= 1, "There was data in the event.")
        self.assertEqual(thread.GetStopReasonDataAtIndex(0),
                process.GetUnixSignals().GetSignalNumberFromName('SIGSTOP'),
                "The stop signal was SIGSTOP")

        # Continue until we exit.
        process.Continue()
        self.assertEqual(process.GetState(), lldb.eStateExited)

        # launch again
        process = self.launch(target)

        # Make sure we do not stop at the signal. We should still get the notification.
        self.set_handle("SIGSTOP", 0, 0, 1)
        self.expect("process continue", substrs=["stopped and restarted", "SIGSTOP"])
        self.assertEqual(process.GetState(), lldb.eStateExited)

        # launch again
        process = self.launch(target)

        # Make sure we do not stop at the signal, and we do not get the notification.
        self.set_handle("SIGSTOP", 0, 0, 0)
        self.expect("process continue", substrs=["stopped and restarted"], matching=False)
        self.assertEqual(process.GetState(), lldb.eStateExited)

        # passing of SIGSTOP is not correctly handled, so not testing that scenario: https://llvm.org/bugs/show_bug.cgi?id=23574

        # reset signal handling to default
        self.set_handle("SIGSTOP", 1, 0, 1)

if __name__ == '__main__':
    import atexit
    lldb.SBDebugger.Initialize()
    atexit.register(lambda: lldb.SBDebugger.Terminate())
    unittest2.main()
