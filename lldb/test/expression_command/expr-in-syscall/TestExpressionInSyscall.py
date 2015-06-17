"""Test that we are able to evaluate expressions when the inferior is blocked in a syscall"""

import os
import unittest2
import lldb
from lldbtest import *
import lldbutil


class ExprSyscallTestCase(TestBase):

    mydir = TestBase.compute_mydir(__file__)

    @skipUnlessDarwin
    @dsym_test
    def test_setpgid_with_dsym(self):
        self.buildDsym()
        self.expr_syscall()

    @expectedFailureAll("llvm.org/pr23659", oslist=["linux"], archs=["i386", "x86_64"])
    @dwarf_test
    def test_setpgid_with_dwarf(self):
        self.buildDwarf()
        self.expr_syscall()

    def expr_syscall(self):
        exe = os.path.join(os.getcwd(), 'a.out')

        # Create a target by the debugger.
        target = self.dbg.CreateTarget(exe)
        self.assertTrue(target, VALID_TARGET)

        listener = lldb.SBListener("my listener")

        # launch the inferior and don't wait for it to stop
        self.dbg.SetAsync(True)
        error = lldb.SBError()
        process = target.Launch (listener,
                None,      # argv
                None,      # envp
                None,      # stdin_path
                None,      # stdout_path
                None,      # stderr_path
                None,      # working directory
                0,         # launch flags
                False,     # Stop at entry
                error)     # error

        self.assertTrue(process and process.IsValid(), PROCESS_IS_VALID)

        event = lldb.SBEvent()

        # Give the child enough time to reach the syscall,
        # while clearing out all the pending events.
        # The last WaitForEvent call will time out after 2 seconds.
        while listener.WaitForEvent(2, event):
            pass

        # now the process should be running (blocked in the syscall)
        self.assertEqual(process.GetState(), lldb.eStateRunning, "Process is running")

        # send the process a signal
        process.SendAsyncInterrupt()
        while listener.WaitForEvent(1, event):
            pass

        # as a result the process should stop
        # in all likelihood we have stopped in the middle of the sleep() syscall
        self.assertEqual(process.GetState(), lldb.eStateStopped, PROCESS_STOPPED)
        thread = process.GetSelectedThread()

        # try evaluating a couple of expressions in this state
        self.expect("expr release_flag = 1", substrs = [" = 1"])
        self.expect("print (int)getpid()", substrs = [str(process.GetProcessID())])

        # and run the process to completion
        process.Continue()

        # process all events
        while listener.WaitForEvent(1, event):
            pass

        self.assertEqual(process.GetState(), lldb.eStateExited)
        self.assertEqual(process.GetExitStatus(), 0)

if __name__ == '__main__':
    import atexit
    lldb.SBDebugger.Initialize()
    atexit.register(lambda: lldb.SBDebugger.Terminate())
    unittest2.main()
