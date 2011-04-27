"""Test Python APIs for target, breakpoint, and process."""

import os, sys, time
import unittest2
import lldb
from lldbtest import *

class HelloWorldTestCase(TestBase):

    mydir = "hello_world"

    @unittest2.skipUnless(sys.platform.startswith("darwin"), "requires Darwin")
    def test_with_dsym_and_run_command(self):
        """Create target, breakpoint, launch a process, and then kill it.

        Use dsym info and lldb "run" command.
        """
        self.buildDsym()
        self.hello_world_python(useLaunchAPI = False)

    @python_api_test
    def test_with_dwarf_and_process_launch_api(self):
        """Create target, breakpoint, launch a process, and then kill it.

        Use dwarf map (no dsym) and process launch API.
        """
        self.buildDwarf()
        self.hello_world_python(useLaunchAPI = True)

    def hello_world_python(self, useLaunchAPI):
        """Create target, breakpoint, launch a process, and then kill it."""

        exe = os.path.join(os.getcwd(), "a.out")

        target = self.dbg.CreateTarget(exe)

        breakpoint = target.BreakpointCreateByLocation("main.c", 4)

        # The default state after breakpoint creation should be enabled.
        self.assertTrue(breakpoint.IsEnabled(),
                        "Breakpoint should be enabled after creation")

        breakpoint.SetEnabled(False)
        self.assertTrue(not breakpoint.IsEnabled(),
                        "Breakpoint.SetEnabled(False) works")

        breakpoint.SetEnabled(True)
        self.assertTrue(breakpoint.IsEnabled(),
                        "Breakpoint.SetEnabled(True) works")

        # rdar://problem/8364687
        # SBTarget.Launch() issue (or is there some race condition)?

        if useLaunchAPI:
            process = target.LaunchSimple(None, None, os.getcwd())
            # The following isn't needed anymore, rdar://8364687 is fixed.
            #
            # Apply some dances after LaunchProcess() in order to break at "main".
            # It only works sometimes.
            #self.breakAfterLaunch(process, "main")
        else:
            # On the other hand, the following line of code are more reliable.
            self.runCmd("run", setCookie=False)

        #self.runCmd("thread backtrace")
        #self.runCmd("breakpoint list")
        #self.runCmd("thread list")

        self.process = target.GetProcess()
        self.assertTrue(self.process.IsValid(), PROCESS_IS_VALID)

        thread = self.process.GetThreadAtIndex(0)
        if thread.GetStopReason() != lldb.eStopReasonBreakpoint:
            from lldbutil import stop_reason_to_str
            self.fail(STOPPED_DUE_TO_BREAKPOINT_WITH_STOP_REASON_AS %
                      stop_reason_to_str(thread.GetStopReason()))

        # The breakpoint should have a hit count of 1.
        self.assertTrue(breakpoint.GetHitCount() == 1, BREAKPOINT_HIT_ONCE)


if __name__ == '__main__':
    import atexit
    lldb.SBDebugger.Initialize()
    atexit.register(lambda: lldb.SBDebugger.Terminate())
    unittest2.main()
