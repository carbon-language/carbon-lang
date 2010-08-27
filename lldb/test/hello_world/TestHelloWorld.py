"""Test Python APIs for target, breakpoint, and process."""

import os, time
import unittest2
import lldb
from lldbtest import *

class TestHelloWorld(TestBase):

    mydir = "hello_world"

    @unittest2.expectedFailure
    def test_hellp_world_python(self):
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
        # SBTarget.LaunchProcess() issue (or is there some race condition)?

        # The following approach of launching a process looks untidy and only
        # works sometimes.
        process = target.LaunchProcess([''], [''], os.ctermid(), False)

        SR = process.GetThreadAtIndex(0).GetStopReason()
        count = 0
        while SR == StopReasonEnum("Invalid") or SR == StopReasonEnum("Signal"):
            print >> sys.stderr, "StopReason =", StopReasonString(SR)

            time.sleep(1.0)
            print >> sys.stderr, "Continuing the process:", process
            process.Continue()

            count = count + 1
            if count == 10:
                print >> sys.stderr, "Reached 10 iterations, giving up..."
                break

            SR = process.GetThreadAtIndex(0).GetStopReason()

        # End of section of launching a process.

        # On the other hand, the following two lines of code are more reliable.
        #self.runCmd("run")
        #process = target.GetProcess()

        self.runCmd("thread backtrace")
        self.runCmd("breakpoint list")
        self.runCmd("thread list")

        # The stop reason of the thread should be breakpoint.
        thread = process.GetThreadAtIndex(0)
        
        print >> sys.stderr, "StopReason =", StopReasonString(thread.GetStopReason())
        self.assertTrue(thread.GetStopReason() == StopReasonEnum("Breakpoint"),
                        STOPPED_DUE_TO_BREAKPOINT)

        # The breakpoint should have a hit count of 1.
        self.assertTrue(breakpoint.GetHitCount() == 1, BREAKPOINT_HIT_ONCE)

        # Now kill the process, and we are done.
        rc = process.Kill()
        self.assertTrue(rc.Success())


if __name__ == '__main__':
    import atexit
    lldb.SBDebugger.Initialize()
    atexit.register(lambda: lldb.SBDebugger.Terminate())
    unittest2.main()
