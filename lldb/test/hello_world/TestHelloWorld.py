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

    @python_api_test
    def test_with_dwarf_and_attach_to_process_with_id_api(self):
        """Create target, breakpoint, spawn a process, and attach to it.

        Use dwarf map (no dsym) and attach to process with id API.
        """
        self.buildDwarf()
        self.hello_world_attach_api()

    def setUp(self):
        # Call super's setUp().
        TestBase.setUp(self)
        # Get the full path to our executable to be debugged.
        self.exe = os.path.join(os.getcwd(), "hello_world")
        # Find a couple of the line numbers within main.c.
        self.line1 = line_number('main.c', '// Set break point at this line.')
        self.line2 = line_number('main.c', '// Waiting to be attached...')

    def hello_world_python(self, useLaunchAPI):
        """Create target, breakpoint, launch a process, and then kill it."""

        target = self.dbg.CreateTarget(self.exe)

        breakpoint = target.BreakpointCreateByLocation("main.c", self.line1)

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
            self.runCmd("run")

        process = target.GetProcess()
        self.assertTrue(process, PROCESS_IS_VALID)

        thread = process.GetThreadAtIndex(0)
        if thread.GetStopReason() != lldb.eStopReasonBreakpoint:
            from lldbutil import stop_reason_to_str
            self.fail(STOPPED_DUE_TO_BREAKPOINT_WITH_STOP_REASON_AS %
                      stop_reason_to_str(thread.GetStopReason()))

        # The breakpoint should have a hit count of 1.
        self.assertTrue(breakpoint.GetHitCount() == 1, BREAKPOINT_HIT_ONCE)

    def hello_world_attach_api(self):
        """Create target, breakpoint, spawn a process, and attach to it."""

        target = self.dbg.CreateTarget(self.exe)

        # Spawn a new process.
        import subprocess
        popen = subprocess.Popen([self.exe, "abc", "xyz"])
        #print "pid of spawned process: %d" % popen.pid

        listener = lldb.SBListener("my.attach.listener")
        error = lldb.SBError()
        process = target.AttachToProcessWithID(listener, popen.pid, error)

        self.assertTrue(process, PROCESS_IS_VALID)

        # Let's check the stack traces of the attached process.
        import lldbutil
        stacktraces = lldbutil.print_stacktraces(process, string_buffer=True)
        self.expect(stacktraces, exe=False,
            substrs = ['main.c:%d' % self.line2,
                       '(int)argc=3'])


if __name__ == '__main__':
    import atexit
    lldb.SBDebugger.Initialize()
    atexit.register(lambda: lldb.SBDebugger.Terminate())
    unittest2.main()
