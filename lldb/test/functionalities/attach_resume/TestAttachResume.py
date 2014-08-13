"""
Test process attach/resume.
"""

import os, time
import unittest2
import lldb
from lldbtest import *
import lldbutil

exe_name = "AttachResume"  # Must match Makefile

class AttachResumeTestCase(TestBase):

    mydir = TestBase.compute_mydir(__file__)

    @expectedFailureFreeBSD('llvm.org/pr19310')
    @dwarf_test
    def test_attach_continue_interrupt_detach(self):
        """Test attach/continue/interrupt/detach"""
        self.buildDwarf()
        self.process_attach_continue_interrupt_detach()

    @expectedFailureLinux('llvm.org/pr19478')
    def process_attach_continue_interrupt_detach(self):
        """Test attach/continue/interrupt/detach"""

        exe = os.path.join(os.getcwd(), exe_name)

        popen = self.spawnSubprocess(exe)
        self.addTearDownHook(self.cleanupSubprocesses)

        self.runCmd("process attach -p " + str(popen.pid))

        self._state = 0
        def process_events():
            event = lldb.SBEvent()
            while self.dbg.GetListener().GetNextEvent(event):
                self._state = lldb.SBProcess.GetStateFromEvent(event)

        # using process.GetState() does not work: llvm.org/pr16172
        def wait_for_state(s, timeout=5):
            t = 0
            period = 0.1
            while self._state != s:
                process_events()
                time.sleep(period)
                t += period
                if t > timeout:
                    return False
            return True

        self.setAsync(True)

        self.runCmd("c")
        self.assertTrue(wait_for_state(lldb.eStateRunning),
            'Process not running after continue')

        self.runCmd("process interrupt")
        self.assertTrue(wait_for_state(lldb.eStateStopped),
            'Process not stopped after interrupt')

        # be sure to continue/interrupt/continue (r204504)
        self.runCmd("c")
        self.assertTrue(wait_for_state(lldb.eStateRunning),
            'Process not running after continue')

        self.runCmd("process interrupt")
        self.assertTrue(wait_for_state(lldb.eStateStopped),
            'Process not stopped after interrupt')

        # check that this breakpoint is auto-cleared on detach (r204752)
        self.runCmd("br set -f main.cpp -l 12")

        self.runCmd("c")
        self.assertTrue(wait_for_state(lldb.eStateRunning),
            'Process not running after continue')

        self.assertTrue(wait_for_state(lldb.eStateStopped),
            'Process not stopped after breakpoint')
        self.expect('br list', 'Breakpoint not hit',
            patterns = ['hit count = 1'])

        self.runCmd("c")
        self.assertTrue(wait_for_state(lldb.eStateRunning),
            'Process not running after continue')

        # make sure to detach while in running state (r204759)
        self.runCmd("detach")
        self.assertTrue(wait_for_state(lldb.eStateDetached),
            'Process not detached after detach')

if __name__ == '__main__':
    import atexit
    lldb.SBDebugger.Initialize()
    atexit.register(lambda: lldb.SBDebugger.Terminate())
    unittest2.main()
