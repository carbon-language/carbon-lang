"""Test lldb reloads the inferior after it was changed during the session."""

import os, time
import unittest2
import lldb
from lldbtest import *
import lldbutil

class ChangedInferiorTestCase(TestBase):

    mydir = TestBase.compute_mydir(__file__)

    @skipUnlessDarwin
    def test_inferior_crashing_dsym(self):
        """Test lldb reloads the inferior after it was changed during the session."""
        self.buildDsym()
        self.inferior_crashing()
        self.cleanup()
        d = {'C_SOURCES': 'main2.c'}
        self.buildDsym(dictionary=d)
        self.setTearDownCleanup(dictionary=d)
        self.inferior_not_crashing()

    def test_inferior_crashing_dwarf(self):
        """Test lldb reloads the inferior after it was changed during the session."""
        self.buildDwarf()
        self.inferior_crashing()
        self.cleanup()
        # lldb needs to recognize the inferior has changed. If lldb needs to check the
        # new module timestamp, make sure it is not the same as the old one, so add a
        # 1 second delay.
        time.sleep(1)
        d = {'C_SOURCES': 'main2.c'}
        self.buildDwarf(dictionary=d)
        self.setTearDownCleanup(dictionary=d)
        self.inferior_not_crashing()

    def setUp(self):
        # Call super's setUp().
        TestBase.setUp(self)
        # Find the line number of the crash.
        self.line1 = line_number('main.c', '// Crash here.')
        self.line2 = line_number('main2.c', '// Not crash here.')

    def inferior_crashing(self):
        """Inferior crashes upon launching; lldb should catch the event and stop."""
        self.exe = os.path.join(os.getcwd(), "a.out")
        self.runCmd("file " + self.exe, CURRENT_EXECUTABLE_SET)

        self.runCmd("run", RUN_FAILED)

        # We should have one crashing thread
        self.assertEquals(
                len(lldbutil.get_crashed_threads(self, self.dbg.GetSelectedTarget().GetProcess())),
                1,
                STOPPED_DUE_TO_EXC_BAD_ACCESS)

        # And it should report the correct line number.
        self.expect("thread backtrace all", substrs = ['main.c:%d' % self.line1])

    def inferior_not_crashing(self):
        """Test lldb reloads the inferior after it was changed during the session."""
        self.runCmd("process kill")
        # Prod the lldb-platform that we have a newly built inferior ready.
        if lldb.lldbtest_remote_sandbox:
            self.runCmd("file " + self.exe, CURRENT_EXECUTABLE_SET)
        self.runCmd("run", RUN_FAILED)
        self.runCmd("process status")

        self.assertNotEquals(
                len(lldbutil.get_crashed_threads(self, self.dbg.GetSelectedTarget().GetProcess())),
                1,
                "Inferior changed, but lldb did not perform a reload")

        # Break inside the main.
        lldbutil.run_break_set_by_file_and_line (self, "main2.c", self.line2, num_expected_locations=1, loc_exact=True)

        self.runCmd("run", RUN_FAILED)

        # The stop reason of the thread should be breakpoint.
        self.expect("thread list", STOPPED_DUE_TO_BREAKPOINT,
            substrs = ['stopped',
                       'stop reason = breakpoint'])

        self.runCmd("frame variable int_ptr")
        self.expect("frame variable *int_ptr",
            substrs = ['= 7'])
        self.expect("expression *int_ptr",
            substrs = ['= 7'])


if __name__ == '__main__':
    import atexit
    lldb.SBDebugger.Initialize()
    atexit.register(lambda: lldb.SBDebugger.Terminate())
    unittest2.main()
