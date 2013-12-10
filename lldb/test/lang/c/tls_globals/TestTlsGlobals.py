"""Test that thread-local storage can be read correctly."""

import os, time
import unittest2
import lldb
from lldbtest import *
import lldbutil

class TlsGlobalTestCase(TestBase):

    mydir = TestBase.compute_mydir(__file__)

    @unittest2.skipUnless(sys.platform.startswith("darwin"), "requires Darwin")
    @dsym_test
    @unittest2.expectedFailure("rdar://7796742")
    def test_with_dsym(self):
        """Test thread-local storage."""
        self.buildDsym()
        self.tls_globals()

    @dwarf_test
    @unittest2.expectedFailure("rdar://7796742")
    def test_with_dwarf(self):
        """Test thread-local storage."""
        self.buildDwarf()
        self.tls_globals()

    def setUp(self):
        TestBase.setUp(self)

        if sys.platform.startswith("freebsd") or sys.platform.startswith("linux"):
            # LD_LIBRARY_PATH must be set so the shared libraries are found on startup
            if "LD_LIBRARY_PATH" in os.environ:
                self.runCmd("settings set target.env-vars " + self.dylibPath + "=" + os.environ["LD_LIBRARY_PATH"] + ":" + os.getcwd())
            else:
                self.runCmd("settings set target.env-vars " + self.dylibPath + "=" + os.getcwd())
            self.addTearDownHook(lambda: self.runCmd("settings remove target.env-vars " + self.dylibPath))

    def tls_globals(self):
        """Test thread-local storage."""

        exe = os.path.join(os.getcwd(), "a.out")
        self.runCmd("file " + exe, CURRENT_EXECUTABLE_SET)

        line1 = line_number('main.c', '// thread breakpoint')
        lldbutil.run_break_set_by_file_and_line (self, "main.c", line1, num_expected_locations=1, loc_exact=True)
        self.runCmd("run", RUN_SUCCEEDED)

        # The stop reason of the thread should be breakpoint.
        self.runCmd("process status", "Get process status")
        self.expect("thread list", STOPPED_DUE_TO_BREAKPOINT,
            substrs = ['stopped',
                       'stop reason = breakpoint'])

        # BUG: sometimes lldb doesn't change threads to the stopped thread.
        # (unrelated to this test).
        self.runCmd("thread select 2", "Change thread")

        # Check that TLS evaluates correctly within the thread.
        self.expect("expr var_static", VARIABLES_DISPLAYED_CORRECTLY,
            patterns = ["\(int\) \$.* = 88"])
        self.expect("expr var_shared", VARIABLES_DISPLAYED_CORRECTLY,
            patterns = ["\(int\) \$.* = 66"])

        # Continue on the main thread
        line2 = line_number('main.c', '// main breakpoint')
        lldbutil.run_break_set_by_file_and_line (self, "main.c", line2, num_expected_locations=1, loc_exact=True)
        self.runCmd("continue", RUN_SUCCEEDED)

        # The stop reason of the thread should be breakpoint.
        self.runCmd("process status", "Get process status")
        self.expect("thread list", STOPPED_DUE_TO_BREAKPOINT,
            substrs = ['stopped',
                       'stop reason = breakpoint'])

        # BUG: sometimes lldb doesn't change threads to the stopped thread.
        # (unrelated to this test).
        self.runCmd("thread select 1", "Change thread")

        # Check that TLS evaluates correctly within the main thread.
        self.expect("expr var_static", VARIABLES_DISPLAYED_CORRECTLY,
            patterns = ["\(int\) \$.* = 44"])
        self.expect("expr var_shared", VARIABLES_DISPLAYED_CORRECTLY,
            patterns = ["\(int\) \$.* = 33"])

if __name__ == '__main__':
    import atexit
    lldb.SBDebugger.Initialize()
    atexit.register(lambda: lldb.SBDebugger.Terminate())
    unittest2.main()
