"""Test that lldb functions correctly after the inferior has crashed."""

import os, time
import unittest2
import lldb, lldbutil
from lldbtest import *

class CrashingInferiorTestCase(TestBase):

    mydir = os.path.join("functionalities", "inferior-crashing")

    @unittest2.skipUnless(sys.platform.startswith("darwin"), "requires Darwin")
    def test_inferior_crashing_dsym(self):
        """Test that lldb reliably catches the inferior crashing (command)."""
        self.buildDsym()
        self.inferior_crashing()

    def test_inferior_crashing_dwarf(self):
        """Test that lldb reliably catches the inferior crashing (command)."""
        self.buildDwarf()
        self.inferior_crashing()

    @unittest2.skipUnless(sys.platform.startswith("darwin"), "requires Darwin")
    def test_inferior_crashing_registers_dsym(self):
        """Test that lldb reliably reads registers from the inferior after crashing (command)."""
        self.buildDsym()
        self.inferior_crashing_registers()

    def test_inferior_crashing_register_dwarf(self):
        """Test that lldb reliably reads registers from the inferior after crashing (command)."""
        self.buildDwarf()
        self.inferior_crashing_registers()

    @python_api_test
    def test_inferior_crashing_python(self):
        """Test that lldb reliably catches the inferior crashing (Python API)."""
        self.buildDefault()
        self.inferior_crashing_python()

    @unittest2.skipUnless(sys.platform.startswith("darwin"), "requires Darwin")
    def test_inferior_crashing_expr(self):
        """Test that the lldb expression interpreter can read from the inferior after crashing (command)."""
        self.buildDsym()
        self.inferior_crashing_expr()

    # bugzilla 15784 - the same commands fail when run on Linux from the lldb command line
    def test_inferior_crashing_expr(self):
        """Test that the lldb expression interpreter can read from the inferior after crashing (command)."""
        self.buildDwarf()
        self.inferior_crashing_expr()

    @unittest2.skipUnless(sys.platform.startswith("darwin"), "requires Darwin")
    def test_inferior_crashing_step(self):
        """Test that lldb functions correctly after stepping through a crash."""
        self.buildDsym()
        self.inferior_crashing_step()

    # bugzilla 15784 - the same commands fail when run on Linux from the lldb command line
    def test_inferior_crashing_step(self):
        """Test that lldb functions correctly after stepping through a crash."""
        self.buildDwarf()
        self.inferior_crashing_step()

    def set_breakpoint(self, line):
        lldbutil.run_break_set_by_file_and_line (self, "main.c", line, num_expected_locations=1, loc_exact=True)

    def setUp(self):
        # Call super's setUp().
        TestBase.setUp(self)
        # Find the line number of the crash.
        self.line = line_number('main.c', '// Crash here.')

    def inferior_crashing(self):
        """Inferior crashes upon launching; lldb should catch the event and stop."""
        exe = os.path.join(os.getcwd(), "a.out")
        self.runCmd("file " + exe, CURRENT_EXECUTABLE_SET)

        self.runCmd("run", RUN_SUCCEEDED)

        if sys.platform.startswith("darwin"):
            stop_reason = 'stop reason = EXC_BAD_ACCESS'
        else:
            stop_reason = 'stop reason = invalid address'

        # The stop reason of the thread should be a bad access exception.
        self.expect("thread list", STOPPED_DUE_TO_EXC_BAD_ACCESS,
            substrs = ['stopped',
                       stop_reason])

        # And it should report the correct line number.
        self.expect("thread backtrace all",
            substrs = [stop_reason,
                       'main.c:%d' % self.line])

    def inferior_crashing_python(self):
        """Inferior crashes upon launching; lldb should catch the event and stop."""
        exe = os.path.join(os.getcwd(), "a.out")

        target = self.dbg.CreateTarget(exe)
        self.assertTrue(target, VALID_TARGET)

        # Now launch the process, and do not stop at entry point.
        # Both argv and envp are null.
        process = target.LaunchSimple(None, None, os.getcwd())

        import lldbutil
        if process.GetState() != lldb.eStateStopped:
            self.fail("Process should be in the 'stopped' state, "
                      "instead the actual state is: '%s'" %
                      lldbutil.state_type_to_str(process.GetState()))

        thread = lldbutil.get_stopped_thread(process, lldb.eStopReasonException)
        if not thread:
            self.fail("Fail to stop the thread upon bad access exception")

        if self.TraceOn():
            lldbutil.print_stacktrace(thread)

    def inferior_crashing_registers(self):
        """Test that lldb can read registers after crashing."""
        exe = os.path.join(os.getcwd(), "a.out")
        self.runCmd("file " + exe, CURRENT_EXECUTABLE_SET)

        self.runCmd("run", RUN_SUCCEEDED)

        if sys.platform.startswith("darwin"):
            stop_reason = 'stop reason = EXC_BAD_ACCESS'
        else:
            stop_reason = 'stop reason = invalid address'

        # lldb should be able to read from registers from the inferior after crashing.
        self.expect("register read eax",
            substrs = ['eax = 0x'])

    def inferior_crashing_expr(self):
        """Test that the lldb expression interpreter can read symbols after crashing."""
        exe = os.path.join(os.getcwd(), "a.out")
        self.runCmd("file " + exe, CURRENT_EXECUTABLE_SET)

        self.runCmd("run", RUN_SUCCEEDED)

        if sys.platform.startswith("darwin"):
            stop_reason = 'stop reason = EXC_BAD_ACCESS'
        else:
            stop_reason = 'stop reason = invalid address'

        # The stop reason of the thread should be a bad access exception.
        self.expect("thread list", STOPPED_DUE_TO_EXC_BAD_ACCESS,
            substrs = ['stopped', stop_reason])

        # The lldb expression interpreter should be able to read from addresses of the inferior after a crash.
        self.expect("p argc",
            startstr = '(int) $0 = 1')

        self.expect("p hello_world",
            substrs = ['Hello'])

    def inferior_crashing_step(self):
        """Test that lldb functions correctly after stepping through a crash."""
        exe = os.path.join(os.getcwd(), "a.out")
        self.runCmd("file " + exe, CURRENT_EXECUTABLE_SET)

        self.set_breakpoint(self.line)
        self.runCmd("run", RUN_SUCCEEDED)

        self.expect("thread list", STOPPED_DUE_TO_BREAKPOINT,
            substrs = ['main.c:%d' % self.line,
                       'stop reason = breakpoint'])

        self.runCmd("next")

        if sys.platform.startswith("darwin"):
            stop_reason = 'stop reason = EXC_BAD_ACCESS'
        else:
            stop_reason = 'stop reason = invalid address'

        # The stop reason of the thread should be a bad access exception.
        self.expect("thread list", STOPPED_DUE_TO_EXC_BAD_ACCESS,
            substrs = ['stopped', stop_reason])

        # The lldb expression interpreter should be able to read from addresses of the inferior after a crash.
        self.expect("p argv[0]",
            substrs = ['a.out'])
        self.expect("p null_ptr",
            substrs = ['= 0x0'])

        # lldb should be able to read from registers from the inferior after crashing.
        self.expect("register read eax",
            substrs = ['eax = 0x'])

        # And it should report the correct line number.
        self.expect("thread backtrace all",
            substrs = [stop_reason,
                       'main.c:%d' % self.line])

if __name__ == '__main__':
    import atexit
    lldb.SBDebugger.Initialize()
    atexit.register(lambda: lldb.SBDebugger.Terminate())
    unittest2.main()
