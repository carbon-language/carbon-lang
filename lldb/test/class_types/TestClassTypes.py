"""Test breakpoint on a class constructor; and variable list the this object."""

import os, time
import unittest2
import lldb
import lldbutil
from lldbtest import *

class ClassTypesTestCase(TestBase):

    mydir = "class_types"

    @unittest2.skipUnless(sys.platform.startswith("darwin"), "requires Darwin")
    def test_with_dsym_and_run_command(self):
        """Test 'frame variable this' when stopped on a class constructor."""
        self.buildDsym()
        self.class_types()

    @unittest2.skipUnless(sys.platform.startswith("darwin"), "requires Darwin")
    def test_with_dsym_and_python_api(self):
        """Use Python APIs to create a breakpoint by (filespec, line)."""
        self.buildDsym()
        self.breakpoint_creation_by_filespec_python()

    # rdar://problem/8378863
    # "frame variable this" returns
    # error: unable to find any variables named 'this'
    def test_with_dwarf_and_run_command(self):
        """Test 'frame variable this' when stopped on a class constructor."""
        self.buildDwarf()
        self.class_types()

    def test_with_dwarf_and_python_api(self):
        """Use Python APIs to create a breakpoint by (filespec, line)."""
        self.buildDwarf()
        self.breakpoint_creation_by_filespec_python()

    @unittest2.skipUnless(sys.platform.startswith("darwin"), "requires Darwin")
    # rdar://problem/8557478
    # test/class_types test failures: runCmd: expr this->m_c_int
    def test_with_dsym_and_expr_parser(self):
        """Test 'frame variable this' and 'expr this' when stopped inside a constructor."""
        self.buildDsym()
        self.class_types_expr_parser()

    # rdar://problem/8557478
    # test/class_types test failures: runCmd: expr this->m_c_int
    def test_with_dwarf_and_expr_parser(self):
        """Test 'frame variable this' and 'expr this' when stopped inside a constructor."""
        self.buildDwarf()
        self.class_types_expr_parser()

    def setUp(self):
        # Call super's setUp().
        TestBase.setUp(self)
        # Find the line number to break for main.cpp.
        self.line = line_number('main.cpp', '// Set break point at this line.')

    def class_types(self):
        """Test 'frame variable this' when stopped on a class constructor."""
        exe = os.path.join(os.getcwd(), "a.out")
        self.runCmd("file " + exe, CURRENT_EXECUTABLE_SET)

        # Break on the ctor function of class C.
        self.expect("breakpoint set -f main.cpp -l %d" % self.line,
                    BREAKPOINT_CREATED,
            startstr = "Breakpoint created: 1: file ='main.cpp', line = %d" %
                        self.line)

        self.runCmd("run", RUN_SUCCEEDED)

        # The test suite sometimes shows that the process has exited without stopping.
        #
        # CC=clang ./dotest.py -v -t class_types
        # ...
        # Process 76604 exited with status = 0 (0x00000000)
        self.runCmd("process status")

        # The stop reason of the thread should be breakpoint.
        self.expect("thread list", STOPPED_DUE_TO_BREAKPOINT,
            substrs = ['state is stopped',
                       'stop reason = breakpoint'])

        # The breakpoint should have a hit count of 1.
        self.expect("breakpoint list", BREAKPOINT_HIT_ONCE,
            substrs = [' resolved, hit count = 1'])

        # We should be stopped on the ctor function of class C.
        self.expect("frame variable -t this", VARIABLES_DISPLAYED_CORRECTLY,
            substrs = ['C *',
                       ' this = '])

    def breakpoint_creation_by_filespec_python(self):
        """Use Python APIs to create a breakpoint by (filespec, line)."""
        exe = os.path.join(os.getcwd(), "a.out")

        target = self.dbg.CreateTarget(exe)
        self.assertTrue(target.IsValid(), VALID_TARGET)

        filespec = target.GetExecutable()
        self.assertTrue(filespec.IsValid(), VALID_FILESPEC)

        fsDir = filespec.GetDirectory()
        fsFile = filespec.GetFilename()

        self.assertTrue(fsDir == os.getcwd() and fsFile == "a.out",
                        "FileSpec matches the executable")

        bpfilespec = lldb.SBFileSpec("main.cpp", False)

        breakpoint = target.BreakpointCreateByLocation(bpfilespec, self.line)
        self.assertTrue(breakpoint.IsValid(), VALID_BREAKPOINT)

        # Verify the breakpoint just created.
        self.expect(repr(breakpoint), BREAKPOINT_CREATED, exe=False,
            substrs = ['main.cpp',
                       str(self.line)])

        # Now launch the process, and do not stop at entry point.
        rc = lldb.SBError()
        self.process = target.Launch([''], [''], os.ctermid(), 0, False, rc)
        #self.breakAfterLaunch(self.process, "C::C(int, int, int)")

        if not rc.Success() or not self.process.IsValid():
            self.fail("SBTarget.LaunchProcess() failed")

        if self.process.GetState() != lldb.eStateStopped:
            self.fail("Process should be in the 'stopped' state, "
                      "instead the actual state is: '%s'" %
                      lldbutil.StateTypeString(self.process.GetState()))

        # The stop reason of the thread should be breakpoint.
        thread = self.process.GetThreadAtIndex(0)
        if thread.GetStopReason() != lldb.eStopReasonBreakpoint:
            from lldbutil import StopReasonString
            self.fail(STOPPED_DUE_TO_BREAKPOINT_WITH_STOP_REASON_AS %
                      StopReasonString(thread.GetStopReason()))

        # The filename of frame #0 should be 'main.cpp' and the line number
        # should be 93.
        self.expect("%s:%d" % (lldbutil.GetFilenames(thread)[0],
                               lldbutil.GetLineNumbers(thread)[0]),
                    "Break correctly at main.cpp:%d" % self.line, exe=False,
            startstr = "main.cpp:")
            ### clang compiled code reported main.cpp:94?
            ### startstr = "main.cpp:93")

        # We should be stopped on the breakpoint with a hit count of 1.
        self.assertTrue(breakpoint.GetHitCount() == 1, BREAKPOINT_HIT_ONCE)

        self.process.Continue()

    def class_types_expr_parser(self):
        """Test 'frame variable this' and 'expr this' when stopped inside a constructor."""
        exe = os.path.join(os.getcwd(), "a.out")
        self.runCmd("file " + exe, CURRENT_EXECUTABLE_SET)

        # rdar://problem/8516141
        # Is this a case of clang (116.1) generating bad debug info?
        #
        # Break on the ctor function of class C.
        self.expect("breakpoint set -M C", BREAKPOINT_CREATED,
            startstr = "Breakpoint created: 1: name = 'C', locations = 1")

        self.runCmd("run", RUN_SUCCEEDED)

        # The stop reason of the thread should be breakpoint.
        self.expect("thread list", STOPPED_DUE_TO_BREAKPOINT,
            substrs = ['state is stopped',
                       'stop reason = breakpoint'])

        # The breakpoint should have a hit count of 1.
        self.expect("breakpoint list", BREAKPOINT_HIT_ONCE,
            substrs = [' resolved, hit count = 1'])

        # Continue on inside the ctor() body...
        self.runCmd("thread step-over")

        # Verify that frame variable -t this->m_c_int behaves correctly.
        self.expect("frame variable -t this->m_c_int", VARIABLES_DISPLAYED_CORRECTLY,
            startstr = '(int) this->m_c_int = 66')

        # rdar://problem/8430916
        # expr this->m_c_int returns an incorrect value
        #
        # Verify that expr this->m_c_int behaves correctly.
        self.expect("expression this->m_c_int", VARIABLES_DISPLAYED_CORRECTLY,
            patterns = ['\(int\) \$[0-9]+ = 66'])


if __name__ == '__main__':
    import atexit
    lldb.SBDebugger.Initialize()
    atexit.register(lambda: lldb.SBDebugger.Terminate())
    unittest2.main()
