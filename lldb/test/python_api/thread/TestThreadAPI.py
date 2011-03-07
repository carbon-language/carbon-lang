"""
Test SBThread APIs.
"""

import os, time
import unittest2
import lldb
from lldbutil import get_stopped_thread
from lldbtest import *

class ThreadAPITestCase(TestBase):

    mydir = os.path.join("python_api", "thread")

    @unittest2.skipUnless(sys.platform.startswith("darwin"), "requires Darwin")
    @python_api_test
    def test_get_stop_description_with_dsym(self):
        """Test Python SBThread.GetStopDescription() API."""
        self.buildDsym()
        self.get_stop_description()

    @python_api_test
    def test_get_stop_description_with_dwarf(self):
        """Test Python SBThread.GetStopDescription() API."""
        self.buildDwarf()
        self.get_stop_description()

    def setUp(self):
        # Call super's setUp().
        TestBase.setUp(self)
        # Find the line number to break inside main().
        self.line = line_number("main.cpp", "// Set break point at this line and check variable 'my_char'.")

    def get_stop_description(self):
        """Test Python SBProcess.ReadMemory() API."""
        exe = os.path.join(os.getcwd(), "a.out")
        self.runCmd("file " + exe, CURRENT_EXECUTABLE_SET)

        target = self.dbg.CreateTarget(exe)
        self.assertTrue(target.IsValid(), VALID_TARGET)

        breakpoint = target.BreakpointCreateByLocation("main.cpp", self.line)
        self.assertTrue(breakpoint.IsValid(), VALID_BREAKPOINT)
        #self.runCmd("breakpoint list")

        # Launch the process, and do not stop at the entry point.
        error = lldb.SBError()
        self.process = target.Launch (self.dbg.GetListener(), None, None, os.ctermid(), os.ctermid(), os.ctermid(), None, 0, False, error)

        thread = get_stopped_thread(self.process, lldb.eStopReasonBreakpoint)
        self.assertTrue(thread != None, "There should be a thread stopped due to breakpoint")
        #self.runCmd("process status")

        # Due to the typemap magic (see lldb.swig), we pass in an (int)length to GetStopDescription
        # and expect to get a Python string as the result object!
        # The 100 is just an arbitrary number specifying the buffer size.
        stop_description = thread.GetStopDescription(100)
        self.expect(stop_description, exe=False,
            startstr = 'breakpoint')


if __name__ == '__main__':
    import atexit
    lldb.SBDebugger.Initialize()
    atexit.register(lambda: lldb.SBDebugger.Terminate())
    unittest2.main()
