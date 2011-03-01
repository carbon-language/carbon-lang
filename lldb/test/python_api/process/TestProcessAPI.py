"""
Test SBProcess APIs, including ReadMemory(), WriteMemory(), and others.
"""

import os, time
import unittest2
import lldb
from lldbtest import *

class ProcessAPITestCase(TestBase):

    mydir = os.path.join("python_api", "process")

    @unittest2.skipUnless(sys.platform.startswith("darwin"), "requires Darwin")
    @python_api_test
    def test_read_memory_with_dsym(self):
        """Test Python SBProcess.ReadMemory() API."""
        self.buildDsym()
        self.read_memory()

    @python_api_test
    def test_read_memory_with_dwarf(self):
        """Test Python SBProcess.ReadMemory() API."""
        self.buildDwarf()
        self.read_memory()

    @unittest2.skipUnless(sys.platform.startswith("darwin"), "requires Darwin")
    @python_api_test
    def test_write_memory_with_dsym(self):
        """Test Python SBProcess.WriteMemory() API."""
        self.buildDsym()
        self.write_memory()

    @python_api_test
    def test_write_memory_with_dwarf(self):
        """Test Python SBProcess.WriteMemory() API."""
        self.buildDwarf()
        self.write_memory()

    def setUp(self):
        # Call super's setUp().
        TestBase.setUp(self)
        # Find the line number to break inside main().
        self.line = line_number("main.cpp", "// Set break point at this line and check variable 'my_char'.")

    def read_memory(self):
        """Test Python SBProcess.ReadMemory() API."""
        exe = os.path.join(os.getcwd(), "a.out")
        self.runCmd("file " + exe, CURRENT_EXECUTABLE_SET)

        target = self.dbg.CreateTarget(exe)
        self.assertTrue(target.IsValid(), VALID_TARGET)

        breakpoint = target.BreakpointCreateByLocation("main.cpp", self.line)
        self.assertTrue(breakpoint.IsValid(), VALID_BREAKPOINT)

        # Launch the process, and do not stop at the entry point.
        error = lldb.SBError()
        self.process = target.Launch (self.dbg.GetListener(), None, None, os.ctermid(), os.ctermid(), os.ctermid(), None, 0, False, error)

        thread = self.process.GetThreadAtIndex(0);
        frame = thread.GetFrameAtIndex(0);

        # Get the SBValue for the global variable 'my_char'.
        val = frame.FindValue("my_char", lldb.eValueTypeVariableGlobal)
        self.DebugSBValue(frame, val)

        # If the variable does not have a load address, there's no sense continuing.
        if not val.GetLocation(frame).startswith("0x"):
            return

        # OK, let's get the hex location of the variable.
        location = int(val.GetLocation(frame), 16)

        # Due to the typemap magic (see lldb.swig), we pass in 1 to ReadMemory and
        # expect to get a Python string as the result object!
        content = self.process.ReadMemory(location, 1, error)
        if not error.Success():
            self.fail("SBProcess.ReadMemory() failed")
        print "memory content:", content

        self.expect(content, "Result from SBProcess.ReadMemory() matches our expected output: 'x'",
                    exe=False,
            startstr = 'x')

    def write_memory(self):
        """Test Python SBProcess.WriteMemory() API."""
        exe = os.path.join(os.getcwd(), "a.out")
        self.runCmd("file " + exe, CURRENT_EXECUTABLE_SET)

        target = self.dbg.CreateTarget(exe)
        self.assertTrue(target.IsValid(), VALID_TARGET)

        breakpoint = target.BreakpointCreateByLocation("main.cpp", self.line)
        self.assertTrue(breakpoint.IsValid(), VALID_BREAKPOINT)

        # Launch the process, and do not stop at the entry point.
        error = lldb.SBError()
        self.process = target.Launch (self.dbg.GetListener(), None, None, os.ctermid(), os.ctermid(), os.ctermid(), None, 0, False, error)

        thread = self.process.GetThreadAtIndex(0);
        frame = thread.GetFrameAtIndex(0);

        # Get the SBValue for the global variable 'my_char'.
        val = frame.FindValue("my_char", lldb.eValueTypeVariableGlobal)
        self.DebugSBValue(frame, val)

        # If the variable does not have a load address, there's no sense continuing.
        if not val.GetLocation(frame).startswith("0x"):
            return

        # OK, let's get the hex location of the variable.
        location = int(val.GetLocation(frame), 16)

        # The program logic makes the 'my_char' variable to have memory content as 'x'.
        # But we want to use the WriteMemory() API to assign 'a' to the variable.

        # Now use WriteMemory() API to write 'a' into the global variable.
        result = self.process.WriteMemory(location, 'a', error)
        if not error.Success() or result != 1:
            self.fail("SBProcess.WriteMemory() failed")

        # Read from the memory location.  This time it should be 'a'.
        # Due to the typemap magic (see lldb.swig), we pass in 1 to ReadMemory and
        # expect to get a Python string as the result object!
        content = self.process.ReadMemory(location, 1, error)
        if not error.Success():
            self.fail("SBProcess.ReadMemory() failed")
        print "memory content:", content

        self.expect(content, "Result from SBProcess.ReadMemory() matches our expected output: 'a'",
                    exe=False,
            startstr = 'a')


if __name__ == '__main__':
    import atexit
    lldb.SBDebugger.Initialize()
    atexit.register(lambda: lldb.SBDebugger.Terminate())
    unittest2.main()
