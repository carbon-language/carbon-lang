"""
Test SBProcess APIs, including ReadMemory(), WriteMemory(), and others.
"""

import os, time
import unittest2
import lldb
from lldbutil import get_stopped_thread, StateTypeString
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

    @unittest2.skipUnless(sys.platform.startswith("darwin"), "requires Darwin")
    @python_api_test
    def test_access_my_int_with_dsym(self):
        """Test access 'my_int' using Python SBProcess.GetByteOrder() and other APIs."""
        self.buildDsym()
        self.access_my_int()

    @python_api_test
    def test_access_my_int_with_dwarf(self):
        """Test access 'my_int' using Python SBProcess.GetByteOrder() and other APIs."""
        self.buildDwarf()
        self.access_my_int()

    @python_api_test
    def test_remote_launch(self):
        """Test SBProcess.RemoteLaunch() API with a process not in eStateConnected, and it should fail."""
        self.buildDefault()
        self.remote_launch_should_fail()

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

        thread = get_stopped_thread(self.process, lldb.eStopReasonBreakpoint)
        self.assertTrue(thread != None, "There should be a thread stopped due to breakpoint")
        frame = thread.GetFrameAtIndex(0)

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

        thread = get_stopped_thread(self.process, lldb.eStopReasonBreakpoint)
        self.assertTrue(thread != None, "There should be a thread stopped due to breakpoint")
        frame = thread.GetFrameAtIndex(0)

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

    def access_my_int(self):
        """Test access 'my_int' using Python SBProcess.GetByteOrder() and other APIs."""
        exe = os.path.join(os.getcwd(), "a.out")
        self.runCmd("file " + exe, CURRENT_EXECUTABLE_SET)

        target = self.dbg.CreateTarget(exe)
        self.assertTrue(target.IsValid(), VALID_TARGET)

        breakpoint = target.BreakpointCreateByLocation("main.cpp", self.line)
        self.assertTrue(breakpoint.IsValid(), VALID_BREAKPOINT)

        # Launch the process, and do not stop at the entry point.
        error = lldb.SBError()
        self.process = target.Launch (self.dbg.GetListener(), None, None, os.ctermid(), os.ctermid(), os.ctermid(), None, 0, False, error)

        thread = get_stopped_thread(self.process, lldb.eStopReasonBreakpoint)
        self.assertTrue(thread != None, "There should be a thread stopped due to breakpoint")
        frame = thread.GetFrameAtIndex(0)

        # Get the SBValue for the global variable 'my_int'.
        val = frame.FindValue("my_int", lldb.eValueTypeVariableGlobal)
        self.DebugSBValue(frame, val)

        # If the variable does not have a load address, there's no sense continuing.
        if not val.GetLocation(frame).startswith("0x"):
            return

        # OK, let's get the hex location of the variable.
        location = int(val.GetLocation(frame), 16)

        # Note that the canonical from of the bytearray is little endian.
        from lldbutil import int_to_bytearray, bytearray_to_int

        byteSize = val.GetByteSize()
        bytes = int_to_bytearray(256, byteSize)

        byteOrder = self.process.GetByteOrder()
        if byteOrder == lldb.eByteOrderBig:
            bytes.reverse()
        elif byteOrder == lldb.eByteOrderLittle:
            pass
        else:
            # Neither big endian nor little endian?  Return for now.
            # Add more logic here if we want to handle other types.
            return

        # The program logic makes the 'my_int' variable to have int type and value of 0.
        # But we want to use the WriteMemory() API to assign 256 to the variable.

        # Now use WriteMemory() API to write 256 into the global variable.
        new_value = str(bytes)
        result = self.process.WriteMemory(location, new_value, error)
        if not error.Success() or result != byteSize:
            self.fail("SBProcess.WriteMemory() failed")

        # Make sure that the val we got originally updates itself to notice the change:
        self.expect(val.GetValue(frame),
                    "SBProcess.ReadMemory() successfully writes (int)256 to the memory location for 'my_int'",
                    exe=False,
            startstr = '256')

        # And for grins, get the SBValue for the global variable 'my_int' again, to make sure that also tracks the new value:
        val = frame.FindValue("my_int", lldb.eValueTypeVariableGlobal)
        self.expect(val.GetValue(frame),
                    "SBProcess.ReadMemory() successfully writes (int)256 to the memory location for 'my_int'",
                    exe=False,
            startstr = '256')

        # Now read the memory content.  The bytearray should have (byte)1 as the second element.
        content = self.process.ReadMemory(location, byteSize, error)
        if not error.Success():
            self.fail("SBProcess.ReadMemory() failed")

        # Use "ascii" as the encoding because each element of 'content' is in the range [0..255].
        new_bytes = bytearray(content, "ascii")

        # The bytearray_to_int utility function expects a little endian bytearray.
        if byteOrder == lldb.eByteOrderBig:
            new_bytes.reverse()

        new_value = bytearray_to_int(new_bytes, byteSize)
        if new_value != 256:
            self.fail("Memory content read from 'my_int' does not match (int)256")

        # Dump the memory content....
        for i in new_bytes:
            print "byte:", i

    def remote_launch_should_fail(self):
        """Test SBProcess.RemoteLaunch() API with a process not in eStateConnected, and it should fail."""
        exe = os.path.join(os.getcwd(), "a.out")
        self.runCmd("file " + exe, CURRENT_EXECUTABLE_SET)

        target = self.dbg.CreateTarget(exe)
        self.assertTrue(target.IsValid(), VALID_TARGET)

        # Launch the process, and do not stop at the entry point.
        error = lldb.SBError()
        process = target.Launch (self.dbg.GetListener(), None, None, os.ctermid(), os.ctermid(), os.ctermid(), None, 0, False, error)

        print "process state:", StateTypeString(process.GetState())
        self.assertTrue(process.GetState() != lldb.eStateConnected)

        success = process.RemoteLaunch(None, None, None, None, None, None, 0, False, error)
        self.assertTrue(not success, "RemoteLaunch() should fail for process state != eStateConnected")


if __name__ == '__main__':
    import atexit
    lldb.SBDebugger.Initialize()
    atexit.register(lambda: lldb.SBDebugger.Terminate())
    unittest2.main()
