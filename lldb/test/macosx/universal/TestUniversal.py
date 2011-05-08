"""Test aspects of lldb commands on universal binaries."""

import os, time
import unittest2
import lldb
from lldbtest import *

class UniversalTestCase(TestBase):

    mydir = os.path.join("macosx", "universal")

    def setUp(self):
        # Call super's setUp().
        TestBase.setUp(self)
        # Find the line number to break inside main().
        self.line = line_number('main.c', '// Set break point at this line.')

    # rdar://problem/8972204 AddressByteSize of 32-bit process should be 4, got 8 instead.
    @unittest2.skipUnless(sys.platform.startswith("darwin") and os.uname()[4]=='i386',
                          "requires Darwin & i386")
    def test_process_launch_for_universal(self):
        """Test process launch of a universal binary."""
        from lldbutil import print_registers

        # Invoke the default build rule.
        self.buildDefault()

        # Note that "testit" is a universal binary.
        exe = os.path.join(os.getcwd(), "testit")

        # By default, x86_64 is assumed if no architecture is specified.
        self.expect("file " + exe, CURRENT_EXECUTABLE_SET,
            startstr = "Current executable set to ",
            substrs = ["testit' (x86_64)."])

        # Break inside the main.
        self.expect("breakpoint set -f main.c -l %d" % self.line,
                    BREAKPOINT_CREATED,
            startstr = "Breakpoint created: 1: file ='main.c', line = %d, locations = 1" %
                        self.line)

        # We should be able to launch the x86_64 executable.
        self.runCmd("run", RUN_SUCCEEDED)

        # Check whether we have a 64-bit process launched.
        target = self.dbg.GetSelectedTarget()
        process = target.GetProcess()
        self.assertTrue(target.IsValid() and process.IsValid() and
                        self.invoke(process, 'GetAddressByteSize') == 8,
                        "64-bit process launched")

        frame = process.GetThreadAtIndex(0).GetFrameAtIndex(0)
        registers = print_registers(frame, string_buffer=True)
        self.expect(registers, exe=False,
            substrs = ['Name: rax'])

        self.runCmd("continue")

        # Now specify i386 as the architecture for "testit".
        self.expect("file " + exe + " -a i386", CURRENT_EXECUTABLE_SET,
            startstr = "Current executable set to ",
            substrs = ["testit' (i386)."])

        # Break inside the main.
        self.expect("breakpoint set -f main.c -l %d" % self.line,
                    BREAKPOINT_CREATED,
            startstr = "Breakpoint created: 1: file ='main.c', line = %d, locations = 1" %
                        self.line)

        # We should be able to launch the i386 executable as well.
        self.runCmd("run", RUN_SUCCEEDED)

        # Check whether we have a 32-bit process launched.
        target = self.dbg.GetSelectedTarget()
        process = target.GetProcess()
        self.assertTrue(target.IsValid() and process.IsValid(),
                        "32-bit process launched")

        pointerSize = self.invoke(process, 'GetAddressByteSize')
        self.assertTrue(pointerSize == 4,
                        "AddressByteSize of 32-bit process should be 4, got %d instead." % pointerSize)

        frame = process.GetThreadAtIndex(0).GetFrameAtIndex(0)
        registers = print_registers(frame, string_buffer=True)
        self.expect(registers, exe=False,
            substrs = ['Name: eax'])

        self.runCmd("continue")


if __name__ == '__main__':
    import atexit
    lldb.SBDebugger.Initialize()
    atexit.register(lambda: lldb.SBDebugger.Terminate())
    unittest2.main()
