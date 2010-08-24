"""Test aspects of lldb commands on universal binaries."""

import os, time
import unittest2
import lldb
from lldbtest import *

class TestUniversal(TestBase):

    mydir = "macosx/universal"

    def test_process_launch_for_universal(self):
        """Test process launch of a universal binary."""

        # Note that "testit" is a universal binary.
        exe = os.path.join(os.getcwd(), "testit")

        # By default, x86_64 is assumed if no architecture is specified.
        self.expect("file " + exe, CURRENT_EXECUTABLE_SET,
            startstr = "Current executable set to ",
            substrs = ["testit' (x86_64)."])

        # Break inside the main.
        self.expect("breakpoint set -f main.c -l 5", BREAKPOINT_CREATED,
            startstr = "Breakpoint created: 1: file ='main.c', line = 5, locations = 1")

        # We should be able to launch the x86_64 executable.
        self.runCmd("run", RUN_STOPPED)
        self.runCmd("continue")

        # Now specify i386 as the architecture for "testit".
        self.expect("file " + exe + " -a i386", CURRENT_EXECUTABLE_SET,
            startstr = "Current executable set to ",
            substrs = ["testit' (i386)."])

        # Break inside the main.
        self.expect("breakpoint set -f main.c -l 5", BREAKPOINT_CREATED,
            startstr = "Breakpoint created: 1: file ='main.c', line = 5, locations = 1")

        # We should be able to launch the i386 executable as well.
        self.runCmd("run", RUN_STOPPED)
        self.runCmd("continue")


if __name__ == '__main__':
    import atexit
    lldb.SBDebugger.Initialize()
    atexit.register(lambda: lldb.SBDebugger.Terminate())
    unittest2.main()
