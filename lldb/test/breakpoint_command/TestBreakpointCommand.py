"""
Test lldb breakpoint command add/list/delete.
"""

import os, time
import unittest2
import lldb
from lldbtest import *

class BreakpointCommandTestCase(TestBase):

    mydir = "breakpoint_command"

    @classmethod
    def classCleanup(cls):
        """Cleanup the test byproduct of breakpoint_command_sequence(self)."""
        system(["/bin/sh", "-c", "rm -f output.txt"])

    @unittest2.skipUnless(sys.platform.startswith("darwin"), "requires Darwin")
    def test_with_dsym(self):
        """Test a sequence of breakpoint command add, list, and delete."""
        self.buildDsym()
        self.breakpoint_command_sequence()
        self.breakpoint_command_script_parameters ()

    def test_with_dwarf(self):
        """Test a sequence of breakpoint command add, list, and delete."""
        self.buildDwarf()
        self.breakpoint_command_sequence()
        self.breakpoint_command_script_parameters ()

    def setUp(self):
        # Call super's setUp().
        TestBase.setUp(self)
        # Find the line number to break inside main().
        self.line = line_number('main.c', '// Set break point at this line.')

    def breakpoint_command_sequence(self):
        """Test a sequence of breakpoint command add, list, and delete."""
        exe = os.path.join(os.getcwd(), "a.out")
        self.runCmd("file " + exe, CURRENT_EXECUTABLE_SET)

        # Add two breakpoints on the same line.
        self.expect("breakpoint set -f main.c -l %d" % self.line,
                    BREAKPOINT_CREATED,
            startstr = "Breakpoint created: 1: file ='main.c', line = %d, locations = 1" %
                        self.line)
        self.expect("breakpoint set -f main.c -l %d" % self.line,
                    BREAKPOINT_CREATED,
            startstr = "Breakpoint created: 2: file ='main.c', line = %d, locations = 1" %
                        self.line)

        # Now add callbacks for the breakpoints just created.
        self.runCmd("breakpoint command add -s command -o 'frame variable -T -s' 1")
        self.runCmd("breakpoint command add -s python -o 'here = open(\"output.txt\", \"w\"); print >> here, \"lldb\"; here.close()' 2")

        # Check that the breakpoint commands are correctly set.

        # The breakpoint list now only contains breakpoint 1.
        self.expect("breakpoint list", "Breakpoints 1 & 2 created",
            substrs = ["1: file ='main.c', line = %d, locations = 1" % self.line,
                       "2: file ='main.c', line = %d, locations = 1" % self.line] )

        self.expect("breakpoint list -f", "Breakpoints 1 & 2 created",
            substrs = ["1: file ='main.c', line = %d, locations = 1" % self.line,
                       "2: file ='main.c', line = %d, locations = 1" % self.line],
            patterns = ["1.1: .+at main.c:%d, .+unresolved, hit count = 0" % self.line,
                        "2.1: .+at main.c:%d, .+unresolved, hit count = 0" % self.line])

        self.expect("breakpoint command list 1", "Breakpoint 1 command ok",
            substrs = ["Breakpoint commands:",
                          "frame variable -T -s"])
        self.expect("breakpoint command list 2", "Breakpoint 2 command ok",
            substrs = ["Breakpoint commands:",
                          "here = open",
                          "print >> here",
                          "here.close()"])

        # Run the program.  Remove 'output.txt' if it exists.
        if os.path.exists('output.txt'):
            os.remove('output.txt')
        self.runCmd("run", RUN_SUCCEEDED)

        # Check that the file 'output.txt' exists and contains the string "lldb".

        # The 'output.txt' file should now exist.
        self.assertTrue(os.path.isfile("output.txt"),
                        "'output.txt' exists due to breakpoint command for breakpoint 2.")

        # Read the output file produced by running the program.
        with open('output.txt', 'r') as f:
            output = f.read()

        self.expect(output, "File 'output.txt' and the content matches", exe=False,
            startstr = "lldb")

        # Finish the program.
        self.runCmd("process continue")

        # Remove the breakpoint command associated with breakpoint 1.
        self.runCmd("breakpoint command delete 1")

        # Remove breakpoint 2.
        self.runCmd("breakpoint delete 2")

        self.expect("breakpoint command list 1",
            startstr = "Breakpoint 1 does not have an associated command.")
        self.expect("breakpoint command list 2", error=True,
            startstr = "error: '2' is not a currently valid breakpoint id.")

        # The breakpoint list now only contains breakpoint 1.
        self.expect("breakpoint list -f", "Breakpoint 1 exists",
            substrs = ["1: file ='main.c', line = %d, locations = 1, resolved = 1" %
                        self.line,
                       "hit count = 1"])

        # Not breakpoint 2.
        self.expect("breakpoint list -f", "No more breakpoint 2", matching=False,
            substrs = ["2: file ='main.c', line = %d, locations = 1, resolved = 1" %
                        self.line])

        # Run the program again, with breakpoint 1 remaining.
        self.runCmd("run", RUN_SUCCEEDED)

        # We should be stopped again due to breakpoint 1.

        # The stop reason of the thread should be breakpoint.
        self.expect("thread list", STOPPED_DUE_TO_BREAKPOINT,
            substrs = ['stopped',
                       'stop reason = breakpoint'])

        # The breakpoint should have a hit count of 2.
        self.expect("breakpoint list -f", BREAKPOINT_HIT_TWICE,
            substrs = ['resolved, hit count = 2'])

    def breakpoint_command_script_parameters (self):
        """Test that the frame and breakpoint location are being properly passed to the script breakpoint command function."""
        exe = os.path.join(os.getcwd(), "a.out")
        self.runCmd("file " + exe, CURRENT_EXECUTABLE_SET)

        # Add a breakpoint.
        self.expect("breakpoint set -f main.c -l %d" % self.line,
                    BREAKPOINT_CREATED,
            startstr = "Breakpoint created: 1: file ='main.c', line = %d, locations = 1" %
                        self.line)

        # Now add callbacks for the breakpoints just created.
        self.runCmd("breakpoint command add -s python -o 'here = open(\"output-2.txt\", \"w\"); print >> here, frame; print >> here, bp_loc; here.close()' 1")

        # Remove 'output-2.txt' if it already exists.

        if (os.path.exists('output-2.txt')):
            os.remove ('output-2.txt')

        # Run program, hit breakpoint, and hopefully write out new version of 'output-2.txt'
        self.runCmd ("run", RUN_SUCCEEDED)

        # Check that the file 'output.txt' exists and contains the string "lldb".

        # The 'output-2.txt' file should now exist.
        self.assertTrue(os.path.isfile("output-2.txt"),
                        "'output-2.txt' exists due to breakpoint command for breakpoint 1.")

        # Read the output file produced by running the program.
        with open('output-2.txt', 'r') as f:
            output = f.read()

        self.expect (output, "File 'output-2.txt' and the content matches", exe=False,
                     startstr = "frame #0:",
                     patterns = ["1.* where = .*main .* resolved, hit count = 1" ])

        # Now remove 'output-2.txt'
        os.remove ('output-2.txt')

if __name__ == '__main__':
    import atexit
    lldb.SBDebugger.Initialize()
    atexit.register(lambda: lldb.SBDebugger.Terminate())
    unittest2.main()
