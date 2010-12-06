"""
Test that breakpoint by symbol name works correctly dlopen'ing a dynamic lib.
"""

import os, time
import re
import unittest2
import lldb
from lldbtest import *

class LoadUnloadTestCase(TestBase):

    mydir = "load_unload"

    def setUp(self):
        # Call super's setUp().
        TestBase.setUp(self)
        # Find the line number to break for main.cpp.
        self.line = line_number('main.c',
                                '// Set break point at this line for test_lldb_process_load_and_unload_commands().')

    def test_lldb_process_load_and_unload_commands(self):
        """Test that lldb process load/unload command work correctly."""

        # Invoke the default build rule.
        self.buildDefault()

        exe = os.path.join(os.getcwd(), "a.out")
        self.runCmd("file " + exe, CURRENT_EXECUTABLE_SET)

        # Break at main.c before the call to dlopen().
        # Use lldb's process load command to load the dylib, instead.

        self.expect("breakpoint set -f main.c -l %d" % self.line,
                    BREAKPOINT_CREATED,
            startstr = "Breakpoint created: 1: file ='main.c', line = %d" %
                        self.line)

        self.runCmd("run", RUN_SUCCEEDED)

        # Make sure that a_function does not exist at this point.
        self.expect("image lookup -n a_function", "a_function should not exist yet",
                    error=True, matching=False,
            patterns = ["1 match found .* %s" % self.mydir])

        # Use lldb 'process load' to load the dylib.
        self.expect("process load liba.dylib", "liba.dylib loaded correctly",
            patterns = ['Loading "liba.dylib".*ok',
                        'Image [0-9]+ loaded'])

        # Search for and match the "Image ([0-9]+) loaded" pattern.
        output = self.res.GetOutput()
        pattern = re.compile("Image ([0-9]+) loaded")
        for l in output.split(os.linesep):
            #print "l:", l
            match = pattern.search(l)
            if match:
                break
        index = match.group(1)

        # Now we should have an entry for a_function.
        self.expect("image lookup -n a_function", "a_function should now exist",
            patterns = ["1 match found .*%s" % self.mydir])

        # Use lldb 'process unload' to unload the dylib.
        self.expect("process unload %s" % index, "liba.dylib unloaded correctly",
            patterns = ["Unloading .* with index %s.*ok" % index])

        self.runCmd("process continue")


    def test_load_unload(self):
        """Test breakpoint by name works correctly with dlopen'ing."""

        # Invoke the default build rule.
        self.buildDefault()

        exe = os.path.join(os.getcwd(), "a.out")
        self.runCmd("file " + exe, CURRENT_EXECUTABLE_SET)

        # Break by function name a_function (not yet loaded).
        self.expect("breakpoint set -n a_function", BREAKPOINT_CREATED,
            startstr = "Breakpoint created: 1: name = 'a_function', locations = 0 (pending)")

        self.runCmd("run", RUN_SUCCEEDED)

        # The stop reason of the thread should be breakpoint and at a_function.
        self.expect("thread list", STOPPED_DUE_TO_BREAKPOINT,
            substrs = ['state is stopped',
                       'a_function',
                       'stop reason = breakpoint'])

        # The breakpoint should have a hit count of 1.
        self.expect("breakpoint list", BREAKPOINT_HIT_ONCE,
            substrs = [' resolved, hit count = 1'])

        # Issue the 'contnue' command.  We should stop agaian at a_function.
        # The stop reason of the thread should be breakpoint and at a_function.
        self.runCmd("continue")

        # rdar://problem/8508987
        # The a_function breakpoint should be encountered twice.
        self.expect("thread list", STOPPED_DUE_TO_BREAKPOINT,
            substrs = ['state is stopped',
                       'a_function',
                       'stop reason = breakpoint'])

        # The breakpoint should have a hit count of 2.
        self.expect("breakpoint list", BREAKPOINT_HIT_ONCE,
            substrs = [' resolved, hit count = 2'])


if __name__ == '__main__':
    import atexit
    lldb.SBDebugger.Initialize()
    atexit.register(lambda: lldb.SBDebugger.Terminate())
    unittest2.main()
