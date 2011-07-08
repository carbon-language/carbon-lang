"""Show global variables and check that they do indeed have global scopes."""

import os, time
import unittest2
import lldb
from lldbtest import *

class GlobalVariablesTestCase(TestBase):

    mydir = os.path.join("lang", "c", "global_variables")

    @unittest2.skipUnless(sys.platform.startswith("darwin"), "requires Darwin")
    def test_with_dsym(self):
        """Test 'frame variable -s -a' which omits args and shows scopes."""
        self.buildDsym()
        self.global_variables()

    def test_with_dwarf(self):
        """Test 'frame variable -s -a' which omits args and shows scopes."""
        self.buildDwarf()
        self.global_variables()

    def setUp(self):
        # Call super's setUp().
        TestBase.setUp(self)
        # Find the line number to break inside main().
        self.line = line_number('main.c', '// Set break point at this line.')

    def global_variables(self):
        """Test 'frame variable -s -a' which omits args and shows scopes."""
        exe = os.path.join(os.getcwd(), "a.out")
        self.runCmd("file " + exe, CURRENT_EXECUTABLE_SET)

        # Break inside the main.
        self.expect("breakpoint set -f main.c -l %d" % self.line,
                    BREAKPOINT_CREATED,
            startstr = "Breakpoint created: 1: file ='main.c', line = %d, locations = 1" %
                        self.line)

        self.runCmd("run", RUN_SUCCEEDED)

        # The stop reason of the thread should be breakpoint.
        self.expect("thread list", STOPPED_DUE_TO_BREAKPOINT,
            substrs = ['stopped',
                       'stop reason = breakpoint'])

        # The breakpoint should have a hit count of 1.
        self.expect("breakpoint list -f", BREAKPOINT_HIT_ONCE,
            substrs = [' resolved, hit count = 1'])

        # Check that GLOBAL scopes are indicated for the variables.
        self.expect("frame variable -T -s -g -a", VARIABLES_DISPLAYED_CORRECTLY,
            substrs = ['GLOBAL: (int) g_file_global_int = 42',
                       'GLOBAL: (const char *) g_file_global_cstr',
                       '"g_file_global_cstr"',
                       'GLOBAL: (const char *) g_file_static_cstr',
                       '"g_file_static_cstr"'])

        # 'frame variable' should support address-of operator.
        self.runCmd("frame variable &g_file_global_int")

        # Exercise the 'target variable' command to display globals in a.c file.
        self.expect("target variable g_a", VARIABLES_DISPLAYED_CORRECTLY,
                    substrs = ['g_a', '123'])
        self.expect("target variable g_marked_spot.x", VARIABLES_DISPLAYED_CORRECTLY,
                    substrs = ['g_marked_spot.x', '20'])

        # rdar://problem/9747668
        # runCmd: target variable g_marked_spot.y
        # output: (int) g_marked_spot.y = <a.o[0x214] can't be resolved,  in not currently loaded.
        #         >
        self.expect("target variable g_marked_spot.y", VARIABLES_DISPLAYED_CORRECTLY,
                    substrs = ['g_marked_spot.y', '21'])


if __name__ == '__main__':
    import atexit
    lldb.SBDebugger.Initialize()
    atexit.register(lambda: lldb.SBDebugger.Terminate())
    unittest2.main()
