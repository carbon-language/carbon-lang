"""
Test lldb target stop-hook command.
"""

import os
import unittest2
import StringIO
import lldb
from lldbtest import *

class StopHookCmdTestCase(TestBase):

    mydir = os.path.join("functionalities", "stop-hook")

    @unittest2.skipUnless(sys.platform.startswith("darwin"), "requires Darwin")
    def test_with_dsym(self):
        """Test a sequence of target stop-hook commands."""
        self.buildDsym()
        self.stop_hook_cmd_sequence()

    def test_with_dwarf(self):
        """Test a sequence of target stop-hook commands."""
        self.buildDwarf()
        self.stop_hook_cmd_sequence()

    def setUp(self):
        # Call super's setUp().
        TestBase.setUp(self)
        # Find the line numbers inside main.cpp.
        self.begl = line_number('main.cpp', '// Set breakpoint here to test target stop-hook.')
        self.endl = line_number('main.cpp', '// End of the line range for which stop-hook is to be run.')
        self.line = line_number('main.cpp', '// Another breakpoint which is outside of the stop-hook range.')

    def stop_hook_cmd_sequence(self):
        """Test a sequence of target stop-hook commands."""
        exe = os.path.join(os.getcwd(), "a.out")
        self.runCmd("file " + exe, CURRENT_EXECUTABLE_SET)

        self.expect('breakpoint set -f main.cpp -l %d' % self.begl,
                    BREAKPOINT_CREATED,
            startstr = "Breakpoint created: 1: file ='main.cpp', line = %d" %
                        self.begl)
        self.expect('breakpoint set -f main.cpp -l %d' % self.line,
                    BREAKPOINT_CREATED,
            startstr = "Breakpoint created: 2: file ='main.cpp', line = %d" %
                        self.line)

        self.runCmd("target stop-hook add -f main.cpp -l %d -e %d -o 'expr ptr'" % (self.begl, self.endl))

        self.expect('target stop-hook list', 'Stop Hook added successfully',
            substrs = ['State: enabled',
                       'expr ptr'])

        self.runCmd('target stop-hook disable')

        self.expect('target stop-hook list', 'Stop Hook disabled successfully',
            substrs = ['State: disabled',
                       'expr ptr'])

        self.runCmd('target stop-hook enable')

        self.expect('target stop-hook list', 'Stop Hook enabled successfully',
            substrs = ['State: enabled',
                       'expr ptr'])

        self.runCmd("settings set auto-confirm true")
        self.addTearDownHook(lambda: self.runCmd("settings set -r auto-confirm"))

        self.runCmd('target stop-hook delete')

        self.expect('target stop-hook list', 'Stop Hook deleted successfully',
            substrs = ['No stop hooks.'])


if __name__ == '__main__':
    import atexit
    lldb.SBDebugger.Initialize()
    atexit.register(lambda: lldb.SBDebugger.Terminate())
    unittest2.main()
