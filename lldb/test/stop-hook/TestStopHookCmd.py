"""
Test lldb target stop-hook command.
"""

import os
import unittest2
import lldb
import pexpect
from lldbtest import *

class StopHookCmdTestCase(TestBase):

    mydir = "stop-hook"

    @unittest2.skipUnless(sys.platform.startswith("darwin"), "requires Darwin")
    def test_with_dsym(self):
        """Test a sequence of target add-hook commands."""
        self.buildDsym()
        self.stop_hook_cmd_sequence()

    def test_with_dwarf(self):
        """Test a sequence of target add-hook commands."""
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

        self.runCmd('target stop-hook list')

        # Now run the program, expect to stop at the the first breakpoint which is within the stop-hook range.
        #self.expect('run', 'Stop hook fired',
        #    substrs = '** Stop Hooks **')
        self.runCmd('run')
        self.runCmd('thread step-over')
        self.expect('thread step-over', 'Stop hook fired again',
            substrs = '** Stop Hooks **')

        # Now continue the inferior, we'll stop at another breakpoint which is outside the stop-hook range.
        self.runCmd('process continue')
        # Verify that the 'Stop Hooks' mechanism is NOT BEING fired off.
        self.expect('thread step-over', 'Stop hook should not be fired', matching=False,
            substrs = '** Stop Hooks **')
        

if __name__ == '__main__':
    import atexit
    lldb.SBDebugger.Initialize()
    atexit.register(lambda: lldb.SBDebugger.Terminate())
    unittest2.main()
