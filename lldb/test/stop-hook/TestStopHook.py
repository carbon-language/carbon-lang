"""
Test lldb target stop-hook command.
"""

import os
import unittest2
import lldb
import pexpect
from lldbtest import *

class StopHookTestCase(TestBase):

    mydir = "stop-hook"

    @unittest2.skipUnless(sys.platform.startswith("darwin"), "requires Darwin")
    def test_with_dsym(self):
        """Test a sequence of target add-hook commands."""
        self.buildDsym()
        self.stop_hook_command_sequence()

    def test_with_dwarf(self):
        """Test a sequence of target add-hook commands."""
        self.buildDwarf()
        self.stop_hook_command_sequence()

    def setUp(self):
        # Call super's setUp().
        TestBase.setUp(self)
        # Find the line numbers inside main.cpp.
        self.begl = line_number('main.cpp', '// Set breakpoint here to test target stop-hook.')
        self.endl = line_number('main.cpp', '// End of the line range for which stop-hook is to be run.')
        self.line = line_number('main.cpp', '// Another breakpoint which is outside of the stop-hook range.')

    def stop_hook_command_sequence(self):
        """Test a sequence of target stop-hook commands."""
        exe = os.path.join(os.getcwd(), "a.out")
        prompt = "\r\n\(lldb\) "
        add_prompt = "Enter your stop hook command\(s\).  Type 'DONE' to end.\r\n> "
        add_prompt1 = "\r\n> "

        child = pexpect.spawn('%s %s' % (self.lldbExec, exe))
        # Turn on logging for what the child sends back.
        if self.TraceOn():
            child.logfile_read = sys.stdout

        # Set the breakpoint, followed by the target stop-hook commands.
        child.expect(prompt)
        child.sendline('breakpoint set -f main.cpp -l %d' % self.begl)
        child.expect(prompt)
        child.sendline('breakpoint set -f main.cpp -l %d' % self.line)
        child.expect(prompt)
        child.sendline('target stop-hook add -f main.cpp -l %d -e %d' % (self.begl, self.endl))
        child.expect(add_prompt)
        child.sendline('expr ptr')
        child.expect(add_prompt1)
        child.sendline('DONE')
        child.expect(prompt)
        child.sendline('target stop-hook list')

        # Now run the program, expect to stop at the the first breakpoint which is within the stop-hook range.
        child.expect(prompt)
        child.sendline('run')
        child.expect(prompt)
        self.DebugPExpect(child)
        child.sendline('thread step-over')
        child.expect(prompt)
        self.DebugPExpect(child)
        # Verify that the 'Stop Hooks' mechanism is fired off.
        self.expect(child.before, exe=False,
            substrs = ['Stop Hooks'])

        # Now continue the inferior, we'll stop at another breakpoint which is outside the stop-hook range.
        child.sendline('process continue')
        child.expect(prompt)
        self.DebugPExpect(child)
        child.sendline('thread step-over')
        child.expect(prompt)
        self.DebugPExpect(child)
        # Verify that the 'Stop Hooks' mechanism is NOT BEING fired off.
        self.expect(child.before, exe=False, matching=False,
            substrs = ['Stop Hooks'])
        

if __name__ == '__main__':
    import atexit
    lldb.SBDebugger.Initialize()
    atexit.register(lambda: lldb.SBDebugger.Terminate())
    unittest2.main()
