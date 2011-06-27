"""
Test lldb target stop-hook mechanism to see whether it fires off correctly .
"""

import os
import unittest2
import lldb
import pexpect
from lldbtest import *

class StopHookMechanismTestCase(TestBase):

    mydir = os.path.join("functionalities", "stop-hook")

    @unittest2.skipUnless(sys.platform.startswith("darwin"), "requires Darwin")
    def test_with_dsym(self):
        """Test the stop-hook mechanism."""
        self.buildDsym()
        self.stop_hook_firing()

    def test_with_dwarf(self):
        """Test the stop-hook mechanism."""
        self.buildDwarf()
        self.stop_hook_firing()

    def setUp(self):
        # Call super's setUp().
        TestBase.setUp(self)
        # Find the line numbers inside main.cpp.
        self.begl = line_number('main.cpp', '// Set breakpoint here to test target stop-hook.')
        self.endl = line_number('main.cpp', '// End of the line range for which stop-hook is to be run.')
        self.line = line_number('main.cpp', '// Another breakpoint which is outside of the stop-hook range.')

    def stop_hook_firing(self):
        """Test the stop-hook mechanism."""
        exe = os.path.join(os.getcwd(), "a.out")
        prompt = "(lldb) "
        add_prompt = "Enter your stop hook command(s).  Type 'DONE' to end.\r\n> "
        add_prompt1 = "\r\n> "

        # So that the child gets torn down after the test.
        self.child = pexpect.spawn('%s %s' % (self.lldbExec, exe))
        child = self.child
        # Turn on logging for what the child sends back.
        if self.TraceOn():
            child.logfile_read = sys.stdout

        # Set the breakpoint, followed by the target stop-hook commands.
        child.expect_exact(prompt)
        child.sendline('breakpoint set -f main.cpp -l %d' % self.begl)
        child.expect_exact(prompt)
        child.sendline('breakpoint set -f main.cpp -l %d' % self.line)
        child.expect_exact(prompt)
        child.sendline('target stop-hook add -f main.cpp -l %d -e %d' % (self.begl, self.endl))
        child.expect_exact(add_prompt)
        child.sendline('expr ptr')
        child.expect_exact(add_prompt1)
        child.sendline('DONE')
        child.expect_exact(prompt)
        child.sendline('target stop-hook list')

        # Now run the program, expect to stop at the the first breakpoint which is within the stop-hook range.
        child.expect_exact(prompt)
        child.sendline('run')
        child.expect_exact(prompt)
        child.sendline('thread step-over')
        #self.DebugPExpect(child)
        child.expect_exact('** End Stop Hooks **')
        #self.DebugPExpect(child)
        # Verify that the 'Stop Hooks' mechanism is fired off.
        self.expect(child.before, exe=False,
            substrs = ['(void *) $'])

        # Now continue the inferior, we'll stop at another breakpoint which is outside the stop-hook range.
        child.sendline('process continue')
        child.expect_exact('// Another breakpoint which is outside of the stop-hook range.')
        #self.DebugPExpect(child)
        child.sendline('thread step-over')
        child.expect_exact('// Another breakpoint which is outside of the stop-hook range.')
        #self.DebugPExpect(child)
        # Verify that the 'Stop Hooks' mechanism is NOT BEING fired off.
        self.expect(child.before, exe=False, matching=False,
            substrs = ['(void *) $'])
        

if __name__ == '__main__':
    import atexit
    lldb.SBDebugger.Initialize()
    atexit.register(lambda: lldb.SBDebugger.Terminate())
    unittest2.main()
