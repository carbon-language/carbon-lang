"""Test lldb's stepping speed."""

import os, sys
import unittest2
import lldb
import pexpect
from lldbbench import *

class RunHooksThenSteppingsBench(BenchBase):

    mydir = os.path.join("benchmarks", "stepping")

    def setUp(self):
        BenchBase.setUp(self)
        self.stepping_avg = None

    @benchmarks_test
    def test_lldb_runhooks_then_steppings(self):
        """Test lldb steppings on a large executable."""
        print
        self.run_lldb_runhooks_then_steppings(50)
        print "lldb stepping benchmark:", self.stopwatch

    def run_lldb_runhooks_then_steppings(self, count):
        # Set self.child_prompt, which is "(lldb) ".
        self.child_prompt = '(lldb) '
        prompt = self.child_prompt

        self.child = pexpect.spawn('%s %s' % (self.lldbHere, self.lldbOption))
        self.child.expect_exact(prompt)
        # So that the child gets torn down after the test.
        child = self.child

        # Turn on logging for what the child sends back.
        if self.TraceOn():
            child.logfile_read = sys.stdout

        #lldb.runHooks = ['process attach -n Mail']

        # Perform the run hooks to bring lldb debugger to the desired state.
        if not lldb.runHooks:
            self.skipTest("No runhooks specified for lldb, skip the test")
        for hook in lldb.runHooks:
            child.sendline(hook)
            child.expect_exact(prompt)

        # Reset the stopwatch now.
        self.stopwatch.reset()
        for i in range(count):
            with self.stopwatch:
                # Disassemble the function.
                child.sendline('next') # Aka 'thread step-over'.
                child.expect_exact(prompt)

        child.sendline('quit')
        try:
            self.child.expect(pexpect.EOF)
        except:
            pass

        self.stepping_avg = self.stopwatch.avg()
        if self.TraceOn():
            print "lldb stepping benchmark:", str(self.stopwatch)
        self.child = None


if __name__ == '__main__':
    import atexit
    lldb.SBDebugger.Initialize()
    atexit.register(lambda: lldb.SBDebugger.Terminate())
    unittest2.main()
