"""Test lldb's stepping speed."""

import os, sys
import unittest2
import lldb
import pexpect
from lldbbench import *

class SteppingSpeedBench(BenchBase):

    mydir = os.path.join("benchmarks", "stepping")

    def setUp(self):
        BenchBase.setUp(self)
        if lldb.bmExecutable:
            self.exe = lldb.bmExecutable
            bmExecutableDefauled = False
        else:
            self.exe = self.lldbHere
            bmExecutableDefauled = True
        if lldb.bmBreakpointSpec:
            self.break_spec = lldb.bmBreakpointSpec
        else:
            if bmExecutableDefauled:
                self.break_spec = '-F Driver::MainLoop()'
            else:
                self.break_spec = '-n main'
        self.stepping_avg = None
        #print "self.exe=%s" % self.exe
        #print "self.break_spec=%s" % self.break_spec

    @benchmarks_test
    def test_run_lldb_steppings(self):
        """Test lldb steppings on a large executable."""
        print
        self.run_lldb_steppings(self.exe, self.break_spec, 50)
        print "lldb stepping benchmark:", self.stopwatch

    def run_lldb_steppings(self, exe, break_spec, count):
        # Set self.child_prompt, which is "(lldb) ".
        self.child_prompt = '(lldb) '
        prompt = self.child_prompt

        # So that the child gets torn down after the test.
        self.child = pexpect.spawn('%s %s %s' % (self.lldbHere, self.lldbOption, exe))
        child = self.child

        # Turn on logging for what the child sends back.
        if self.TraceOn():
            child.logfile_read = sys.stdout

        child.expect_exact(prompt)
        child.sendline('breakpoint set %s' % break_spec)
        child.expect_exact(prompt)
        child.sendline('run')
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
