"""Test lldb's stepping speed."""

from __future__ import print_function

import sys
import lldb
from lldbsuite.test import configuration
from lldbsuite.test import lldbtest_config
from lldbsuite.test.lldbbench import *
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class SteppingSpeedBench(BenchBase):

    mydir = TestBase.compute_mydir(__file__)

    def setUp(self):
        BenchBase.setUp(self)
        self.exe = lldbtest_config.lldbExec
        self.break_spec = '-n main'
        self.count = 50

        self.trace("self.exe=%s" % self.exe)
        self.trace("self.break_spec=%s" % self.break_spec)

    @benchmarks_test
    @no_debug_info_test
    @expectedFailureAll(
        oslist=["windows"],
        bugnumber="llvm.org/pr22274: need a pexpect replacement for windows")
    def test_run_lldb_steppings(self):
        """Test lldb steppings on a large executable."""
        print()
        self.run_lldb_steppings(self.exe, self.break_spec, self.count)
        print("lldb stepping benchmark:", self.stopwatch)

    def run_lldb_steppings(self, exe, break_spec, count):
        import pexpect
        # Set self.child_prompt, which is "(lldb) ".
        self.child_prompt = '(lldb) '
        prompt = self.child_prompt

        # So that the child gets torn down after the test.
        self.child = pexpect.spawn(
            '%s %s %s' %
            (lldbtest_config.lldbExec, self.lldbOption, exe))
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
                child.sendline('next')  # Aka 'thread step-over'.
                child.expect_exact(prompt)

        child.sendline('quit')
        try:
            self.child.expect(pexpect.EOF)
        except:
            pass

        self.child = None
