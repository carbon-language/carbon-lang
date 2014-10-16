"""
Test that the lldb driver's batch mode works correctly.
"""

import os, time
import unittest2
import lldb
import pexpect
from lldbtest import *

class DriverBatchModeTest (TestBase):

    mydir = TestBase.compute_mydir(__file__)

    @unittest2.skipUnless(sys.platform.startswith("darwin"), "requires Darwin")
    @unittest2.expectedFailure("<rdar://problem/18684124>, lldb doesn't reliably print the prompt when run under pexpect")
    @dsym_test
    def test_driver_batch_mode_with_dsym(self):
        """Test that the lldb driver's batch mode works correctly."""
        self.buildDsym()
        self.setTearDownCleanup()
        self.batch_mode ()

    @unittest2.expectedFailure("<rdar://problem/18684124>, lldb doesn't reliably print the prompt when run under pexpect")
    @dwarf_test
    def test_driver_batch_mode_with_dwarf(self):
        """Test that the lldb driver's batch mode works correctly."""
        self.buildDwarf()
        self.setTearDownCleanup()
        self.batch_mode()

    def setUp(self):
        # Call super's setUp().
        TestBase.setUp(self)
        # Our simple source filename.
        self.source = 'main.c'

    def expect_string (self, string):
        """This expects for "string", with timeout & EOF being test fails."""
        try:
            self.child.expect_exact(string)
        except pexpect.EOF:
            self.fail ("Got EOF waiting for '%s'"%(string))
        except pexpect.TIMEOUT:
            self.fail ("Timed out waiting for '%s'"%(string))


    def batch_mode (self):
        exe = os.path.join(os.getcwd(), "a.out")
        prompt = "(lldb) "

        # First time through, pass CRASH so the process will crash and stop in batch mode.
        run_commands = ' -b -o "break set -n main" -o "run" -o "continue" '
        self.child = pexpect.spawn('%s %s %s %s -- CRASH' % (self.lldbHere, self.lldbOption, run_commands, exe))
        child = self.child
        # Turn on logging for what the child sends back.
        if self.TraceOn():
            child.logfile_read = sys.stdout

        # We should see the "run":
        self.expect_string ("run")
        # We should have hit the breakpoint & continued:
        self.expect_string ("continue")
        # The App should have crashed:
        self.expect_string("About to crash")
        # Then we should have a live prompt:
        self.expect_string (prompt)
        self.child.sendline("frame variable touch_me_not")
        self.expect_string ('(char *) touch_me_not')
        
        self.deletePexpectChild()

        # Now do it again, and see make sure if we don't crash, we quit:
        run_commands = ' -b -o "break set -n main" -o "run" -o "continue" '
        self.child = pexpect.spawn('%s %s %s %s -- NOCRASH' % (self.lldbHere, self.lldbOption, run_commands, exe))
        child = self.child
        # Turn on logging for what the child sends back.
        if self.TraceOn():
            child.logfile_read = sys.stdout

        # We should see the "run":
        self.expect_string ("run")
        # We should have hit the breakpoint & continued:
        self.expect_string ("continue")
        # The App should have not have crashed:
        self.expect_string("Got there on time and it did not crash.")
        # Then we should have a live prompt:
        self.expect_string ("exited")
        index = self.child.expect([pexpect.EOF, pexpect.TIMEOUT])
        self.assertTrue(index == 0, "lldb didn't close on successful batch completion.")

        
       

        
        



