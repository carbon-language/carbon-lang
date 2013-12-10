"""
Test some lldb platform commands.
"""

import os, time
import unittest2
import lldb
from lldbtest import *

class PlatformCommandTestCase(TestBase):

    mydir = TestBase.compute_mydir(__file__)

    def test_help_platform(self):
        self.runCmd("help platform")

    def test_list(self):
        self.expect("platform list",
            patterns = ['^Available platforms:'])

    def test_process_list(self):
        self.expect("platform process list",
            substrs = ['PID', 'ARCH', 'NAME'])

    def test_process_info_with_no_arg(self):
        """This is expected to fail and to return a proper error message."""
        self.expect("platform process info", error=True,
            substrs = ['one or more process id(s) must be specified'])

    def test_status(self):
        self.expect("platform status",
            substrs = ['Platform', 'Triple', 'OS Version', 'Kernel', 'Hostname'])

    def test_shell(self):
        """ Test that the platform shell command can invoke ls. """
        self.expect("platform shell ls /",
            substrs = ["dev", "tmp", "usr"])

    def test_shell_builtin(self):
        """ Test a shell built-in command (echo) """
        self.expect("platform shell echo hello lldb",
            substrs = ["hello lldb"])

    #FIXME: re-enable once platform shell -t can specify the desired timeout
    def test_shell_timeout(self):
        """ Test a shell built-in command (sleep) that times out """
        self.skipTest("due to taking too long to complete.")
        self.expect("platform shell sleep 15", error=True,
                substrs = ["error: timed out waiting for shell command to complete"])


if __name__ == '__main__':
    import atexit
    lldb.SBDebugger.Initialize()
    atexit.register(lambda: lldb.SBDebugger.Terminate())
    unittest2.main()
