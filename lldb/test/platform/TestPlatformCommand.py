"""
Test some lldb platform commands.
"""

import os, time
import unittest2
import lldb
from lldbtest import *

class PlatformCommandTestCase(TestBase):

    mydir = "platform"

    def test_help_platform(self):
        self.runCmd("help platform")

    def test_list(self):
        self.expect("platform list",
            patterns = ['^Available platforms:'])

    def test_process_list(self):
        self.expect("platform process list",
            substrs = ['PID', 'ARCH', 'NAME'])

    def test_status(self):
        self.expect("platform status",
            substrs = ['Platform', 'Triple', 'OS Version', 'Kernel', 'Hostname'])


if __name__ == '__main__':
    import atexit
    lldb.SBDebugger.Initialize()
    atexit.register(lambda: lldb.SBDebugger.Terminate())
    unittest2.main()
