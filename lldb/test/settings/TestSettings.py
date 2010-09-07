"""
Test lldb settings command.
"""

import os, time
import unittest2
import lldb
from lldbtest import *

class SettingsCommandTestCase(TestBase):

    mydir = "settings"

    def test_set_prompt(self):
        """Test that 'set prompt' actually changes the prompt."""
        self.runCmd("settings set -o prompt 'lldb2'")
        self.expect("settings show prompt",
            startstr = "prompt (string) = 'lldb2'")
        self.expect("settings show",
            substrs = ["prompt (string) = 'lldb2'"])


if __name__ == '__main__':
    import atexit
    lldb.SBDebugger.Initialize()
    atexit.register(lambda: lldb.SBDebugger.Terminate())
    unittest2.main()
