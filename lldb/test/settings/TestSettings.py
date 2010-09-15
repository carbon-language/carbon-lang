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

        # Use '-o' option to override the existing instance setting.
        self.runCmd("settings set -o prompt 'lldb2'")

        # Immediately test the setting.
        self.expect("settings show prompt",
            startstr = "prompt (string) = 'lldb2'")

        # The overall display should also reflect the new setting.
        self.expect("settings show",
            substrs = ["prompt (string) = 'lldb2'"])

    def test_set_term_width(self):
        """Test that 'set term-width' actually changes the term-width."""

        # No '-o' option is needed for static setting.
        self.runCmd("settings set term-width 70")

        # Immediately test the setting.
        self.expect("settings show term-width",
            startstr = "term-width (int) = '70'")

        # The overall display should also reflect the new setting.
        self.expect("settings show",
            startstr = "term-width (int) = '70'")

    @unittest2.skipUnless(sys.platform.startswith("darwin"), "requires Darwin")
    def test_with_dsym(self):
        """Test that run-args and env-vars are passed to the launched process."""
        self.buildDsym()
        self.pass_run_args_and_env_vars()

    def test_with_dwarf(self):
        """Test that run-args and env-vars are passed to the launched process."""
        self.buildDwarf()
        self.pass_run_args_and_env_vars()

    def pass_run_args_and_env_vars(self):
        """Test that run-args and env-vars are passed to the launched process."""
        exe = os.path.join(os.getcwd(), "a.out")
        self.runCmd("file " + exe, CURRENT_EXECUTABLE_SET)

        # Set the run-args and the env-vars.
        self.runCmd('settings set process.run-args A B C')
        self.runCmd('settings set process.env-vars ["MY_ENV_VAR"]=YES')

        self.runCmd("run", RUN_SUCCEEDED)

        # Read the output file produced by running the program.
        output = open('/tmp/output.txt', 'r').read()

        self.assertTrue(output.startswith("argv[1] matches") and
                        output.find("argv[2] matches") > 0 and
                        output.find("argv[3] matches") > 0 and
                        output.find("Environment variable 'MY_ENV_VAR' successfully passed.") > 0)


if __name__ == '__main__':
    import atexit
    lldb.SBDebugger.Initialize()
    atexit.register(lambda: lldb.SBDebugger.Terminate())
    unittest2.main()
