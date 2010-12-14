"""
Test lldb settings command.
"""

import os, time
import unittest2
import lldb
from lldbtest import *

class SettingsCommandTestCase(TestBase):

    mydir = "settings"

    @classmethod
    def classCleanup(cls):
        """Cleanup the test byproducts."""
        system(["/bin/sh", "-c", "rm -f output1.txt"])
        system(["/bin/sh", "-c", "rm -f output2.txt"])
        system(["/bin/sh", "-c", "rm -f stdout.txt"])

    def test_set_prompt(self):
        """Test that 'set prompt' actually changes the prompt."""

        # Set prompt to 'lldb2'.
        self.runCmd("settings set prompt 'lldb2'")

        # Immediately test the setting.
        self.expect("settings show prompt", SETTING_MSG("prompt"),
            startstr = "prompt (string) = 'lldb2'")

        # The overall display should also reflect the new setting.
        self.expect("settings show", SETTING_MSG("prompt"),
            substrs = ["prompt (string) = 'lldb2'"])

        # Use '-r' option to reset to the original default prompt.
        self.runCmd("settings set -r prompt")

    def test_set_term_width(self):
        """Test that 'set term-width' actually changes the term-width."""

        self.runCmd("settings set term-width 70")

        # Immediately test the setting.
        self.expect("settings show term-width", SETTING_MSG("term-width"),
            startstr = "term-width (int) = '70'")

        # The overall display should also reflect the new setting.
        self.expect("settings show", SETTING_MSG("term-width"),
            substrs = ["term-width (int) = '70'"])

    def test_set_auto_confirm(self):
        """Test that after 'set auto-confirm true', manual confirmation should not kick in."""
        self.buildDefault()

        exe = os.path.join(os.getcwd(), "a.out")
        self.runCmd("file " + exe, CURRENT_EXECUTABLE_SET)

        self.runCmd("settings set auto-confirm true")

        # Immediately test the setting.
        self.expect("settings show auto-confirm", SETTING_MSG("auto-confirm"),
            startstr = "auto-confirm (boolean) = 'true'")

        # Now 'breakpoint delete' should just work fine without confirmation
        # prompt from the command interpreter.
        self.runCmd("breakpoint set -n main")
        self.expect("breakpoint delete",
            startstr = "All breakpoints removed")

        # Restore the original setting of auto-confirm.
        self.runCmd("settings set -r auto-confirm")
        self.expect("settings show auto-confirm", SETTING_MSG("auto-confirm"),
            startstr = "auto-confirm (boolean) = 'false'")

    @unittest2.skipUnless(sys.platform.startswith("darwin"), "requires Darwin")
    def test_run_args_and_env_vars_with_dsym(self):
        """Test that run-args and env-vars are passed to the launched process."""
        self.buildDsym()
        self.pass_run_args_and_env_vars()

    def test_run_args_and_env_vars_with_dwarf(self):
        """Test that run-args and env-vars are passed to the launched process."""
        self.buildDwarf()
        self.pass_run_args_and_env_vars()

    def pass_run_args_and_env_vars(self):
        """Test that run-args and env-vars are passed to the launched process."""
        exe = os.path.join(os.getcwd(), "a.out")
        self.runCmd("file " + exe, CURRENT_EXECUTABLE_SET)

        # Set the run-args and the env-vars.
        self.runCmd('settings set target.process.run-args A B C')
        self.runCmd('settings set target.process.env-vars ["MY_ENV_VAR"]=YES')
        # And add hooks to restore the settings during tearDown().
        self.addTearDownHook(
            lambda: self.runCmd("settings set -r target.process.run-args"))
        self.addTearDownHook(
            lambda: self.runCmd("settings set -r target.process.env-vars"))

        self.runCmd("run", RUN_SUCCEEDED)

        # Read the output file produced by running the program.
        with open('output2.txt', 'r') as f:
            output = f.read()

        self.expect(output, exe=False,
            substrs = ["argv[1] matches",
                       "argv[2] matches",
                       "argv[3] matches",
                       "Environment variable 'MY_ENV_VAR' successfully passed."])

    def test_pass_host_env_vars(self):
        """Test that the host env vars are passed to the launched process."""
        self.buildDefault()

        exe = os.path.join(os.getcwd(), "a.out")
        self.runCmd("file " + exe, CURRENT_EXECUTABLE_SET)

        # By default, inherit-env is 'true'.
        self.expect('settings show target.process.inherit-env', "Default inherit-env is 'true'",
            startstr = "target.process.inherit-env (boolean) = 'true'")

        # Set some host environment variables now.
        os.environ["MY_HOST_ENV_VAR1"] = "VAR1"
        os.environ["MY_HOST_ENV_VAR2"] = "VAR2"

        # This is the function to unset the two env variables set above.
        def unset_env_variables():
            os.environ.pop("MY_HOST_ENV_VAR1")
            os.environ.pop("MY_HOST_ENV_VAR2")

        self.addTearDownHook(unset_env_variables)
        self.runCmd("run", RUN_SUCCEEDED)

        # Read the output file produced by running the program.
        with open('output1.txt', 'r') as f:
            output = f.read()

        self.expect(output, exe=False,
            substrs = ["The host environment variable 'MY_HOST_ENV_VAR1' successfully passed.",
                       "The host environment variable 'MY_HOST_ENV_VAR2' successfully passed."])

    @unittest2.expectedFailure
    # rdar://problem/8435794
    # settings set target.process.output-path does not seem to work
    def test_set_output_path(self):
        """Test that setting target.process.output-path for the launched process works."""
        self.buildDefault()

        exe = os.path.join(os.getcwd(), "a.out")
        self.runCmd("file " + exe, CURRENT_EXECUTABLE_SET)

        # Set the output-path and verify it is set.
        self.runCmd("settings set target.process.output-path 'stdout.txt'")
        # And add a hook to restore original setting of target.process.output-path
        # later on during tearDown().
        self.addTearDownHook(
            lambda: self.runCmd("settings set -r target.process.output-path"))

        self.expect("settings show target.process.output-path",
                    SETTING_MSG("target.process.output-path"),
            startstr = "target.process.output-path (string) = 'stdout.txt'")

        self.runCmd("run", RUN_SUCCEEDED)

        # The 'stdout.txt' file should now exist.
        self.assertTrue(os.path.isfile("stdout.txt"),
                        "'stdout.txt' exists due to target.process.output-path.")

        # Read the output file produced by running the program.
        with open('stdout.txt', 'r') as f:
            output = f.read()

        self.expect(output, exe=False,
            startstr = "This message should go to standard out.")


if __name__ == '__main__':
    import atexit
    lldb.SBDebugger.Initialize()
    atexit.register(lambda: lldb.SBDebugger.Terminate())
    unittest2.main()
