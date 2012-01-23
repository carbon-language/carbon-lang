"""
Test the lldb command line completion mechanism.
"""

import os
import unittest2
import lldb
import pexpect
from lldbtest import *

class CommandLineCompletionTestCase(TestBase):

    mydir = os.path.join("functionalities", "completion")

    @classmethod
    def classCleanup(cls):
        """Cleanup the test byproducts."""
        system(["/bin/sh", "-c", "rm -f child_send.txt"])
        system(["/bin/sh", "-c", "rm -f child_read.txt"])

    def test_settings_append_target_er(self):
        """Test that 'settings append target.er' completes to 'settings append target.error-path'."""
        self.complete_from_to('settings append target.er', 'settings append target.error-path')

    def test_settings_insert_after_target_en(self):
        """Test that 'settings insert-after target.en' completes to 'settings insert-after target.env-vars'."""
        self.complete_from_to('settings insert-after target.en', 'settings insert-after target.env-vars')

    def test_settings_insert_before_target_en(self):
        """Test that 'settings insert-before target.en' completes to 'settings insert-before target.env-vars'."""
        self.complete_from_to('settings insert-before target.en', 'settings insert-before target.env-vars')

    def test_settings_replace_target_ru(self):
        """Test that 'settings replace target.ru' completes to 'settings replace target.run-args'."""
        self.complete_from_to('settings replace target.ru', 'settings replace target.run-args')

    def test_settings_s(self):
        """Test that 'settings s' completes to ['Available completions:', 'set', 'show']."""
        self.complete_from_to('settings s', ['Available completions:', 'set', 'show'])

    def test_settings_set_th(self):
        """Test that 'settings set th' completes to 'settings set thread-format'."""
        self.complete_from_to('settings set th', 'settings set thread-format')

    def test_settings_s_dash(self):
        """Test that 'settings set -' completes to ['Available completions:', '-n', '-r']."""
        self.complete_from_to('settings set -', ['Available completions:', '-n', '-r'])

    def test_settings_set_dash_r_th(self):
        """Test that 'settings set -r th' completes to 'settings set -r thread-format'."""
        self.complete_from_to('settings set -r th', 'settings set -r thread-format')

    def test_settings_set_ta(self):
        """Test that 'settings set ta' completes to 'settings set target.'."""
        self.complete_from_to('settings set ta', 'settings set target.')

    def test_settings_set_target_pr(self):
        """Test that 'settings set target.pr' completes to ['Available completions:',
        'target.prefer-dynamic-value', 'target.process.']."""
        self.complete_from_to('settings set target.pr',
                              ['Available completions:',
                               'target.prefer-dynamic-value',
                               'target.process.'])

    def test_settings_set_target_process(self):
        """Test that 'settings set target.process' completes to 'settings set target.process.'."""
        self.complete_from_to('settings set target.process', 'settings set target.process.')

    def test_settings_set_target_process_dot(self):
        """Test that 'settings set target.process.' completes to 'settings set target.process.thread.'."""
        self.complete_from_to('settings set target.process.', 'settings set target.process.thread.')

    def test_settings_set_target_process_thread_dot(self):
        """Test that 'settings set target.process.thread.' completes to ['Available completions:',
        'target.process.thread.step-avoid-regexp', 'target.process.thread.trace-thread']."""
        self.complete_from_to('settings set target.process.thread.',
                              ['Available completions:',
                               'target.process.thread.step-avoid-regexp',
                               'target.process.thread.trace-thread'])

    def complete_from_to(self, str_input, patterns):
        """Test the completion mechanism completes str_input to pattern, where
        patterns could be a pattern-string or a list of pattern-strings"""
        prompt = "(lldb) "
        add_prompt = "Enter your stop hook command(s).  Type 'DONE' to end.\r\n> "
        add_prompt1 = "\r\n> "

        # So that the child gets torn down after the test.
        self.child = pexpect.spawn('%s %s' % (self.lldbHere, self.lldbOption))
        child = self.child
        # Turn on logging for input/output to/from the child.
        with open('child_send.txt', 'w') as f_send:
            with open('child_read.txt', 'w') as f_read:
                child.logfile_send = f_send
                child.logfile_read = f_read

                child.expect_exact(prompt)
                child.setecho(True)
                # Sends str_input and a Tab to invoke the completion machinery.
                child.send("%s\t" % str_input)
                child.sendline('')
                child.expect_exact(prompt)

        # Set logfile to None to stop logging.
        child.logfile_send = None
        child.logfile_read = None
        
        with open('child_send.txt', 'r') as fs:
            if self.TraceOn():
                print "\n\nContents of child_send.txt:"
                print fs.read()
        with open('child_read.txt', 'r') as fr:
            from_child = fr.read()
            if self.TraceOn():
                print "\n\nContents of child_read.txt:"
                print from_child

            self.assertFalse(patterns is None)
            if type(patterns) is not types.ListType:
                patterns = [patterns]

            # Test that str_input completes to our patterns.
            # If each pattern matches from_child, the completion mechanism works!
            for p in patterns:
                self.expect(from_child, msg=COMPLETIOND_MSG(str_input, p), exe=False,
                    patterns = [p])


if __name__ == '__main__':
    import atexit
    lldb.SBDebugger.Initialize()
    atexit.register(lambda: lldb.SBDebugger.Terminate())
    unittest2.main()
