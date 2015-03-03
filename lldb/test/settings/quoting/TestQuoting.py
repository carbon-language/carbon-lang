"""
Test quoting of arguments to lldb commands
"""

import os, time, re
import unittest2
import lldb
from lldbtest import *

class SettingsCommandTestCase(TestBase):

    mydir = TestBase.compute_mydir(__file__)

    @classmethod
    def classCleanup(cls):
        """Cleanup the test byproducts."""
        cls.RemoveTempFile("stdout.txt")

    def test_no_quote(self):
        self.do_test_args("a b c", "a\0b\0c\0")

    def test_single_quote(self):
        self.do_test_args("'a b c'", "a b c\0")

    def test_double_quote(self):
        self.do_test_args('"a b c"', "a b c\0")

    def test_double_quote(self):
        self.do_test_args('"a b c"', 'a b c\0')

    def test_single_quote_escape(self):
        self.do_test_args("'a b\\' c", "a b\\\0c\0")

    def test_double_quote_escape(self):
        self.do_test_args('"a b\\" c"', 'a b" c\0')

    def test_double_quote_escape(self):
        self.do_test_args('"a b\\" c"', 'a b" c\0')

    def test_double_quote_escape2(self):
        self.do_test_args('"a b\\\\" c', 'a b\\\0c\0')

    def test_double_quote_escape2(self):
        self.do_test_args('"a b\\\\" c', 'a b\\\0c\0')

    def test_single_in_double(self):
        self.do_test_args('"a\'b"', "a'b\0")

    def test_double_in_single(self):
        self.do_test_args("'a\"b'", 'a"b\0')

    def test_combined(self):
        self.do_test_args('"a b"c\'d e\'', 'a bcd e\0')

    def test_bare_single(self):
        self.do_test_args("a\\'b", "a'b\0")

    def test_bare_double(self):
        self.do_test_args('a\\"b', 'a"b\0')

    def do_test_args(self, args_in, args_out):
        """Test argument parsing. Run the program with args_in. The program dumps its arguments
        to stdout. Compare the stdout with args_out."""
        self.buildDefault()

        exe = os.path.join(os.getcwd(), "a.out")
        self.runCmd("file " + exe, CURRENT_EXECUTABLE_SET)

        self.runCmd("process launch -o stdout.txt -- " + args_in)

        if lldb.remote_platform:
            src_file_spec = lldb.SBFileSpec('stdout.txt', False)
            dst_file_spec = lldb.SBFileSpec('stdout.txt', True)
            lldb.remote_platform.Get(src_file_spec, dst_file_spec)

        with open('stdout.txt', 'r') as f:
            output = f.read()

        self.RemoveTempFile("stdout.txt")

        self.assertEqual(output, args_out)

if __name__ == '__main__':
    import atexit
    lldb.SBDebugger.Initialize()
    atexit.register(lambda: lldb.SBDebugger.Terminate())
    unittest2.main()
