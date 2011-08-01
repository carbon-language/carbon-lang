"""Test evaluating expressions repeatedly comparing lldb against gdb."""

import os
import unittest2
import lldb
import pexpect
from lldbbench import *

class RepeatedExprsCase(BenchBase):

    mydir = os.path.join("benchmarks", "example")

    @benchmarks_test
    def test_with_lldb(self):
        """Test repeated expressions with lldb."""
        self.buildDefault()
        self.run_lldb_repeated_exprs()

    @benchmarks_test
    def test_with_gdb(self):
        """Test repeated expressions with gdb."""
        self.buildDefault()
        self.run_gdb_repeated_exprs()

    def run_lldb_repeated_exprs(self):
        print "running "+self.testMethodName
        print "benchmarks result for "+self.testMethodName

    def run_gdb_repeated_exprs(self):
        print "running "+self.testMethodName
        print "benchmarks result for "+self.testMethodName

if __name__ == '__main__':
    import atexit
    lldb.SBDebugger.Initialize()
    atexit.register(lambda: lldb.SBDebugger.Terminate())
    unittest2.main()
