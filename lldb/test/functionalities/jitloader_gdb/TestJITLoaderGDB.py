"""Test for the JITLoaderGDB interface"""

import os
import unittest2
import lldb
from lldbtest import *
import lldbutil
import re


class JITLoaderGDBTestCase(TestBase):

    mydir = TestBase.compute_mydir(__file__)

    @skipTestIfFn(lambda x: True, "llvm.org/pr24702", "Skipped because the test crashes the test runner")
    @unittest2.expectedFailure("llvm.org/pr24702")
    @dsym_test
    def test_bogus_values_with_dsym(self):
        self.buildDsym()
        self.bogus_values_test()

    @skipTestIfFn(lambda x: True, "llvm.org/pr24702", "Skipped because the test crashes the test runner")
    @unittest2.expectedFailure("llvm.org/pr24702")
    @dwarf_test
    def test_bogus_values_with_dwarf(self):
        self.buildDwarf()
        self.bogus_values_test()

    def bogus_values_test(self):
        """Test that we handle inferior misusing the GDB JIT interface"""
        exe = os.path.join(os.getcwd(), "a.out")

        # Create a target by the debugger.
        target = self.dbg.CreateTarget(exe)
        self.assertTrue(target, VALID_TARGET)

        # launch the process, do not stop at entry point.
        process = target.LaunchSimple(None, None, self.get_process_working_directory())
        self.assertTrue(process, PROCESS_IS_VALID)

        # The inferior will now pass bogus values over the interface. Make sure we don't crash.

        self.assertEqual(process.GetState(), lldb.eStateExited)
        self.assertEqual(process.GetExitStatus(), 0)

if __name__ == '__main__':
    import atexit
    lldb.SBDebugger.Initialize()
    atexit.register(lambda: lldb.SBDebugger.Terminate())
    unittest2.main()
