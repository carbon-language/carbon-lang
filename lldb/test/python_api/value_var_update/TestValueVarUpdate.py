"""Test SBValue::GetValueDidChange"""

import os, sys, time
import unittest2
import lldb
import time
from lldbtest import *

class HelloWorldTestCase(TestBase):

    mydir = TestBase.compute_mydir(__file__)

    @unittest2.skipUnless(sys.platform.startswith("darwin"), "requires Darwin")
    @python_api_test
    @dsym_test
    def test_with_dsym_and_process_launch_api(self):
        """Test SBValue::GetValueDidChange"""
        self.buildDsym(dictionary=self.d)
        self.setTearDownCleanup(dictionary=self.d)
        self.do_test()

    @expectedFailureFreeBSD("llvm.org/pr21620 GetValueDidChange() wrong")
    @python_api_test
    @dwarf_test
    def test_with_dwarf_and_process_launch_api(self):
        """Test SBValue::GetValueDidChange"""
        self.buildDwarf(dictionary=self.d)
        self.setTearDownCleanup(dictionary=self.d)
        self.do_test()

    def setUp(self):
        # Call super's setUp().
        TestBase.setUp(self)
        # Get the full path to our executable to be attached/debugged.
        self.exe = os.path.join(os.getcwd(), self.testMethodName)
        self.d = {'EXE': self.testMethodName}

    def do_test(self):
        """Create target, breakpoint, launch a process, and then kill it."""

        target = self.dbg.CreateTarget(self.exe)

        breakpoint = target.BreakpointCreateBySourceRegex("break here", lldb.SBFileSpec("main.c"))

        self.runCmd("run", RUN_SUCCEEDED)
        
        # The stop reason of the thread should be breakpoint.
        self.expect("thread list", STOPPED_DUE_TO_BREAKPOINT,
            substrs = ['stopped',
                       'stop reason = breakpoint'])

        i = self.frame().FindVariable("i")
        i_val = i.GetValueAsUnsigned(0)
        
        if self.TraceOn(): self.runCmd("frame variable")
        
        self.runCmd("continue")

        if self.TraceOn(): self.runCmd("frame variable")
        
        self.assertTrue(i_val != i.GetValueAsUnsigned(0), "GetValue() is saying a lie")
        self.assertTrue(i.GetValueDidChange(), "GetValueDidChange() is saying a lie")


if __name__ == '__main__':
    import atexit
    lldb.SBDebugger.Initialize()
    atexit.register(lambda: lldb.SBDebugger.Terminate())
    unittest2.main()
