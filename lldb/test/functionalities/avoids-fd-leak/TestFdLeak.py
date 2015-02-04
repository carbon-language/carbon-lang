"""
Test whether a process started by lldb has no extra file descriptors open.
"""

import os
import unittest2
import lldb
from lldbtest import *
import lldbutil

class AvoidsFdLeakTestCase(TestBase):

    mydir = TestBase.compute_mydir(__file__)

    @expectedFailureWindows("The check for descriptor leakage needs to be implemented differently")
    def test_fd_leak (self):
        self.buildDefault()
        exe = os.path.join (os.getcwd(), "a.out")

        target = self.dbg.CreateTarget(exe)

        process = target.LaunchSimple (None, None, self.get_process_working_directory())
        self.assertTrue(process, PROCESS_IS_VALID)

        self.assertTrue(process.GetState() == lldb.eStateExited)
        self.assertTrue(process.GetExitStatus() == 0)

if __name__ == '__main__':
    import atexit
    lldb.SBDebugger.Initialize()
    atexit.register(lambda: lldb.SBDebugger.Terminate())
    unittest2.main()
