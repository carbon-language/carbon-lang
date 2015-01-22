"""Test the lldb public C++ api when doing multiple debug sessions simultaneously."""

import os, re, StringIO
import unittest2
from lldbtest import *
import lldbutil
import lldb
import subprocess

class TestMultipleSimultaneousDebuggers(TestBase):

    mydir = TestBase.compute_mydir(__file__)

    def setUp(self):
        TestBase.setUp(self)
        self.lib_dir = os.environ["LLDB_LIB_DIR"]

    @skipIfi386
    @skipIfNoSBHeaders
    @expectedFailureDarwin("llvm.org/pr20282") # intermittent
    @expectedFailureFreeBSD("llvm.org/pr20282")
    @expectedFailureLinux("llvm.org/pr20282")
    def test_multiple_debuggers(self):
        env = {self.dylibPath : self.getLLDBLibraryEnvVal()}

        self.driver_exe = os.path.join(os.getcwd(), "multi-process-driver")
        self.buildDriver('multi-process-driver.cpp', self.driver_exe)
        self.addTearDownHook(lambda: os.remove(self.driver_exe))
        self.signBinary(self.driver_exe)

        self.inferior_exe = os.path.join(os.getcwd(), "testprog")
        self.buildDriver('testprog.cpp', self.inferior_exe)
        self.addTearDownHook(lambda: os.remove(self.inferior_exe))

# check_call will raise a CalledProcessError if multi-process-driver doesn't return
# exit code 0 to indicate success.  We can let this exception go - the test harness
# will recognize it as a test failure.

        if self.TraceOn():
            print "Running test %s" % self.driver_exe
            check_call([self.driver_exe, self.inferior_exe], env=env)
        else:
            with open(os.devnull, 'w') as fnull:
                check_call([self.driver_exe, self.inferior_exe], env=env, stdout=fnull, stderr=fnull)

if __name__ == '__main__':
    import atexit
    lldb.SBDebugger.Initialize()
    atexit.register(lambda: lldb.SBDebugger.Terminate())
    unittest2.main()
