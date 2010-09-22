"""
Test that variables of basic types are displayed correctly.
"""

import os, time
import unittest2
import lldb
from lldbtest import *

class BasicTypesTestCase(TestBase):

    mydir = "types"

    def test_int_type_with_dsym(self):
        """Test that int-type variables are displayed correctly."""
        d = {'CXX_SOURCES': 'int.cpp'}
        self.buildDsym(dictionary=d)
        self.setTearDownCleanup(dictionary=d)
        self.int_type()

    def test_int_type_with_dwarf(self):
        """Test that int-type variables are displayed correctly."""
        d = {'CXX_SOURCES': 'int.cpp'}
        self.buildDwarf(dictionary=d)
        self.setTearDownCleanup(dictionary=d)
        self.int_type()

    def int_type(self):
        """Test that int-type variables are displayed correctly."""
        self.runCmd("file a.out", CURRENT_EXECUTABLE_SET)

        self.runCmd("breakpoint set --name Puts")

        self.runCmd("run", RUN_SUCCEEDED)

        self.runCmd("thread step-out", STEP_OUT_SUCCEEDED)

        self.runCmd("frame variable a")


if __name__ == '__main__':
    import atexit
    lldb.SBDebugger.Initialize()
    atexit.register(lambda: lldb.SBDebugger.Terminate())
    unittest2.main()
