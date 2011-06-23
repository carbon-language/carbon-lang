"""
Test that variables of floating point types are displayed correctly.
"""

import AbstractBase
import unittest2
import lldb
import sys

class FloatTypesTestCase(AbstractBase.GenericTester):

    mydir = "types"

    @unittest2.skipUnless(sys.platform.startswith("darwin"), "requires Darwin")
    def test_float_type_with_dsym(self):
        """Test that float-type variables are displayed correctly."""
        d = {'CXX_SOURCES': 'float.cpp', 'EXE': self.exe_name}
        self.buildDsym(dictionary=d)
        self.setTearDownCleanup(dictionary=d)
        self.float_type(self.exe_name)

    def test_float_type_with_dwarf(self):
        """Test that float-type variables are displayed correctly."""
        d = {'CXX_SOURCES': 'float.cpp', 'EXE': self.exe_name}
        self.buildDwarf(dictionary=d)
        self.setTearDownCleanup(dictionary=d)
        self.float_type(self.exe_name)

    @unittest2.skipUnless(sys.platform.startswith("darwin"), "requires Darwin")
    def test_double_type_with_dsym(self):
        """Test that double-type variables are displayed correctly."""
        d = {'CXX_SOURCES': 'double.cpp', 'EXE': self.exe_name}
        self.buildDsym(dictionary=d)
        self.setTearDownCleanup(dictionary=d)
        self.double_type(self.exe_name)

    def test_double_type_with_dwarf(self):
        """Test that double-type variables are displayed correctly."""
        d = {'CXX_SOURCES': 'double.cpp', 'EXE': self.exe_name}
        self.buildDwarf(dictionary=d)
        self.setTearDownCleanup(dictionary=d)
        self.double_type(self.exe_name)

    def float_type(self, exe_name):
        """Test that float-type variables are displayed correctly."""
        self.generic_type_tester(exe_name, set(['float']))

    def double_type(self, exe_name):
        """Test that double-type variables are displayed correctly."""
        self.generic_type_tester(exe_name, set(['double']))


if __name__ == '__main__':
    import atexit
    lldb.SBDebugger.Initialize()
    atexit.register(lambda: lldb.SBDebugger.Terminate())
    unittest2.main()
