"""
Test that variables of floating point types are displayed correctly.
"""

import AbstractBase
import unittest2
import lldb

class FloatTypesTestCase(AbstractBase.GenericTester):

    mydir = "types"

    def test_float_types_with_dsym(self):
        """Test that float-type variables are displayed correctly."""
        d = {'CXX_SOURCES': 'float.cpp'}
        self.buildDsym(dictionary=d)
        self.setTearDownCleanup(dictionary=d)
        self.float_type()

    def test_float_type_with_dwarf(self):
        """Test that float-type variables are displayed correctly."""
        d = {'CXX_SOURCES': 'float.cpp'}
        self.buildDwarf(dictionary=d)
        self.setTearDownCleanup(dictionary=d)
        self.float_type()

    def test_double_type_with_dsym(self):
        """Test that double-type variables are displayed correctly."""
        d = {'CXX_SOURCES': 'double.cpp'}
        self.buildDsym(dictionary=d)
        self.setTearDownCleanup(dictionary=d)
        self.double_type()

    def test_double_type_with_dwarf(self):
        """Test that double-type variables are displayed correctly."""
        d = {'CXX_SOURCES': 'double.cpp'}
        self.buildDwarf(dictionary=d)
        self.setTearDownCleanup(dictionary=d)
        self.double_type()

    def float_type(self):
        """Test that float-type variables are displayed correctly."""
        self.generic_type_tester(set(['float']))

    def double_type(self):
        """Test that double-type variables are displayed correctly."""
        self.generic_type_tester(set(['double']))


if __name__ == '__main__':
    import atexit
    lldb.SBDebugger.Initialize()
    atexit.register(lambda: lldb.SBDebugger.Terminate())
    unittest2.main()
