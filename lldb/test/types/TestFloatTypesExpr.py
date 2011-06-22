"""
Test that variable expressions of floating point types are evaluated correctly.
"""

import AbstractBase
import unittest2
import lldb
import sys

class FloatTypesExprTestCase(AbstractBase.GenericTester):

    mydir = "types"

    # rdar://problem/8493023
    # test/types failures for Test*TypesExpr.py: element offset computed wrong and sign error?

    @unittest2.skipUnless(sys.platform.startswith("darwin"), "requires Darwin")
    def test_float_type_with_dsym(self):
        """Test that float-type variable expressions are evaluated correctly."""
        d = {'CXX_SOURCES': 'float.cpp', 'EXE': 'float_type_dsym'}
        self.buildDsym(dictionary=d)
        self.setTearDownCleanup(dictionary=d)
        self.float_type_expr('float_type_dsym')

    def test_float_type_with_dwarf(self):
        """Test that float-type variable expressions are evaluated correctly."""
        d = {'CXX_SOURCES': 'float.cpp', 'EXE': 'float_type_dwarf'}
        self.buildDwarf(dictionary=d)
        self.setTearDownCleanup(dictionary=d)
        self.float_type_expr('float_type_dwarf')

    @unittest2.skipUnless(sys.platform.startswith("darwin"), "requires Darwin")
    def test_double_type_with_dsym(self):
        """Test that double-type variable expressions are evaluated correctly."""
        d = {'CXX_SOURCES': 'double.cpp', 'EXE': 'double_type_dsym'}
        self.buildDsym(dictionary=d)
        self.setTearDownCleanup(dictionary=d)
        self.double_type_expr('double_type_dsym')

    def test_double_type_with_dwarf(self):
        """Test that double-type variable expressions are evaluated correctly."""
        d = {'CXX_SOURCES': 'double.cpp', 'EXE': 'double_type_dwarf'}
        self.buildDwarf(dictionary=d)
        self.setTearDownCleanup(dictionary=d)
        self.double_type_expr('double_type_dwarf')

    def float_type_expr(self, exe_name):
        """Test that float-type variable expressions are evaluated correctly."""
        self.generic_type_expr_tester(exe_name, set(['float']))

    def double_type_expr(self, exe_name):
        """Test that double-type variable expressions are evaluated correctly."""
        self.generic_type_expr_tester(exe_name, set(['double']))


if __name__ == '__main__':
    import atexit
    lldb.SBDebugger.Initialize()
    atexit.register(lambda: lldb.SBDebugger.Terminate())
    unittest2.main()
