"""
Test that variable expressions of floating point types are evaluated correctly.
"""

import AbstractBase
import unittest2
import lldb

@unittest2.skip("rdar://problem/8488437 test/types/TestIntegerTypesExpr.py asserts")
class FloatTypesTestCase(AbstractBase.GenericTester):

    mydir = "types"

    def test_float_types_with_dsym(self):
        """Test that float-type variable expressions are evaluated correctly."""
        d = {'CXX_SOURCES': 'float.cpp'}
        self.buildDsym(dictionary=d)
        self.setTearDownCleanup(dictionary=d)
        self.float_type_expr()

    def test_float_type_with_dwarf(self):
        """Test that float-type variable expressions are evaluated correctly."""
        d = {'CXX_SOURCES': 'float.cpp'}
        self.buildDwarf(dictionary=d)
        self.setTearDownCleanup(dictionary=d)
        self.float_type_expr()

    def test_double_type_with_dsym(self):
        """Test that double-type variable expressions are evaluated correctly."""
        d = {'CXX_SOURCES': 'double.cpp'}
        self.buildDsym(dictionary=d)
        self.setTearDownCleanup(dictionary=d)
        self.double_type_expr()

    def test_double_type_with_dwarf(self):
        """Test that double-type variable expressions are evaluated correctly."""
        d = {'CXX_SOURCES': 'double.cpp'}
        self.buildDwarf(dictionary=d)
        self.setTearDownCleanup(dictionary=d)
        self.double_type_expr()

    def float_type_expr(self):
        """Test that float-type variable expressions are evaluated correctly."""
        self.generic_type_expr_tester(set(['float']))

    def double_type_expr(self):
        """Test that double-type variable expressions are evaluated correctly."""
        self.generic_type_expr_tester(set(['double']))


if __name__ == '__main__':
    import atexit
    lldb.SBDebugger.Initialize()
    atexit.register(lambda: lldb.SBDebugger.Terminate())
    unittest2.main()
