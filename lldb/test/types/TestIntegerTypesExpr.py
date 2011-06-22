"""
Test that variable expressions of integer basic types are evaluated correctly.
"""

import AbstractBase
import unittest2
import lldb
import sys

class IntegerTypesExprTestCase(AbstractBase.GenericTester):

    mydir = "types"

    @unittest2.skipUnless(sys.platform.startswith("darwin"), "requires Darwin")
    def test_char_type_with_dsym(self):
        """Test that char-type variable expressions are evaluated correctly."""
        d = {'CXX_SOURCES': 'char.cpp', 'EXE': 'char_type_dsym'}
        self.buildDsym(dictionary=d)
        self.setTearDownCleanup(dictionary=d)
        self.char_type_expr('char_type_dsym')

    def test_char_type_with_dwarf(self):
        """Test that char-type variable expressions are evaluated correctly."""
        d = {'CXX_SOURCES': 'char.cpp', 'EXE': 'char_type_dwarf'}
        self.buildDwarf(dictionary=d)
        self.setTearDownCleanup(dictionary=d)
        self.char_type_expr('char_type_dwarf')

    @unittest2.skipUnless(sys.platform.startswith("darwin"), "requires Darwin")
    def test_unsigned_char_type_with_dsym(self):
        """Test that 'unsigned_char'-type variable expressions are evaluated correctly."""
        d = {'CXX_SOURCES': 'unsigned_char.cpp', 'EXE': 'unsigned_char_type_dsym'}
        self.buildDsym(dictionary=d)
        self.setTearDownCleanup(dictionary=d)
        self.unsigned_char_type_expr('unsigned_char_type_dsym')

    def test_unsigned_char_type_with_dwarf(self):
        """Test that 'unsigned char'-type variable expressions are evaluated correctly."""
        d = {'CXX_SOURCES': 'unsigned_char.cpp', 'EXE': 'unsigned_char_type_dwarf'}
        self.buildDwarf(dictionary=d)
        self.setTearDownCleanup(dictionary=d)
        self.unsigned_char_type_expr('unsigned_char_type_dwarf')

    @unittest2.skipUnless(sys.platform.startswith("darwin"), "requires Darwin")
    def test_short_type_with_dsym(self):
        """Test that short-type variable expressions are evaluated correctly."""
        d = {'CXX_SOURCES': 'short.cpp', 'EXE': 'short_type_dsym'}
        self.buildDsym(dictionary=d)
        self.setTearDownCleanup(dictionary=d)
        self.short_type_expr('short_type_dsym')

    def test_short_type_with_dwarf(self):
        """Test that short-type variable expressions are evaluated correctly."""
        d = {'CXX_SOURCES': 'short.cpp', 'EXE': 'short_type_dwarf'}
        self.buildDwarf(dictionary=d)
        self.setTearDownCleanup(dictionary=d)
        self.short_type_expr('short_type_dwarf')

    @unittest2.skipUnless(sys.platform.startswith("darwin"), "requires Darwin")
    def test_unsigned_short_type_with_dsym(self):
        """Test that 'unsigned_short'-type variable expressions are evaluated correctly."""
        d = {'CXX_SOURCES': 'unsigned_short.cpp', 'EXE': 'unsigned_short_type_dsym'}
        self.buildDsym(dictionary=d)
        self.setTearDownCleanup(dictionary=d)
        self.unsigned_short_type_expr('unsigned_short_type_dsym')

    def test_unsigned_short_type_with_dwarf(self):
        """Test that 'unsigned short'-type variable expressions are evaluated correctly."""
        d = {'CXX_SOURCES': 'unsigned_short.cpp', 'EXE': 'unsigned_short_type_dwarf'}
        self.buildDwarf(dictionary=d)
        self.setTearDownCleanup(dictionary=d)
        self.unsigned_short_type_expr('unsigned_short_type_dwarf')

    @unittest2.skipUnless(sys.platform.startswith("darwin"), "requires Darwin")
    def test_int_type_with_dsym(self):
        """Test that int-type variable expressions are evaluated correctly."""
        d = {'CXX_SOURCES': 'int.cpp', 'EXE': 'int_type_dsym'}
        self.buildDsym(dictionary=d)
        self.setTearDownCleanup(dictionary=d)
        self.int_type_expr('int_type_dsym')

    def test_int_type_with_dwarf(self):
        """Test that int-type variable expressions are evaluated correctly."""
        d = {'CXX_SOURCES': 'int.cpp', 'EXE': 'int_type_dwarf'}
        self.buildDwarf(dictionary=d)
        self.setTearDownCleanup(dictionary=d)
        self.int_type_expr('int_type_dwarf')

    @unittest2.skipUnless(sys.platform.startswith("darwin"), "requires Darwin")
    def test_unsigned_int_type_with_dsym(self):
        """Test that 'unsigned_int'-type variable expressions are evaluated correctly."""
        d = {'CXX_SOURCES': 'unsigned_int.cpp', 'EXE': 'unsigned_int_type_dsym'}
        self.buildDsym(dictionary=d)
        self.setTearDownCleanup(dictionary=d)
        self.unsigned_int_type_expr('unsigned_int_type_dsym')

    def test_unsigned_int_type_with_dwarf(self):
        """Test that 'unsigned int'-type variable expressions are evaluated correctly."""
        d = {'CXX_SOURCES': 'unsigned_int.cpp', 'EXE': 'unsigned_int_type_dwarf'}
        self.buildDwarf(dictionary=d)
        self.setTearDownCleanup(dictionary=d)
        self.unsigned_int_type_expr('unsigned_int_type_dwarf')

    @unittest2.skipUnless(sys.platform.startswith("darwin"), "requires Darwin")
    def test_long_type_with_dsym(self):
        """Test that long-type variable expressions are evaluated correctly."""
        d = {'CXX_SOURCES': 'long.cpp', 'EXE': 'long_type_dsym'}
        self.buildDsym(dictionary=d)
        self.setTearDownCleanup(dictionary=d)
        self.long_type_expr('long_type_dsym')

    def test_long_type_with_dwarf(self):
        """Test that long-type variable expressions are evaluated correctly."""
        d = {'CXX_SOURCES': 'long.cpp', 'EXE': 'long_type_dwarf'}
        self.buildDwarf(dictionary=d)
        self.setTearDownCleanup(dictionary=d)
        self.long_type_expr('long_type_dwarf')

    @unittest2.skipUnless(sys.platform.startswith("darwin"), "requires Darwin")
    def test_unsigned_long_type_with_dsym(self):
        """Test that 'unsigned long'-type variable expressions are evaluated correctly."""
        d = {'CXX_SOURCES': 'unsigned_long.cpp', 'EXE': 'unsigned_long_type_dsym'}
        self.buildDsym(dictionary=d)
        self.setTearDownCleanup(dictionary=d)
        self.unsigned_long_type_expr('unsigned_long_type_dsym')

    def test_unsigned_long_type_with_dwarf(self):
        """Test that 'unsigned long'-type variable expressions are evaluated correctly."""
        d = {'CXX_SOURCES': 'unsigned_long.cpp', 'EXE': 'unsigned_long_type_dwarf'}
        self.buildDwarf(dictionary=d)
        self.setTearDownCleanup(dictionary=d)
        self.unsigned_long_type_expr('unsigned_long_type_dwarf')

    # rdar://problem/8482903
    # test suite failure for types dir -- "long long" and "unsigned long long"

    @unittest2.skipUnless(sys.platform.startswith("darwin"), "requires Darwin")
    def test_long_long_type_with_dsym(self):
        """Test that 'long long'-type variable expressions are evaluated correctly."""
        d = {'CXX_SOURCES': 'long_long.cpp', 'EXE': 'long_long_type_dsym'}
        self.buildDsym(dictionary=d)
        self.setTearDownCleanup(dictionary=d)
        self.long_long_type_expr('long_long_type_dsym')

    def test_long_long_type_with_dwarf(self):
        """Test that 'long long'-type variable expressions are evaluated correctly."""
        d = {'CXX_SOURCES': 'long_long.cpp', 'EXE': 'long_long_type_dwarf'}
        self.buildDwarf(dictionary=d)
        self.setTearDownCleanup(dictionary=d)
        self.long_long_type_expr('long_long_type_dwarf')

    @unittest2.skipUnless(sys.platform.startswith("darwin"), "requires Darwin")
    def test_unsigned_long_long_type_with_dsym(self):
        """Test that 'unsigned long long'-type variable expressions are evaluated correctly."""
        d = {'CXX_SOURCES': 'unsigned_long_long.cpp', 'EXE': 'unsigned_long_long_type_dsym'}
        self.buildDsym(dictionary=d)
        self.setTearDownCleanup(dictionary=d)
        self.unsigned_long_long_type_expr('unsigned_long_long_type_dsym')

    def test_unsigned_long_long_type_with_dwarf(self):
        """Test that 'unsigned long long'-type variable expressions are evaluated correctly."""
        d = {'CXX_SOURCES': 'unsigned_long_long.cpp', 'EXE': 'unsigned_long_long_type_dwarf'}
        self.buildDwarf(dictionary=d)
        self.setTearDownCleanup(dictionary=d)
        self.unsigned_long_long_type_expr('unsigned_long_long_type_dwarf')

    def char_type_expr(self, exe_name):
        """Test that char-type variable expressions are evaluated correctly."""
        self.generic_type_expr_tester(exe_name, set(['char']), quotedDisplay=True)

    def unsigned_char_type_expr(self, exe_name):
        """Test that 'unsigned char'-type variable expressions are evaluated correctly."""
        self.generic_type_expr_tester(exe_name, set(['unsigned', 'char']), quotedDisplay=True)

    def short_type_expr(self, exe_name):
        """Test that short-type variable expressions are evaluated correctly."""
        self.generic_type_expr_tester(exe_name, set(['short']))

    def unsigned_short_type_expr(self, exe_name):
        """Test that 'unsigned short'-type variable expressions are evaluated correctly."""
        self.generic_type_expr_tester(exe_name, set(['unsigned', 'short']))

    def int_type_expr(self, exe_name):
        """Test that int-type variable expressions are evaluated correctly."""
        self.generic_type_expr_tester(exe_name, set(['int']))

    def unsigned_int_type_expr(self, exe_name):
        """Test that 'unsigned int'-type variable expressions are evaluated correctly."""
        self.generic_type_expr_tester(exe_name, set(['unsigned', 'int']))

    def long_type_expr(self, exe_name):
        """Test that long-type variable expressions are evaluated correctly."""
        self.generic_type_expr_tester(exe_name, set(['long']))

    def unsigned_long_type_expr(self, exe_name):
        """Test that 'unsigned long'-type variable expressions are evaluated correctly."""
        self.generic_type_expr_tester(exe_name, set(['unsigned', 'long']))

    def long_long_type_expr(self, exe_name):
        """Test that long long-type variable expressions are evaluated correctly."""
        self.generic_type_expr_tester(exe_name, set(['long long']))

    def unsigned_long_long_type_expr(self, exe_name):
        """Test that 'unsigned long long'-type variable expressions are evaluated correctly."""
        self.generic_type_expr_tester(exe_name, set(['unsigned', 'long long']))


if __name__ == '__main__':
    import atexit
    lldb.SBDebugger.Initialize()
    atexit.register(lambda: lldb.SBDebugger.Terminate())
    unittest2.main()
