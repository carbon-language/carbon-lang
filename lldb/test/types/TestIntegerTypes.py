"""
Test that variables of integer basic types are displayed correctly.
"""

import AbstractBase
import unittest2
import lldb
import sys

class IntegerTypesTestCase(AbstractBase.GenericTester):

    mydir = "types"

    @unittest2.skipUnless(sys.platform.startswith("darwin"), "requires Darwin")
    def test_char_type_with_dsym(self):
        """Test that char-type variables are displayed correctly."""
        d = {'CXX_SOURCES': 'char.cpp', 'EXE': self.exe_name}
        self.buildDsym(dictionary=d)
        self.setTearDownCleanup(dictionary=d)
        self.char_type(self.exe_name)

    def test_char_type_with_dwarf(self):
        """Test that char-type variables are displayed correctly."""
        d = {'CXX_SOURCES': 'char.cpp', 'EXE': self.exe_name}
        self.buildDwarf(dictionary=d)
        self.setTearDownCleanup(dictionary=d)
        self.char_type(self.exe_name)

    @unittest2.skipUnless(sys.platform.startswith("darwin"), "requires Darwin")
    def test_unsigned_char_type_with_dsym(self):
        """Test that 'unsigned_char'-type variables are displayed correctly."""
        d = {'CXX_SOURCES': 'unsigned_char.cpp', 'EXE': self.exe_name}
        self.buildDsym(dictionary=d)
        self.setTearDownCleanup(dictionary=d)
        self.unsigned_char_type(self.exe_name)

    def test_unsigned_char_type_with_dwarf(self):
        """Test that 'unsigned char'-type variables are displayed correctly."""
        d = {'CXX_SOURCES': 'unsigned_char.cpp', 'EXE': self.exe_name}
        self.buildDwarf(dictionary=d)
        self.setTearDownCleanup(dictionary=d)
        self.unsigned_char_type(self.exe_name)

    @unittest2.skipUnless(sys.platform.startswith("darwin"), "requires Darwin")
    def test_short_type_with_dsym(self):
        """Test that short-type variables are displayed correctly."""
        d = {'CXX_SOURCES': 'short.cpp', 'EXE': self.exe_name}
        self.buildDsym(dictionary=d)
        self.setTearDownCleanup(dictionary=d)
        self.short_type(self.exe_name)

    def test_short_type_with_dwarf(self):
        """Test that short-type variables are displayed correctly."""
        d = {'CXX_SOURCES': 'short.cpp', 'EXE': self.exe_name}
        self.buildDwarf(dictionary=d)
        self.setTearDownCleanup(dictionary=d)
        self.short_type(self.exe_name)

    @unittest2.skipUnless(sys.platform.startswith("darwin"), "requires Darwin")
    def test_unsigned_short_type_with_dsym(self):
        """Test that 'unsigned_short'-type variables are displayed correctly."""
        d = {'CXX_SOURCES': 'unsigned_short.cpp', 'EXE': self.exe_name}
        self.buildDsym(dictionary=d)
        self.setTearDownCleanup(dictionary=d)
        self.unsigned_short_type(self.exe_name)

    def test_unsigned_short_type_with_dwarf(self):
        """Test that 'unsigned short'-type variables are displayed correctly."""
        d = {'CXX_SOURCES': 'unsigned_short.cpp', 'EXE': self.exe_name}
        self.buildDwarf(dictionary=d)
        self.setTearDownCleanup(dictionary=d)
        self.unsigned_short_type(self.exe_name)

    @unittest2.skipUnless(sys.platform.startswith("darwin"), "requires Darwin")
    def test_int_type_with_dsym(self):
        """Test that int-type variables are displayed correctly."""
        d = {'CXX_SOURCES': 'int.cpp', 'EXE': self.exe_name}
        self.buildDsym(dictionary=d)
        self.setTearDownCleanup(dictionary=d)
        self.int_type(self.exe_name)

    def test_int_type_with_dwarf(self):
        """Test that int-type variables are displayed correctly."""
        d = {'CXX_SOURCES': 'int.cpp', 'EXE': self.exe_name}
        self.buildDwarf(dictionary=d)
        self.setTearDownCleanup(dictionary=d)
        self.int_type(self.exe_name)

    @unittest2.skipUnless(sys.platform.startswith("darwin"), "requires Darwin")
    def test_unsigned_int_type_with_dsym(self):
        """Test that 'unsigned_int'-type variables are displayed correctly."""
        d = {'CXX_SOURCES': 'unsigned_int.cpp', 'EXE': self.exe_name}
        self.buildDsym(dictionary=d)
        self.setTearDownCleanup(dictionary=d)
        self.unsigned_int_type(self.exe_name)

    def test_unsigned_int_type_with_dwarf(self):
        """Test that 'unsigned int'-type variables are displayed correctly."""
        d = {'CXX_SOURCES': 'unsigned_int.cpp', 'EXE': self.exe_name}
        self.buildDwarf(dictionary=d)
        self.setTearDownCleanup(dictionary=d)
        self.unsigned_int_type(self.exe_name)

    @unittest2.skipUnless(sys.platform.startswith("darwin"), "requires Darwin")
    def test_long_type_with_dsym(self):
        """Test that long-type variables are displayed correctly."""
        d = {'CXX_SOURCES': 'long.cpp', 'EXE': self.exe_name}
        self.buildDsym(dictionary=d)
        self.setTearDownCleanup(dictionary=d)
        self.long_type(self.exe_name)

    def test_long_type_with_dwarf(self):
        """Test that long-type variables are displayed correctly."""
        d = {'CXX_SOURCES': 'long.cpp', 'EXE': self.exe_name}
        self.buildDwarf(dictionary=d)
        self.setTearDownCleanup(dictionary=d)
        self.long_type(self.exe_name)

    @unittest2.skipUnless(sys.platform.startswith("darwin"), "requires Darwin")
    def test_unsigned_long_type_with_dsym(self):
        """Test that 'unsigned long'-type variables are displayed correctly."""
        d = {'CXX_SOURCES': 'unsigned_long.cpp', 'EXE': self.exe_name}
        self.buildDsym(dictionary=d)
        self.setTearDownCleanup(dictionary=d)
        self.unsigned_long_type(self.exe_name)

    def test_unsigned_long_type_with_dwarf(self):
        """Test that 'unsigned long'-type variables are displayed correctly."""
        d = {'CXX_SOURCES': 'unsigned_long.cpp', 'EXE': self.exe_name}
        self.buildDwarf(dictionary=d)
        self.setTearDownCleanup(dictionary=d)
        self.unsigned_long_type(self.exe_name)

    # rdar://problem/8482903
    # test suite failure for types dir -- "long long" and "unsigned long long"

    @unittest2.skipUnless(sys.platform.startswith("darwin"), "requires Darwin")
    def test_long_long_type_with_dsym(self):
        """Test that 'long long'-type variables are displayed correctly."""
        d = {'CXX_SOURCES': 'long_long.cpp', 'EXE': self.exe_name}
        self.buildDsym(dictionary=d)
        self.setTearDownCleanup(dictionary=d)
        self.long_long_type(self.exe_name)

    def test_long_long_type_with_dwarf(self):
        """Test that 'long long'-type variables are displayed correctly."""
        d = {'CXX_SOURCES': 'long_long.cpp', 'EXE': self.exe_name}
        self.buildDwarf(dictionary=d)
        self.setTearDownCleanup(dictionary=d)
        self.long_long_type(self.exe_name)

    @unittest2.skipUnless(sys.platform.startswith("darwin"), "requires Darwin")
    def test_unsigned_long_long_type_with_dsym(self):
        """Test that 'unsigned long long'-type variables are displayed correctly."""
        d = {'CXX_SOURCES': 'unsigned_long_long.cpp', 'EXE': self.exe_name}
        self.buildDsym(dictionary=d)
        self.setTearDownCleanup(dictionary=d)
        self.unsigned_long_long_type(self.exe_name)

    def test_unsigned_long_long_type_with_dwarf(self):
        """Test that 'unsigned long long'-type variables are displayed correctly."""
        d = {'CXX_SOURCES': 'unsigned_long_long.cpp', 'EXE': self.exe_name}
        self.buildDwarf(dictionary=d)
        self.setTearDownCleanup(dictionary=d)
        self.unsigned_long_long_type(self.exe_name)

    def char_type(self, exe_name):
        """Test that char-type variables are displayed correctly."""
        self.generic_type_tester(exe_name, set(['char']), quotedDisplay=True)

    def unsigned_char_type(self, exe_name):
        """Test that 'unsigned char'-type variables are displayed correctly."""
        self.generic_type_tester(exe_name, set(['unsigned', 'char']), quotedDisplay=True)

    def short_type(self, exe_name):
        """Test that short-type variables are displayed correctly."""
        self.generic_type_tester(exe_name, set(['short']))

    def unsigned_short_type(self, exe_name):
        """Test that 'unsigned short'-type variables are displayed correctly."""
        self.generic_type_tester(exe_name, set(['unsigned', 'short']))

    def int_type(self, exe_name):
        """Test that int-type variables are displayed correctly."""
        self.generic_type_tester(exe_name, set(['int']))

    def unsigned_int_type(self, exe_name):
        """Test that 'unsigned int'-type variables are displayed correctly."""
        self.generic_type_tester(exe_name, set(['unsigned', 'int']))

    def long_type(self, exe_name):
        """Test that long-type variables are displayed correctly."""
        self.generic_type_tester(exe_name, set(['long']))

    def unsigned_long_type(self, exe_name):
        """Test that 'unsigned long'-type variables are displayed correctly."""
        self.generic_type_tester(exe_name, set(['unsigned', 'long']))

    def long_long_type(self, exe_name):
        """Test that long long-type variables are displayed correctly."""
        self.generic_type_tester(exe_name, set(['long long']))

    def unsigned_long_long_type(self, exe_name):
        """Test that 'unsigned long long'-type variables are displayed correctly."""
        self.generic_type_tester(exe_name, set(['unsigned', 'long long']))


if __name__ == '__main__':
    import atexit
    lldb.SBDebugger.Initialize()
    atexit.register(lambda: lldb.SBDebugger.Terminate())
    unittest2.main()
