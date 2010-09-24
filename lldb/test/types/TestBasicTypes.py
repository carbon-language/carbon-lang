"""
Test that variables of basic types are displayed correctly.
"""

import os, time
import re
import unittest2
import lldb
from lldbtest import *

def Msg(var, val):
    return "'frame variable %s' matches the compiler's output: %s" % (var, val)

class BasicTypesTestCase(TestBase):

    mydir = "types"

    # This is the pattern by design to match the " var = 'value'" output from
    # printf() stmts (see basic_type.cpp).
    pattern = re.compile(" (\*?a[^=]*) = '([^=]*)'$")

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

    def test_unsigned_int_type_with_dsym(self):
        """Test that 'unsigned_int'-type variables are displayed correctly."""
        d = {'CXX_SOURCES': 'unsigned_int.cpp'}
        self.buildDsym(dictionary=d)
        self.setTearDownCleanup(dictionary=d)
        self.unsigned_int_type()

    def test_unsigned_int_type_with_dwarf(self):
        """Test that 'unsigned int'-type variables are displayed correctly."""
        d = {'CXX_SOURCES': 'unsigned_int.cpp'}
        self.buildDwarf(dictionary=d)
        self.setTearDownCleanup(dictionary=d)
        self.unsigned_int_type()

    def test_long_type_with_dsym(self):
        """Test that long-type variables are displayed correctly."""
        d = {'CXX_SOURCES': 'long.cpp'}
        self.buildDsym(dictionary=d)
        self.setTearDownCleanup(dictionary=d)
        self.long_type()

    def test_long_type_with_dwarf(self):
        """Test that long-type variables are displayed correctly."""
        d = {'CXX_SOURCES': 'long.cpp'}
        self.buildDwarf(dictionary=d)
        self.setTearDownCleanup(dictionary=d)
        self.long_type()

    def test_unsigned_long_type_with_dsym(self):
        """Test that 'unsigned long'-type variables are displayed correctly."""
        d = {'CXX_SOURCES': 'unsigned_long.cpp'}
        self.buildDsym(dictionary=d)
        self.setTearDownCleanup(dictionary=d)
        self.unsigned_long_type()

    def test_unsigned_long_type_with_dwarf(self):
        """Test that 'unsigned long'-type variables are displayed correctly."""
        d = {'CXX_SOURCES': 'unsigned_long.cpp'}
        self.buildDwarf(dictionary=d)
        self.setTearDownCleanup(dictionary=d)
        self.unsigned_long_type()

    def int_type(self):
        """Test that int-type variables are displayed correctly."""
        self.generic_type_tester("int")

    def unsigned_int_type(self):
        """Test that 'unsigned int'-type variables are displayed correctly."""
        self.generic_type_tester("unsigned int")

    def long_type(self):
        """Test that long-type variables are displayed correctly."""
        self.generic_type_tester("long")

    def unsigned_long_type(self):
        """Test that 'unsigned long'-type variables are displayed correctly."""
        self.generic_type_tester("unsigned long")

    def generic_type_tester(self, type):
        """Test that variables with basic types are displayed correctly."""

        # First, capture the golden output emitted by the oracle, i.e., the
        # series of printf statements.
        go = system("./a.out")
        # This golden list contains a list of (variable, value) pairs extracted
        # from the golden output.
        gl = []

        # Scan the golden output line by line, looking for the pattern:
        #
        #     variable = 'value'
        #
        # Filter out the following lines, for the time being:
        #
        #     'a_ref = ...'
        #     'a_class_ref.m_a = ...'
        #     'a_class_ref.m_b = ...'
        #     'a_struct_ref.a = ...'
        #     'a_struct_ref.b = ...'
        #     'a_union_zero_ref.a = ...'
        #     'a_union_nonzero_ref.u.a = ...'
        #
        # rdar://problem/8471016 frame variable a_ref should display the referenced value as well
        # rdar://problem/8470987 frame variable a_class_ref.m_a does not work
        notnow = set(['a_ref',
                      'a_class_ref.m_a', 'a_class_ref.m_b',
                      'a_struct_ref.a', 'a_struct_ref.b',
                      'a_union_zero_ref.a', 'a_union_nonzero_ref.u.a'])
        for line in go.split(os.linesep):
            match = self.pattern.search(line)
            if match:
                var, val = match.group(1), match.group(2)
                if var in notnow:
                    continue
                gl.append((var, val))
        #print "golden list:", gl

        # Bring the program to the point where we can issue a series of
        # 'frame variable' command.
        self.runCmd("file a.out", CURRENT_EXECUTABLE_SET)
        self.runCmd("breakpoint set --name Puts")
        self.runCmd("run", RUN_SUCCEEDED)
        self.runCmd("thread step-out", STEP_OUT_SUCCEEDED)

        # Now iterate through the golden list, comparing against the output from
        # 'frame variable var'.
        for var, val in gl:
            self.runCmd("frame variable %s" % var)
            output = self.res.GetOutput()
            
            # Extract the display type and canonicalize its atoms into a set.
            # Same for the input type string.
            dt = re.match("^\((.*)\)", output).group(1)
            ds = set(dt.split())
            ts = set(type.split())

            # The display type set must be a superset of the input type set.
            if not ds.issuperset(ts):
                self.fail("The display type: %s must match the input type: %s" %
                          (dt, type))

            # The (var, val) pair must match, too.
            self.expect(output, Msg(var, val), exe=False,
                substrs = [" %s = %s" % (var, val)])


if __name__ == '__main__':
    import atexit
    lldb.SBDebugger.Initialize()
    atexit.register(lambda: lldb.SBDebugger.Terminate())
    unittest2.main()
