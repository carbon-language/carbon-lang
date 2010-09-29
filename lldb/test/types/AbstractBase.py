"""
Abstract base class of basic types provides a generic type tester method.
"""

import os, time
import re
import lldb
from lldbtest import *

def Msg(var, val):
    return "'frame variable %s' matches the output (from compiled code): %s" % (var, val)

class GenericTester(TestBase):

    # This is the pattern by design to match the " var = 'value'" output from
    # printf() stmts (see basic_type.cpp).
    pattern = re.compile(" (\*?a[^=]*) = '([^=]*)'$")

    def generic_type_tester(self, atoms, quotedDisplay=False):
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

        #self.runCmd("frame variable")

        # Now iterate through the golden list, comparing against the output from
        # 'frame variable var'.
        for var, val in gl:
            self.runCmd("frame variable %s" % var)
            output = self.res.GetOutput()
            
            # The input type is in a canonical form as a set named atoms.
            # The display type string must conatin each and every element.
            #
            # Example:
            #     runCmd: frame variable a_array_bounded[0]
            #     output: (char) a_array_bounded[0] = 'a'
            #
            try:
                dt = re.match("^\((.*)\)", output).group(1)
            except:
                self.fail("Data type from expression parser is parsed correctly")

            # Expect the display type string to contain each and every atoms.
            self.expect(dt,
                        "Display type: '%s' must contain the type atoms: '%s'" %
                        (dt, atoms),
                        exe=False,
                substrs = list(atoms))

            # The (var, val) pair must match, too.
            nv = (" %s = '%s'" if quotedDisplay else " %s = %s") % (var, val)
            self.expect(output, Msg(var, val), exe=False,
                substrs = [nv])

    def generic_type_expr_tester(self, atoms, quotedDisplay=False):
        """Test that variable expressions with basic types are evaluated correctly."""

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
        for line in go.split(os.linesep):
            match = self.pattern.search(line)
            if match:
                var, val = match.group(1), match.group(2)
                gl.append((var, val))
        #print "golden list:", gl

        # Bring the program to the point where we can issue a series of
        # 'frame variable' command.
        self.runCmd("file a.out", CURRENT_EXECUTABLE_SET)
        self.runCmd("breakpoint set --name Puts")
        self.runCmd("run", RUN_SUCCEEDED)
        self.runCmd("thread step-out", STEP_OUT_SUCCEEDED)

        #self.runCmd("frame variable")

        # Now iterate through the golden list, comparing against the output from
        # 'frame variable var'.
        for var, val in gl:
            self.runCmd("expr %s" % var)
            output = self.res.GetOutput()
            
            # The input type is in a canonical form as a set named atoms.
            # The display type string must conatin each and every element.
            #
            # Example:
            #     runCmd: expr a
            #     output: $0 = (double) 1100.12
            #
            try:
                dt = re.match("^\$[0-9]+ = \((.*)\)", output).group(1)
            except:
                self.fail("Data type from expression parser is parsed correctly")

            # Expect the display type string to contain each and every atoms.
            self.expect(dt,
                        "Display type: '%s' must contain the type atoms: '%s'" %
                        (dt, atoms),
                        exe=False,
                substrs = list(atoms))

            # The val part must match, too.
            valPart = ("'%s'" if quotedDisplay else "%s") % val
            self.expect(output, Msg(var, val), exe=False,
                substrs = [valPart])
