"""
Abstract base class of basic types provides a generic type tester method.
"""

import os, time
import re
import lldb
from lldbtest import *

def Msg(var, val, using_frame_variable):
    return "'%s %s' matches the output (from compiled code): %s" % (
        'frame variable -T' if using_frame_variable else 'expression' ,var, val)

class GenericTester(TestBase):

    # This is the pattern by design to match the " var = 'value'" output from
    # printf() stmts (see basic_type.cpp).
    pattern = re.compile(" (\*?a[^=]*) = '([^=]*)'$")

    # Assert message.
    DATA_TYPE_GROKKED = "Data type from expr parser output is parsed correctly"

    def generic_type_tester(self, atoms, quotedDisplay=False):
        """Test that variables with basic types are displayed correctly."""

        # First, capture the golden output emitted by the oracle, i.e., the
        # series of printf statements.
        go = system("./a.out", sender=self)[0]
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
        # 'frame variable -T' command.
        self.runCmd("file a.out", CURRENT_EXECUTABLE_SET)
        puts_line = line_number ("basic_type.cpp", "// Here is the line we will break on to check variables")
        self.expect("breakpoint set -f basic_type.cpp -l %d" % puts_line,
                    BREAKPOINT_CREATED,
            startstr = "Breakpoint created: 1: file ='basic_type.cpp', line = %d, locations = 1" %
                        puts_line)

        self.runCmd("run", RUN_SUCCEEDED)
        self.expect("process status", STOPPED_DUE_TO_BREAKPOINT,
            substrs = [" at basic_type.cpp:%d" % puts_line,
                       "stop reason = breakpoint"])

        #self.runCmd("frame variable -T")

        # Now iterate through the golden list, comparing against the output from
        # 'frame variable -T var'.
        for var, val in gl:
            self.runCmd("frame variable -T %s" % var)
            output = self.res.GetOutput()
            
            # The input type is in a canonical form as a set of named atoms.
            # The display type string must conatin each and every element.
            #
            # Example:
            #     runCmd: frame variable -T a_array_bounded[0]
            #     output: (char) a_array_bounded[0] = 'a'
            #
            try:
                dt = re.match("^\((.*)\)", output).group(1)
            except:
                self.fail(self.DATA_TYPE_GROKKED)

            # Expect the display type string to contain each and every atoms.
            self.expect(dt,
                        "Display type: '%s' must contain the type atoms: '%s'" %
                        (dt, atoms),
                        exe=False,
                substrs = list(atoms))

            # The (var, val) pair must match, too.
            nv = ("%s = '%s'" if quotedDisplay else "%s = %s") % (var, val)
            self.expect(output, Msg(var, val, True), exe=False,
                substrs = [nv])

    def generic_type_expr_tester(self, atoms, quotedDisplay=False):
        """Test that variable expressions with basic types are evaluated correctly."""

        # First, capture the golden output emitted by the oracle, i.e., the
        # series of printf statements.
        go = system("./a.out", sender=self)[0]
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
        # 'expr' command.
        self.runCmd("file a.out", CURRENT_EXECUTABLE_SET)
        puts_line = line_number ("basic_type.cpp", "// Here is the line we will break on to check variables.")
        self.expect("breakpoint set -f basic_type.cpp -l %d" % puts_line,
                    BREAKPOINT_CREATED,
            startstr = "Breakpoint created: 1: file ='basic_type.cpp', line = %d, locations = 1" %
                        puts_line)
        self.runCmd("run", RUN_SUCCEEDED)
        self.expect("process status", STOPPED_DUE_TO_BREAKPOINT,
            substrs = [" at basic_type.cpp:%d" % puts_line,
                       "stop reason = breakpoint"])

        #self.runCmd("frame variable -T")

        # Now iterate through the golden list, comparing against the output from
        # 'expr var'.
        for var, val in gl:
            # Don't overwhelm the expression mechanism.
            # This slows down the test suite quite a bit, to enable it, define
            # the environment variable LLDB_TYPES_EXPR_TIME_WAIT.  For example:
            #
            #     export LLDB_TYPES_EXPR_TIME_WAIT=0.5
            #
            # causes a 0.5 second delay between 'expression' commands.
            if "LLDB_TYPES_EXPR_TIME_WAIT" in os.environ:
                time.sleep(float(os.environ["LLDB_TYPES_EXPR_TIME_WAIT"]))

            self.runCmd("expression %s" % var)
            output = self.res.GetOutput()
            
            # The input type is in a canonical form as a set named atoms.
            # The display type string must conatin each and every element.
            #
            # Example:
            #     runCmd: expr a
            #     output: (double) $0 = 1100.12
            #
            try:
                dt = re.match("^\((.*)\) \$[0-9]+ = ", output).group(1)
            except:
                self.fail(self.DATA_TYPE_GROKKED)

            # Expect the display type string to contain each and every atoms.
            self.expect(dt,
                        "Display type: '%s' must contain the type atoms: '%s'" %
                        (dt, atoms),
                        exe=False,
                substrs = list(atoms))

            # The val part must match, too.
            valPart = ("'%s'" if quotedDisplay else "%s") % val
            self.expect(output, Msg(var, val, False), exe=False,
                substrs = [valPart])
