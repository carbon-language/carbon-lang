"""
Test some target commands: create, list, select.
"""

import unittest2
import lldb
import sys
from lldbtest import *

class targetCommandTestCase(TestBase):

    mydir = "target"

    def setUp(self):
        # Call super's setUp().
        TestBase.setUp(self)
        # Find the line numbers for our breakpoints.
        self.line_b = line_number('b.c', '// Set break point at this line.')
        self.line_c = line_number('c.c', '// Set break point at this line.')

    def test_target_command_with_dwarf(self):
        """Test some target commands: create, list, select."""
        da = {'C_SOURCES': 'a.c', 'EXE': 'a.out'}
        self.buildDefault(dictionary=da)
        self.addTearDownCleanup(dictionary=da)

        db = {'C_SOURCES': 'b.c', 'EXE': 'b.out'}
        self.buildDefault(dictionary=db)
        self.addTearDownCleanup(dictionary=db)

        dc = {'C_SOURCES': 'c.c', 'EXE': 'c.out'}
        self.buildDefault(dictionary=dc)
        self.addTearDownCleanup(dictionary=dc)

        self.do_target_command()

    def do_target_command(self):
        """Exercise 'target create', 'target list', 'target select' commands."""
        exe_a = os.path.join(os.getcwd(), "a.out")
        exe_b = os.path.join(os.getcwd(), "b.out")
        exe_c = os.path.join(os.getcwd(), "c.out")

        self.runCmd("target list")
        output = self.res.GetOutput()
        if output.startswith("No targets"):
            # We start from index 0.
            base = 0
        else:
            # Find the largest index of the existing list.
            import re
            pattern = re.compile("target #(\d+):")
            for line in reversed(output.split(os.linesep)):
                match = pattern.search(line)
                if match:
                    # We will start from (index + 1) ....
                    base = int(match.group(1), 10) + 1
                    #print "base is:", base
                    break;

        self.runCmd("target create " + exe_a, CURRENT_EXECUTABLE_SET)
        self.runCmd("run", RUN_SUCCEEDED)

        self.runCmd("target create " + exe_b, CURRENT_EXECUTABLE_SET)
        self.runCmd("breakpoint set -f %s -l %d" % ('b.c', self.line_b),
                    BREAKPOINT_CREATED)
        self.runCmd("run", RUN_SUCCEEDED)

        self.runCmd("target create " + exe_c, CURRENT_EXECUTABLE_SET)
        self.runCmd("breakpoint set -f %s -l %d" % ('c.c', self.line_c),
                    BREAKPOINT_CREATED)
        self.runCmd("run", RUN_SUCCEEDED)

        self.runCmd("target list")

        self.runCmd("target select %d" % base)
        self.runCmd("thread backtrace")

        self.runCmd("target select %d" % (base + 2))
        self.expect("thread backtrace", STOPPED_DUE_TO_BREAKPOINT,
            substrs = ['c.c:%d' % self.line_c,
                       'stop reason = breakpoint'])

        self.runCmd("target select %d" % (base + 1))
        self.expect("thread backtrace", STOPPED_DUE_TO_BREAKPOINT,
            substrs = ['b.c:%d' % self.line_b,
                       'stop reason = breakpoint'])

        self.runCmd("target list")


if __name__ == '__main__':
    import atexit
    lldb.SBDebugger.Initialize()
    atexit.register(lambda: lldb.SBDebugger.Terminate())
    unittest2.main()
