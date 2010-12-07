"""
Test many basic expression commands.
"""

import os, time
import unittest2
import lldb
from lldbtest import *

class BasicExprCommandsTestCase(TestBase):

    mydir = os.path.join("expression_command", "test")

    def setUp(self):
        # Call super's setUp().
        TestBase.setUp(self)
        # Find the line number to break for main.c.
        self.line = line_number('main.cpp',
                                '// Please test many expressions while stopped at this line:')

    def test_many_expr_commands(self):
        """These basic expression commands should work as expected."""
        self.buildDefault()

        self.runCmd("file a.out", CURRENT_EXECUTABLE_SET)

        self.expect("breakpoint set -f main.cpp -l %d" % self.line,
                    BREAKPOINT_CREATED,
            startstr = "Breakpoint created: 1: file ='main.cpp', line = %d" %
                        self.line)

        self.runCmd("run", RUN_SUCCEEDED)

        self.expect("expression 2",
            patterns = ["\(int\) \$.* = 2"])
        # (int) $0 = 1

        self.expect("expression 2ull",
            patterns = ["\(unsigned long long\) \$.* = 2"])
        # (unsigned long long) $1 = 2

        self.expect("expression 2.234f",
            patterns = ["\(float\) \$.* = 2\.234"])
        # (float) $2 = 2.234

        self.expect("expression 2.234",
            patterns = ["\(double\) \$.* = 2\.234"])
        # (double) $3 = 2.234

        self.expect("expression 2+3",
            patterns = ["\(int\) \$.* = 5"])
        # (int) $4 = 5

        self.expect("expression argc",
            patterns = ["\(int\) \$.* = 1"])
        # (int) $5 = 1

        self.expect("expression argc + 22",
            patterns = ["\(int\) \$.* = 23"])
        # (int) $6 = 23

        self.expect("expression argv",
            patterns = ["\(const char \*\*\) \$.* = 0x"])
        # (const char *) $7 = ...

        self.expect("expression argv[0]",
            substrs = ["(const char *)",
                       os.path.join(self.mydir, "a.out")])
        # (const char *) $8 = 0x... "/Volumes/data/lldb/svn/trunk/test/expression_command/test/a.out"


    @unittest2.expectedFailure
    # rdar://problem/8686536
    # CommandInterpreter::HandleCommand is stripping \'s from input for WantsRawCommand commands
    def test_expr_commands_can_handle_quotes(self):
        """Throw some expression commands with quotes at lldb."""
        self.buildDefault()

        self.runCmd("file a.out", CURRENT_EXECUTABLE_SET)

        self.expect("breakpoint set -f main.cpp -l %d" % self.line,
                    BREAKPOINT_CREATED,
            startstr = "Breakpoint created: 1: file ='main.cpp', line = %d" %
                        self.line)

        self.runCmd("run", RUN_SUCCEEDED)

        # runCmd: expression 'a'
        # output: (char) $0 = 'a'
        self.runCmd("expression 'a'")

        # runCmd: expression printf ("\n\n\tHello there!")
        # output: (unsigned long) $1 = 15
        self.runCmd('expression printf ("\\n\\n\\tHello there!")')

        # runCmd: expression printf("\t\x68\n")
        # output: (unsigned long) $2 = 3
        self.runCmd('expression printf("\\t\\x68\\n")')

        # runCmd: expression printf("\"\n")
        # output: (unsigned long) $3 = 2
        self.runCmd('expression printf("\\"\\n")')

        # runCmd: expression printf("'\n")
        # output: (unsigned long) $4 = 2
        self.runCmd('expression printf("\'\\n")')

        # runCmd: command alias print_hi expression printf ("\n\tHi!")
        # output: 
        self.runCmd('command alias print_hi expression printf ("\\n\\tHi!")')
        # This fails currently.
        self.runCmd('print_hi')


if __name__ == '__main__':
    import atexit
    lldb.SBDebugger.Initialize()
    atexit.register(lambda: lldb.SBDebugger.Terminate())
    unittest2.main()
