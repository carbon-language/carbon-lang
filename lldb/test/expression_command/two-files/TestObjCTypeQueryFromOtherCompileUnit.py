"""
Regression test for <rdar://problem/8981098>:

The expression parser's type search only looks in the current compilation unit for types.
"""

import unittest2
import lldb
from lldbtest import *

class ObjCTypeQueryTestCase(TestBase):

    mydir = os.path.join("expression_command", "two-files")

    def setUp(self):
        # Call super's setUp().
        TestBase.setUp(self)
        # Find the line number to break for main.c.
        self.line = line_number('main.c',
                                "// Set breakpoint here, then do 'expr (NSArray*)array_token'.")

    @unittest2.skipUnless(sys.platform.startswith("darwin"), "requires Darwin")
    def test_with_dsym(self):
        """The expression parser's type search should be wider than the current compilation unit."""
        self.buildDsym()
        self.type_query_from_other_cu()

    def test_with_dwarf(self):
        """The expression parser's type search should be wider than the current compilation unit."""
        self.buildDwarf()
        self.type_query_from_other_cu()

    def type_query_from_other_cu(self):
        """The expression parser's type search should be wider than the current compilation unit."""
        self.runCmd("file a.out", CURRENT_EXECUTABLE_SET)

        self.expect("breakpoint set -f main.c -l %d" % self.line,
                    BREAKPOINT_CREATED,
            startstr = "Breakpoint created: 1: file ='main.c', line = %d" %
                        self.line)

        self.runCmd("run", RUN_SUCCEEDED)

        # Now do a NSArry type query from the 'main.c' compile uint.
        self.expect("expression (NSArray*)array_token",
            substrs = ['(NSArray *) $0 ='])
        # (NSArray *) $0 = 0x00007fff70118398


if __name__ == '__main__':
    import atexit
    lldb.SBDebugger.Initialize()
    atexit.register(lambda: lldb.SBDebugger.Terminate())
    unittest2.main()
