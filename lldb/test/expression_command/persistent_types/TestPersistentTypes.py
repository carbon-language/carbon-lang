"""
Test that lldb persistent types works correctly.
"""

import os, time
import unittest2
import lldb
from lldbtest import *

class PersistenttypesTestCase(TestBase):

    mydir = TestBase.compute_mydir(__file__)

    def test_persistent_types(self):
        """Test that lldb persistent types works correctly."""
        self.buildDefault()

        self.runCmd("file a.out", CURRENT_EXECUTABLE_SET)

        self.runCmd("breakpoint set --name main")

        self.runCmd("run", RUN_SUCCEEDED)

        self.runCmd("expression struct $foo { int a; int b; };")

        self.expect("expression struct $foo $my_foo; $my_foo.a = 2; $my_foo.b = 3;",
                    startstr = "(int) $0 = 3")

        self.expect("expression $my_foo",
                    substrs = ['a = 2', 'b = 3'])

        self.runCmd("expression typedef int $bar")

        self.expect("expression $bar i = 5; i",
                    startstr = "($bar) $1 = 5")

        self.runCmd("expression struct $foobar { char a; char b; char c; char d; };")
        self.runCmd("next")

        self.expect("memory read foo -t $foobar",
                    substrs = ['($foobar) 0x', ' = ', "a = 'H'","b = 'e'","c = 'l'","d = 'l'"]) # persistent types are OK to use for memory read

        self.expect("memory read foo -t foobar",
                    substrs = ['($foobar) 0x', ' = ', "a = 'H'","b = 'e'","c = 'l'","d = 'l'"],matching=False,error=True) # the type name is $foobar, make sure we settle for nothing less


if __name__ == '__main__':
    import atexit
    lldb.SBDebugger.Initialize()
    atexit.register(lambda: lldb.SBDebugger.Terminate())
    unittest2.main()
