"""
Test that objective-c constant strings are generated correctly by the expression
parser.
"""

import os, time
import unittest2
import lldb
from lldbtest import *

@unittest2.skipUnless(sys.platform.startswith("darwin"), "requires Darwin")
class ConstStringTestCase(TestBase):

    mydir = os.path.join("lang", "objc", "foundation")
    d = {'OBJC_SOURCES': 'const-strings.m'}

    def test_break_with_dsym(self):
        """Test constant string generation amd comparison by the expression parser."""
        self.buildDsym(dictionary=self.d)
        self.setTearDownCleanup(self.d)
        self.objc_const_strings()

    def test_break_with_dwarf(self):
        """Test constant string generation amd comparison by the expression parser."""
        self.buildDwarf(dictionary=self.d)
        self.setTearDownCleanup(self.d)
        self.objc_const_strings()

    def setUp(self):
        # Call super's setUp().
        TestBase.setUp(self)
        # Find the line number to break inside main().
        self.main_source = "const-strings.m"
        self.line = line_number(self.main_source, '// Set breakpoint here.')

    def objc_const_strings(self):
        """Test constant string generation amd comparison by the expression parser."""
        exe = os.path.join(os.getcwd(), "a.out")
        self.runCmd("file " + exe, CURRENT_EXECUTABLE_SET)

        self.expect("breakpoint set -f %s -l %d" % (self.main_source, self.line),
                    BREAKPOINT_CREATED,
            startstr = "Breakpoint created: 1: file ='%s', line = %d, locations = 1" %
                        (self.main_source, self.line))

        self.runCmd("run", RUN_SUCCEEDED)
        self.expect("process status", STOPPED_DUE_TO_BREAKPOINT,
            substrs = [" at %s:%d" % (self.main_source, self.line),
                       "stop reason = breakpoint"])

        self.expect('expression (int)[str compare:@"hello"]',
            startstr = "(int) $0 = 0")
        self.expect('expression (int)[str compare:@"world"]',
            startstr = "(int) $1 = -1")

        # Test empty strings, too.
        self.expect('expression (int)[@"" length]',
            startstr = "(int) $2 = 0")

        self.expect('expression (int)[@"123" length]',
            startstr = "(int) $3 = 3")

        
if __name__ == '__main__':
    import atexit
    lldb.SBDebugger.Initialize()
    atexit.register(lambda: lldb.SBDebugger.Terminate())
    unittest2.main()
