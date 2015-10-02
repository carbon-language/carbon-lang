"""
Test type lookup command.
"""

import os, time
import unittest2
import lldb
from lldbtest import *
import datetime
import lldbutil

class TypeLookupTestCase(TestBase):

    mydir = TestBase.compute_mydir(__file__)

    def setUp(self):
        # Call super's setUp().
        TestBase.setUp(self)
        # Find the line number to break at.
        self.line = line_number('main.m', '// break here')

    @skipUnlessDarwin
    def test_type_lookup(self):
        """Test type lookup command."""
        self.build()
        self.runCmd("file a.out", CURRENT_EXECUTABLE_SET)

        lldbutil.run_break_set_by_file_and_line (self, "main.m", self.line, num_expected_locations=1, loc_exact=True)

        self.runCmd("run", RUN_SUCCEEDED)

        # The stop reason of the thread should be breakpoint.
        self.expect("thread list", STOPPED_DUE_TO_BREAKPOINT,
            substrs = ['stopped',
                       'stop reason = breakpoint'])
                       
        self.expect('type lookup NoSuchType', substrs=['@interface'], matching=False)
        self.expect('type lookup NSURL', substrs=['NSURL'])
        self.expect('type lookup NSArray', substrs=['NSArray'])
        self.expect('type lookup NSObject', substrs=['NSObject', 'isa'])

if __name__ == '__main__':
    import atexit
    lldb.SBDebugger.Initialize()
    atexit.register(lambda: lldb.SBDebugger.Terminate())
    unittest2.main()
