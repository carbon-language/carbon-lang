"""Look up enum type information and check for correct display."""

import os, time
import unittest2
import lldb
from lldbtest import *

class TestEnumTypes(TestBase):

    mydir = "enum_types"

    def test_image_lookup_for_enum_type(self):
        """Test 'image lookup -t days' and check for correct display."""
        exe = os.path.join(os.getcwd(), "a.out")
        self.runCmd("file " + exe, CURRENT_EXECUTABLE_SET)

        # Break inside the main.
        self.expect("breakpoint set -f main.c -l 26", BREAKPOINT_CREATED,
            startstr = "Breakpoint created: 1: file ='main.c', line = 26, locations = 1")

        self.runCmd("run", RUN_STOPPED)

        # The stop reason of the thread should be breakpoint.
        self.expect("thread list", STOPPED_DUE_TO_BREAKPOINT,
            substrs = ['state is Stopped',
                       'stop reason = breakpoint'])

        # The breakpoint should have a hit count of 1.
        self.expect("breakpoint list", BREAKPOINT_HIT_ONCE,
            substrs = [' resolved, hit count = 1'])

        # Look up information about the 'days' enum type.
        # Check for correct display.
        self.expect("image lookup -t days", DATA_TYPES_DISPLAYED_CORRECTLY,
            substrs = ['enum days {',
                       'Monday,',
                       'Tuesday',
                       'Wednesday',
                       'Thursday',
                       'Friday',
                       'Saturday',
                       'Sunday,',
                       'kNumDays',
                       '}'])


if __name__ == '__main__':
    import atexit
    lldb.SBDebugger.Initialize()
    atexit.register(lambda: lldb.SBDebugger.Terminate())
    unittest2.main()
