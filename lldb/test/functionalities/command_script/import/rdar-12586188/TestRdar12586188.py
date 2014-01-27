"""Check that we handle an ImportError in a special way when command script importing files."""

import os, sys, time
import unittest2
import lldb
from lldbtest import *

class Rdar12586188TestCase(TestBase):

    mydir = TestBase.compute_mydir(__file__)

    @python_api_test
    def test_rdar12586188_command(self):
        """Check that we handle an ImportError in a special way when command script importing files."""
        self.run_test()

    def setUp(self):
        # Call super's setUp().
        TestBase.setUp(self)

    def run_test(self):
        """Check that we handle an ImportError in a special way when command script importing files."""

        self.expect("command script import ./fail12586188.py --allow-reload",
                error=True, substrs = ['raise ImportError("I do not want to be imported")'])
        self.expect("command script import ./fail212586188.py --allow-reload",
                error=True, substrs = ['raise ValueError("I do not want to be imported")'])

if __name__ == '__main__':
    import atexit
    lldb.SBDebugger.Initialize()
    atexit.register(lambda: lldb.SBDebugger.Terminate())
    unittest2.main()
