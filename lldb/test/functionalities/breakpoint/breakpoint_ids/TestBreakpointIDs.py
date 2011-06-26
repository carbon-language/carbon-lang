"""
Test lldb breakpoint ids.
"""

import os, time
import unittest2
import lldb
from lldbtest import *

class BreakpointIDTestCase(TestBase):

    mydir = os.path.join("functionalities", "breakpoint", "breakpoint_ids")

    @unittest2.skipUnless(sys.platform.startswith("darwin"), "requires Darwin")
    def test_with_dsym (self):
        self.buildDsym ()
        self.breakpoint_id_tests ()

    def test_with_dwarf (self):
        self.buildDwarf ()
        self.breakpoint_id_tests ()

    def breakpoint_id_tests (self):
        exe = os.path.join (os.getcwd(), "a.out")
        self.expect("file " + exe,
                    patterns = [ "Current executable set to .*a.out" ])

        self.expect ("breakpoint set -n product",
                     startstr = "Breakpoint created: 1: name = 'product', locations =")

        self.expect ("breakpoint set -n sum",
                     startstr = "Breakpoint created: 2: name = 'sum', locations =")

        self.expect ("breakpoint set -n junk",
                     startstr = "Breakpoint created: 3: name = 'junk', locations = 0 (pending)",
                     substrs = [ "WARNING:  Unable to resolve breakpoint to any actual locations." ] )

        self.expect ("breakpoint disable 1.1 - 2.2 ",
                     COMMAND_FAILED_AS_EXPECTED, error = True,
                     startstr = "error: Invalid range: Ranges that specify particular breakpoint locations must be within the same major breakpoint; you specified two different major breakpoints, 1 and 2.")

        self.expect ("breakpoint disable 2 - 2.2",
                     COMMAND_FAILED_AS_EXPECTED, error = True,
                     startstr = "error: Invalid breakpoint id range:  Either both ends of range must specify a breakpoint location, or neither can specify a breakpoint location.")

        self.expect ("breakpoint disable 2.1 - 2",
                     COMMAND_FAILED_AS_EXPECTED, error = True,
                     startstr = "error: Invalid breakpoint id range:  Either both ends of range must specify a breakpoint location, or neither can specify a breakpoint location.")

        self.expect ("breakpoint disable 2.1 - 2.2",
                     startstr = "2 breakpoints disabled.")

        self.expect ("breakpoint enable 2.*",
                     patterns = [ ".* breakpoints enabled."] )

if __name__ == '__main__':
    import atexit
    lldb.SBDebugger.Initialize()
    atexit.register(lambda: lldb.SBDebugger.Terminate())
    unittest2.main()

