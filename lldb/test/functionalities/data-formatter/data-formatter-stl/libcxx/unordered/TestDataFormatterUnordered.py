"""
Test lldb data formatter subsystem.
"""

import os, time
import unittest2
import lldb
from lldbtest import *
import lldbutil

class LibcxxMultiMapDataFormatterTestCase(TestBase):

    mydir = TestBase.compute_mydir(__file__)

    @unittest2.skipUnless(sys.platform.startswith("darwin"), "requires Darwin")
    @dsym_test
    def test_with_dsym_and_run_command(self):
        """Test data formatter commands."""
        self.buildDsym()
        self.data_formatter_commands()

    @skipIfLinux # No standard locations for libc++ on Linux, so skip for now 
    @dwarf_test
    def test_with_dwarf_and_run_command(self):
        """Test data formatter commands."""
        self.buildDwarf()
        self.data_formatter_commands()

    def setUp(self):
        # Call super's setUp().
        TestBase.setUp(self)

    def look_for_content_and_continue(self,var_name,substrs):
        self.expect( ("frame variable %s" % var_name), substrs )
        self.runCmd("continue")

    def data_formatter_commands(self):
        """Test that that file and class static variables display correctly."""
        self.runCmd("file a.out", CURRENT_EXECUTABLE_SET)

        lldbutil.run_break_set_by_source_regexp (self, "Set break point at this line.")

        self.runCmd("run", RUN_SUCCEEDED)

        # The stop reason of the thread should be breakpoint.
        self.expect("thread list", STOPPED_DUE_TO_BREAKPOINT,
            substrs = ['stopped',
                       'stop reason = breakpoint'])

        # This is the function to remove the custom formats in order to have a
        # clean slate for the next test case.
        def cleanup():
            self.runCmd('type format clear', check=False)
            self.runCmd('type summary clear', check=False)
            self.runCmd('type filter clear', check=False)
            self.runCmd('type synth clear', check=False)
            self.runCmd("settings set target.max-children-count 256", check=False)

        # Execute the cleanup function during test case tear down.
        self.addTearDownHook(cleanup)

        self.expect('image list', substrs = self.getLibcPlusPlusLibs())

        self.look_for_content_and_continue("map",['size=5 {,''hello','world','this','is','me'])
        self.look_for_content_and_continue("mmap",['size=6 {','first = 3','second = "this"','first = 2','second = "hello"'])
        self.look_for_content_and_continue("iset",['size=5 {','[0] = 5','[2] = 3','[3] = 2'])
        self.look_for_content_and_continue("sset",['size=5 {','[0] = "is"','[1] = "world"','[4] = "hello"'])
        self.look_for_content_and_continue("imset",['size=6 {','[0] = 3','[1] = 3','[2] = 3','[4] = 2','[5] = 1'])
        self.look_for_content_and_continue("smset",['size=5 {','[0] = "is"','[1] = "is"','[2] = "world"','[3] = "world"'])

if __name__ == '__main__':
    import atexit
    lldb.SBDebugger.Initialize()
    atexit.register(lambda: lldb.SBDebugger.Terminate())
    unittest2.main()
