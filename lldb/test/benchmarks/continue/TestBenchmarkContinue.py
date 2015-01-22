"""
Test lldb data formatter subsystem.
"""

import os, time
import unittest2
import lldb
from lldbbench import *
import lldbutil

class TestBenchmarkContinue(BenchBase):

    mydir = TestBase.compute_mydir(__file__)

    @benchmarks_test
    @unittest2.skipUnless(sys.platform.startswith("darwin"), "requires Darwin")
    @dsym_test
    def test_with_dsym_and_run_command(self):
        """Benchmark different ways to continue a process"""
        self.buildDsym()
        self.data_formatter_commands()

    @benchmarks_test
    @skipIfLinux # No standard locations for libc++ on Linux, so skip for now 
    @dwarf_test
    def test_with_dwarf_and_run_command(self):
        """Benchmark different ways to continue a process"""
        self.buildDwarf()
        self.data_formatter_commands()

    def setUp(self):
        # Call super's setUp().
        BenchBase.setUp(self)

    def data_formatter_commands(self):
        """Benchmark different ways to continue a process"""
        self.runCmd("file a.out", CURRENT_EXECUTABLE_SET)

        bkpt = self.target().FindBreakpointByID(lldbutil.run_break_set_by_source_regexp (self, "// break here"))

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
        
        runCmd_sw = Stopwatch()
        lldbutil_sw = Stopwatch()

        for i in range(0,15):
            runCmd_sw.start()
            self.runCmd("continue")
            runCmd_sw.stop()
        
        for i in range(0,15):
            lldbutil_sw.start()
            lldbutil.continue_to_breakpoint(self.process(), bkpt)
            lldbutil_sw.stop()
            
        print "runCmd: %s\nlldbutil: %s" % (runCmd_sw,lldbutil_sw)

if __name__ == '__main__':
    import atexit
    lldb.SBDebugger.Initialize()
    atexit.register(lambda: lldb.SBDebugger.Terminate())
    unittest2.main()
