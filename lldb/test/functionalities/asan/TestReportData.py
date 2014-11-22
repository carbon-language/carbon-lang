"""
Test the AddressSanitizer runtime support for report breakpoint and data extraction.
"""

import os, time
import unittest2
import lldb
from lldbtest import *
import lldbutil
import json

class AsanTestReportDataCase(TestBase):

    mydir = TestBase.compute_mydir(__file__)

    # The default compiler ("clang") may not support Address Sanitizer or it
    # may not have the debugging API which was recently added, so we're calling
    # self.useBuiltClang() to use clang from the llvm-build directory instead

    @unittest2.skipUnless(sys.platform.startswith("darwin"), "requires Darwin")
    @skipIfRemote
    @dsym_test
    def test_with_dsym (self):
        compiler = self.findBuiltClang ()
        self.buildDsym (None, compiler)
        self.asan_tests ()

    @skipIfFreeBSD # llvm.org/pr21136 runtimes not yet available by default
    @skipIfRemote
    @expectedFailureLinux # non-core functionality, need to reenable and fix later (DES 2014.11.07)
    @dwarf_test
    def test_with_dwarf (self):
        compiler = self.findBuiltClang ()
        self.buildDwarf (None, compiler)
        self.asan_tests ()

    def setUp(self):
        # Call super's setUp().
        TestBase.setUp(self)
        self.line_malloc = line_number('main.c', '// malloc line')
        self.line_malloc2 = line_number('main.c', '// malloc2 line')
        self.line_free = line_number('main.c', '// free line')
        self.line_breakpoint = line_number('main.c', '// break line')
        self.line_crash = line_number('main.c', '// BOOM line')

    def asan_tests (self):
        exe = os.path.join (os.getcwd(), "a.out")
        self.expect("file " + exe, patterns = [ "Current executable set to .*a.out" ])
        self.runCmd("run")

        # ASan will relaunch the process to insert its library.
        self.expect("thread list", "Process should be stopped due to exec.",
            substrs = ['stopped', 'stop reason = exec'])

        # no extended info when we have no ASan report
        thread = self.dbg.GetSelectedTarget().process.GetSelectedThread()
        s = lldb.SBStream()
        self.assertFalse(thread.GetStopReasonExtendedInfoAsJSON(s))

        self.runCmd("continue")

        self.expect("thread list", "Process should be stopped due to ASan report",
            substrs = ['stopped', 'stop reason = Use of deallocated memory detected'])

        self.assertEqual(self.dbg.GetSelectedTarget().process.GetSelectedThread().GetStopReason(), lldb.eStopReasonInstrumentation)

        self.expect("bt", "The backtrace should show the crashing line",
            substrs = ['main.c:%d' % self.line_crash])

        self.expect("thread info -s", "The extended stop info should contain the ASan provided fields",
            substrs = ["access_size", "access_type", "address", "pc", "description", "heap-use-after-free"])

        output_lines = self.res.GetOutput().split('\n')
        json_line = output_lines[2]
        data = json.loads(json_line)
        self.assertEqual(data["description"], "heap-use-after-free")
        self.assertEqual(data["instrumentation_class"], "AddressSanitizer")
        self.assertEqual(data["stop_type"], "fatal_error")

        # now let's try the SB API
        process = self.dbg.GetSelectedTarget().process
        thread = process.GetSelectedThread()

        s = lldb.SBStream()
        self.assertTrue(thread.GetStopReasonExtendedInfoAsJSON(s))
        s = s.GetData()
        data2 = json.loads(s)
        self.assertEqual(data, data2)

if __name__ == '__main__':
    import atexit
    lldb.SBDebugger.Initialize()
    atexit.register(lambda: lldb.SBDebugger.Terminate())
    unittest2.main()
