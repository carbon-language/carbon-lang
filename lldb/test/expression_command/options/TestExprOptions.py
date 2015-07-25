"""
Test expression command options.

Test cases:

o test_expr_options:
  Test expression command options.
"""

import os, time
import unittest2
import lldb
import lldbutil
from lldbtest import *

class ExprOptionsTestCase(TestBase):

    mydir = TestBase.compute_mydir(__file__)

    def setUp(self):
        # Call super's setUp().
        TestBase.setUp(self)

        self.main_source = "main.cpp"
        self.main_source_spec = lldb.SBFileSpec (self.main_source)
        self.line = line_number('main.cpp', '// breakpoint_in_main')
        self.exe = os.path.join(os.getcwd(), "a.out")

    @expectedFailureLinux # ObjC expression broken on Linux?
    def test_expr_options(self):
        """These expression command options should work as expected."""
        self.buildDefault()

        # Set debugger into synchronous mode
        self.dbg.SetAsync(False)

        # Create a target by the debugger.
        target = self.dbg.CreateTarget(self.exe)
        self.assertTrue(target, VALID_TARGET)

        # Set breakpoints inside main.
        breakpoint = target.BreakpointCreateBySourceRegex('// breakpoint_in_main', self.main_source_spec)
        self.assertTrue(breakpoint)

        # Now launch the process, and do not stop at entry point.
        process = target.LaunchSimple(None, None, self.get_process_working_directory())
        self.assertTrue(process, PROCESS_IS_VALID)

        threads = lldbutil.get_threads_stopped_at_breakpoint(process, breakpoint)
        self.assertEqual(len(threads), 1)

        frame = threads[0].GetFrameAtIndex(0)
        options = lldb.SBExpressionOptions()

        # -- test --language on ObjC builtin type using the SB API's --
        # Make sure we can evaluate the ObjC builtin type 'id':
        val = frame.EvaluateExpression('id my_id = 0; my_id')
        self.assertTrue(val.IsValid())
        self.assertTrue(val.GetError().Success())
        self.assertEqual(val.GetValueAsUnsigned(0), 0)
        self.DebugSBValue(val)

        # Make sure it still works if language is set to ObjC:
        options.SetLanguage(lldb.eLanguageTypeObjC)
        val = frame.EvaluateExpression('id my_id = 0; my_id', options)
        self.assertTrue(val.IsValid())
        self.assertTrue(val.GetError().Success())
        self.assertEqual(val.GetValueAsUnsigned(0), 0)
        self.DebugSBValue(val)

        # Make sure it fails if language is set to C:
        options.SetLanguage(lldb.eLanguageTypeC)
        val = frame.EvaluateExpression('id my_id = 0; my_id', options)
        self.assertTrue(val.IsValid())
        self.assertFalse(val.GetError().Success())

        # -- test --language on C++ expression using the SB API's --
        # Make sure we can evaluate 'ns::func'.
        val = frame.EvaluateExpression('ns::func')
        self.assertTrue(val.IsValid())
        self.assertTrue(val.GetError().Success())
        self.DebugSBValue(val)

        # Make sure it still works if language is set to C++:
        options.SetLanguage(lldb.eLanguageTypeC_plus_plus)
        val = frame.EvaluateExpression('ns::func', options)
        self.assertTrue(val.IsValid())
        self.assertTrue(val.GetError().Success())
        self.DebugSBValue(val)

        # Make sure it fails if language is set to C:
        options.SetLanguage(lldb.eLanguageTypeC)
        val = frame.EvaluateExpression('ns::func', options)
        self.assertTrue(val.IsValid())
        self.assertFalse(val.GetError().Success())

if __name__ == '__main__':
    import atexit
    lldb.SBDebugger.Initialize()
    atexit.register(lambda: lldb.SBDebugger.Terminate())
    unittest2.main()
