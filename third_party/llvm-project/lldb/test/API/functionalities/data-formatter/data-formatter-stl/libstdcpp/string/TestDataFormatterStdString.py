# coding=utf8
"""
Test lldb data formatter subsystem.
"""



import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class StdStringDataFormatterTestCase(TestBase):

    mydir = TestBase.compute_mydir(__file__)

    def setUp(self):
        # Call super's setUp().
        TestBase.setUp(self)
        # Find the line number to break at.
        self.line = line_number('main.cpp', '// Set break point at this line.')

    @add_test_categories(["libstdcxx"])
    @expectedFailureAll(bugnumber="llvm.org/pr50861", compiler="gcc")
    def test_with_run_command(self):
        """Test that that file and class static variables display correctly."""
        self.build()
        self.runCmd("file " + self.getBuildArtifact("a.out"), CURRENT_EXECUTABLE_SET)

        lldbutil.run_break_set_by_file_and_line(
            self, "main.cpp", self.line, num_expected_locations=-1)

        self.runCmd("run", RUN_SUCCEEDED)

        # The stop reason of the thread should be breakpoint.
        self.expect("thread list", STOPPED_DUE_TO_BREAKPOINT,
                    substrs=['stopped',
                             'stop reason = breakpoint'])

        # This is the function to remove the custom formats in order to have a
        # clean slate for the next test case.
        def cleanup():
            self.runCmd('type format clear', check=False)
            self.runCmd('type summary clear', check=False)
            self.runCmd('type filter clear', check=False)
            self.runCmd('type synth clear', check=False)
            self.runCmd(
                "settings set target.max-children-count 256",
                check=False)

        # Execute the cleanup function during test case tear down.
        self.addTearDownHook(cleanup)

        var_wempty = self.frame().FindVariable('wempty')
        var_s = self.frame().FindVariable('s')
        var_S = self.frame().FindVariable('S')
        var_mazeltov = self.frame().FindVariable('mazeltov')
        var_empty = self.frame().FindVariable('empty')
        var_q = self.frame().FindVariable('q')
        var_Q = self.frame().FindVariable('Q')
        var_uchar = self.frame().FindVariable('uchar')

        self.assertEqual(var_wempty.GetSummary(), 'L""', "wempty summary wrong")
        self.assertEqual(
            var_s.GetSummary(), 'L"hello world! מזל טוב!"',
            "s summary wrong")
        self.assertEqual(var_S.GetSummary(), 'L"!!!!"', "S summary wrong")
        self.assertEqual(
            var_mazeltov.GetSummary(), 'L"מזל טוב"',
            "mazeltov summary wrong")
        self.assertEqual(var_empty.GetSummary(), '""', "empty summary wrong")
        self.assertEqual(
            var_q.GetSummary(), '"hello world"',
            "q summary wrong")
        self.assertEqual(
            var_Q.GetSummary(), '"quite a long std::strin with lots of info inside it"',
            "Q summary wrong")
        self.assertEqual(var_uchar.GetSummary(), '"aaaaa"', "u summary wrong")

        self.runCmd("next")

        self.assertEqual(
            var_S.GetSummary(), 'L"!!!!!"',
            "new S summary wrong")
