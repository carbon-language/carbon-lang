"""
Test lldb data formatter subsystem.
"""

from __future__ import print_function


import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class DataFormatterSynthValueTestCase(TestBase):

    mydir = TestBase.compute_mydir(__file__)

    def setUp(self):
        # Call super's setUp().
        TestBase.setUp(self)
        # Find the line number to break at.
        self.line = line_number('main.cpp', 'break here')

    @expectedFailureAll(bugnumber="llvm.org/pr50814", compiler="gcc")
    def test_with_run_command(self):
        """Test using Python synthetic children provider to provide a value."""
        self.build()
        self.runCmd("file " + self.getBuildArtifact("a.out"), CURRENT_EXECUTABLE_SET)

        lldbutil.run_break_set_by_file_and_line(
            self, "main.cpp", self.line, num_expected_locations=1, loc_exact=True)

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

        # Execute the cleanup function during test case tear down.
        self.addTearDownHook(cleanup)

        x = self.frame().FindVariable("x")
        x.SetPreferSyntheticValue(True)
        y = self.frame().FindVariable("y")
        y.SetPreferSyntheticValue(True)
        z = self.frame().FindVariable("z")
        z.SetPreferSyntheticValue(True)
        q = self.frame().FindVariable("q")
        z.SetPreferSyntheticValue(True)

        x_val = x.GetValueAsUnsigned
        y_val = y.GetValueAsUnsigned
        z_val = z.GetValueAsUnsigned
        q_val = q.GetValueAsUnsigned

        if self.TraceOn():
            print(
                "x_val = %s; y_val = %s; z_val = %s; q_val = %s" %
                (x_val(), y_val(), z_val(), q_val()))

        self.assertNotEqual(x_val(), 3, "x == 3 before synthetics")
        self.assertNotEqual(y_val(), 4, "y == 4 before synthetics")
        self.assertNotEqual(z_val(), 7, "z == 7 before synthetics")
        self.assertNotEqual(q_val(), 8, "q == 8 before synthetics")

        # now set up the synth
        self.runCmd("script from myIntSynthProvider import *")
        self.runCmd("type synth add -l myIntSynthProvider myInt")
        self.runCmd("type synth add -l myArraySynthProvider myArray")
        self.runCmd("type synth add -l myIntSynthProvider myIntAndStuff")

        if self.TraceOn():
            print(
                "x_val = %s; y_val = %s; z_val = %s; q_val = %s" %
                (x_val(), y_val(), z_val(), q_val()))

        self.assertEqual(x_val(), 3, "x != 3 after synthetics")
        self.assertEqual(y_val(), 4, "y != 4 after synthetics")
        self.assertEqual(z_val(), 7, "z != 7 after synthetics")
        self.assertEqual(q_val(), 8, "q != 8 after synthetics")

        self.expect("frame variable x", substrs=['3'])
        self.expect(
            "frame variable x",
            substrs=['theValue = 3'],
            matching=False)
        self.expect("frame variable q", substrs=['8'])
        self.expect(
            "frame variable q",
            substrs=['theValue = 8'],
            matching=False)

        # check that an aptly defined synthetic provider does not affect
        # one-lining
        self.expect(
            "expression struct Struct { myInt theInt{12}; }; Struct()",
            substrs=['(theInt = 12)'])

        # check that we can use a synthetic value in a summary
        self.runCmd("type summary add hasAnInt -s ${var.theInt}")
        hi = self.frame().FindVariable("hi")
        self.assertEqual(hi.GetSummary(), "42")

        ma = self.frame().FindVariable("ma")
        self.assertTrue(ma.IsValid())
        self.assertEqual(ma.GetNumChildren(15), 15)
        self.assertEqual(ma.GetNumChildren(16), 16)
        self.assertEqual(ma.GetNumChildren(17), 16)
