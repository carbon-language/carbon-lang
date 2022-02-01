"""
Test lldb data formatter subsystem.
"""


import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class StdSmartPtrDataFormatterTestCase(TestBase):
    mydir = TestBase.compute_mydir(__file__)

    @add_test_categories(["libstdcxx"])
    @expectedFailureAll(bugnumber="llvm.org/pr50861", compiler="gcc")
    def test_with_run_command(self):
        self.build()
        self.runCmd("file " + self.getBuildArtifact("a.out"), CURRENT_EXECUTABLE_SET)

        lldbutil.run_break_set_by_source_regexp(
            self, "Set break point at this line.")
        self.runCmd("run", RUN_SUCCEEDED)

        # The stop reason of the thread should be breakpoint.
        self.expect("thread list", STOPPED_DUE_TO_BREAKPOINT,
                    substrs=['stopped', 'stop reason = breakpoint'])

        self.expect("frame variable nsp", substrs=['nsp = nullptr'])
        self.expect("frame variable isp", substrs=['isp = 123'])
        self.expect("frame variable ssp", substrs=['ssp = "foobar"'])

        self.expect("frame variable nwp", substrs=['nwp = nullptr'])
        self.expect("frame variable iwp", substrs=['iwp = 123'])
        self.expect("frame variable swp", substrs=['swp = "foobar"'])

        self.runCmd("continue")

        self.expect("frame variable nsp", substrs=['nsp = nullptr'])
        self.expect("frame variable isp", substrs=['isp = nullptr'])
        self.expect("frame variable ssp", substrs=['ssp = nullptr'])

        self.expect("frame variable nwp", substrs=['nwp = nullptr'])
        self.expect("frame variable iwp", substrs=['iwp = nullptr'])
        self.expect("frame variable swp", substrs=['swp = nullptr'])
