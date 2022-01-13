"""Test that a forward-declared class works when its complete definition is in a library"""



import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class ForwardDeclTestCase(TestBase):

    mydir = TestBase.compute_mydir(__file__)

    def setUp(self):
        # Call super's setUp().
        TestBase.setUp(self)
        # Find the line number to break inside main().
        self.source = 'main.m'
        self.line = line_number(self.source, '// Set breakpoint 0 here.')
        self.shlib_names = ["Container"]

    def do_test(self, dictionary=None):
        self.build(dictionary=dictionary)

        # Create a target by the debugger.
        target = self.dbg.CreateTarget(self.getBuildArtifact("a.out"))
        self.assertTrue(target, VALID_TARGET)

        # Create the breakpoint inside function 'main'.
        breakpoint = target.BreakpointCreateByLocation(self.source, self.line)
        self.assertTrue(breakpoint, VALID_BREAKPOINT)

        # Register our shared libraries for remote targets so they get
        # automatically uploaded
        environment = self.registerSharedLibrariesWithTarget(
            target, self.shlib_names)

        # Now launch the process, and do not stop at entry point.
        process = target.LaunchSimple(
            None, environment, self.get_process_working_directory())
        self.assertTrue(process, PROCESS_IS_VALID)

        # The stop reason of the thread should be breakpoint.
        self.expect("thread list", STOPPED_DUE_TO_BREAKPOINT,
                    substrs=['stopped',
                             'stop reason = breakpoint'])

        # The breakpoint should have a hit count of 1.
        self.expect("breakpoint list -f", BREAKPOINT_HIT_ONCE,
                    substrs=[' resolved, hit count = 1'])

        # This should display correctly.
        self.expect("expression [j getMember]", VARIABLES_DISPLAYED_CORRECTLY,
                    substrs=["= 0x"])

    def test_expr(self):
        self.do_test()

    @no_debug_info_test
    @skipIf(compiler=no_match("clang"))
    @skipIf(compiler_version=["<", "7.0"])
    def test_debug_names(self):
        """Test that we are able to find complete types when using DWARF v5
        accelerator tables"""
        self.do_test(
            dict(CFLAGS_EXTRAS="-dwarf-version=5 -mllvm -accel-tables=Dwarf"))
