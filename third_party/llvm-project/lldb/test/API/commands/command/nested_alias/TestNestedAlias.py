"""
Test that an alias can reference other aliases without crashing.
"""



import lldb
from lldbsuite.test.lldbtest import *
import lldbsuite.test.lldbutil as lldbutil


class NestedAliasTestCase(TestBase):

    mydir = TestBase.compute_mydir(__file__)
    NO_DEBUG_INFO_TESTCASE = True

    def setUp(self):
        # Call super's setUp().
        TestBase.setUp(self)
        # Find the line number to break inside main().
        self.line = line_number('main.cpp', '// break here')

    def test_nested_alias(self):
        """Test that an alias can reference other aliases without crashing."""
        self.build()
        exe = self.getBuildArtifact("a.out")
        self.runCmd("file " + exe, CURRENT_EXECUTABLE_SET)

        # Break in main() after the variables are assigned values.
        lldbutil.run_break_set_by_file_and_line(
            self, "main.cpp", self.line, num_expected_locations=1, loc_exact=True)

        self.runCmd("run", RUN_SUCCEEDED)

        # The stop reason of the thread should be breakpoint.
        self.expect("thread list", STOPPED_DUE_TO_BREAKPOINT,
                    substrs=['stopped', 'stop reason = breakpoint'])

        # The breakpoint should have a hit count of 1.
        lldbutil.check_breakpoint(self, bpno = 1, expected_hit_count = 1)

        # This is the function to remove the custom aliases in order to have a
        # clean slate for the next test case.
        def cleanup():
            self.runCmd('command unalias read', check=False)
            self.runCmd('command unalias rd', check=False)
            self.runCmd('command unalias fo', check=False)
            self.runCmd('command unalias foself', check=False)
            self.runCmd('command unalias add_two', check=False)
            self.runCmd('command unalias two', check=False)

        # Execute the cleanup function during test case tear down.
        self.addTearDownHook(cleanup)

        self.runCmd('command alias read memory read -f A')
        self.runCmd('command alias rd read -c 3')

        self.expect(
            'memory read -f A -c 3 `&my_ptr[0]`',
            substrs=[
                'deadbeef',
                'main.cpp:',
                'feedbeef'])
        self.expect(
            'rd `&my_ptr[0]`',
            substrs=[
                'deadbeef',
                'main.cpp:',
                'feedbeef'])

        self.expect(
            'memory read -f A -c 3 `&my_ptr[0]`',
            substrs=['deadfeed'],
            matching=False)
        self.expect('rd `&my_ptr[0]`', substrs=['deadfeed'], matching=False)

        self.runCmd('command alias fo frame variable -O --')
        self.runCmd('command alias foself fo self')

        self.expect(
            'help foself',
            substrs=[
                '--show-all-children',
                '--raw-output'],
            matching=False)
        self.expect(
            'help foself',
            substrs=[
                'Show variables for the current',
                'stack frame.'],
            matching=True)

        # Check that foself was resolved and is now independent of 'fo'.
        self.runCmd('command unalias fo')
        self.expect(
            'help foself',
            substrs=[
                'Show variables for the current',
                'stack frame.'],
            matching=True)

        # Check that aliases can be created for raw input commands.
        self.expect('command alias two expr -- 2')
        self.expect('command alias add_two two +')
        self.expect('add_two 3', patterns=[' = 5$'])
