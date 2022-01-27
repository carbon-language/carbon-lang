"""
Test lldb-vscode setBreakpoints request
"""

from __future__ import print_function

import unittest2
import vscode
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil
import lldbvscode_testcase


class TestVSCode_correct_thread(lldbvscode_testcase.VSCodeTestCaseBase):

    mydir = TestBase.compute_mydir(__file__)

    @skipIfWindows
    @skipIfRemote
    def test_correct_thread(self):
        '''
            Tests that the correct thread is selected if we continue from
            a thread that goes away and hit a breakpoint in another thread.
            In this case, the selected thread should be the thread that
            just hit the breakpoint, and not the first thread in the list.
        '''
        program = self.getBuildArtifact("a.out")
        self.build_and_launch(program)
        source = 'main.c'
        breakpoint_line = line_number(source, '// break here')
        lines = [breakpoint_line]
        # Set breakpoint in the thread function
        breakpoint_ids = self.set_source_breakpoints(source, lines)
        self.assertEqual(len(breakpoint_ids), len(lines),
                        "expect correct number of breakpoints")
        self.continue_to_breakpoints(breakpoint_ids)
        # We're now stopped at the breakpoint in the first thread, thread #2.
        # Continue to join the first thread and hit the breakpoint in the
        # second thread, thread #3.
        self.vscode.request_continue()
        stopped_event = self.vscode.wait_for_stopped()
        # Verify that the description is the relevant breakpoint,
        # preserveFocusHint is False and threadCausedFocus is True
        self.assertTrue(stopped_event[0]['body']['description'].startswith('breakpoint %s.' % breakpoint_ids[0]))
        self.assertFalse(stopped_event[0]['body']['preserveFocusHint'])
        self.assertTrue(stopped_event[0]['body']['threadCausedFocus'])
