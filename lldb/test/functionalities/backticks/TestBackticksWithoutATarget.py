"""
Test that backticks without a target should work (not infinite looping).
"""

from __future__ import print_function

import use_lldb_suite

import os, time
import lldb
from lldbtest import *

class BackticksWithNoTargetTestCase(TestBase):

    mydir = TestBase.compute_mydir(__file__)

    @no_debug_info_test
    def test_backticks_no_target(self):
        """A simple test of backticks without a target."""
        self.expect("print `1+2-3`",
            substrs = [' = 0'])
