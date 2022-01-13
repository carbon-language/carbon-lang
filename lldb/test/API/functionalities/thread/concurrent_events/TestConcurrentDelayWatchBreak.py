
import unittest2

from lldbsuite.test.decorators import *
from lldbsuite.test.concurrent_base import ConcurrentEventsBase
from lldbsuite.test.lldbtest import TestBase


@skipIfWindows
class ConcurrentDelayWatchBreak(ConcurrentEventsBase):

    mydir = ConcurrentEventsBase.compute_mydir(__file__)

    # Atomic sequences are not supported yet for MIPS in LLDB.
    @skipIf(triple='^mips')
    @skipIf(
        oslist=["ios", "watchos", "tvos", "bridgeos", "macosx"],
        archs=['arm64', 'arm64e', 'arm64_32', 'arm'],
        bugnumber="rdar://81811539")
    @add_test_categories(["watchpoint"])
    def test(self):
        """Test (1-second delay) watchpoint and a breakpoint in multiple threads."""
        self.build(dictionary=self.getBuildFlags())
        self.do_thread_actions(
            num_breakpoint_threads=1,
            num_delay_watchpoint_threads=1)
