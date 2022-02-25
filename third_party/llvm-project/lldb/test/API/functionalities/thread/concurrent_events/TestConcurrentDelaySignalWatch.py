
import unittest2

from lldbsuite.test.decorators import *
from lldbsuite.test.concurrent_base import ConcurrentEventsBase
from lldbsuite.test.lldbtest import TestBase


@skipIfWindows
class ConcurrentDelaySignalWatch(ConcurrentEventsBase):

    mydir = ConcurrentEventsBase.compute_mydir(__file__)

    # Atomic sequences are not supported yet for MIPS in LLDB.
    @skipIf(triple='^mips')
    @add_test_categories(["watchpoint"])
    def test(self):
        """Test a watchpoint and a (1 second delay) signal in multiple threads."""
        self.build(dictionary=self.getBuildFlags())
        self.do_thread_actions(
            num_delay_signal_threads=1,
            num_watchpoint_threads=1)
