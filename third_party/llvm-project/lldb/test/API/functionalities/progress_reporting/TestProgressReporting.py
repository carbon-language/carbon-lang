"""
Test that we are able to broadcast and receive progress events from lldb
"""
import lldb

import lldbsuite.test.lldbutil as lldbutil

from lldbsuite.test.lldbtest import *


class TestProgressReporting(TestBase):

    mydir = TestBase.compute_mydir(__file__)

    def setUp(self):
        TestBase.setUp(self)
        self.broadcaster = self.dbg.GetBroadcaster()
        self.listener = lldbutil.start_listening_from(self.broadcaster,
                                        lldb.SBDebugger.eBroadcastBitProgress)

    def test_dwarf_symbol_loading_progress_report(self):
        """Test that we are able to fetch dwarf symbol loading progress events"""
        self.build()

        lldbutil.run_to_source_breakpoint(self, 'break here', lldb.SBFileSpec('main.c'))

        event = lldbutil.fetch_next_event(self, self.listener, self.broadcaster)
        ret_args = lldb.SBDebugger.GetProgressFromEvent(event)
        self.assertGreater(len(ret_args), 0)
        message = ret_args[0]
        self.assertGreater(len(message), 0)

