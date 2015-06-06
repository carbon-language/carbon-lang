import unittest2

import gdbremote_testcase
import signal
from lldbtest import *

class TestGdbRemoteAbort(gdbremote_testcase.GdbRemoteTestCaseBase):
    mydir = TestBase.compute_mydir(__file__)

    def inferior_abort_received(self):
        procs = self.prep_debug_monitor_and_inferior(inferior_args=["abort"])
        self.assertIsNotNone(procs)

        self.test_sequence.add_log_lines([
            "read packet: $vCont;c#a8",
            {"direction":"send", "regex":r"^\$T([0-9a-fA-F]{2}).*#[0-9a-fA-F]{2}$", "capture":{ 1:"hex_exit_code"} },
            ], True)

        context = self.expect_gdbremote_sequence()
        self.assertIsNotNone(context)

        hex_exit_code = context.get("hex_exit_code")
        self.assertIsNotNone(hex_exit_code)
        self.assertEquals(int(hex_exit_code, 16),
                          lldbutil.get_signal_number('SIGABRT'))

    @debugserver_test
    @dsym_test
    def test_inferior_abort_received_debugserver_dsym(self):
        self.init_debugserver_test()
        self.buildDsym()
        self.inferior_abort_received()

    @llgs_test
    @dwarf_test
    @skipIfTargetAndroid(api_levels=[16]) # abort() on API 16 raises SIGSEGV!
    def test_inferior_abort_received_llgs_dwarf(self):
        self.init_llgs_test()
        self.buildDwarf()
        self.inferior_abort_received()

