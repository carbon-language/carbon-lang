import unittest2

# Add the directory above ours to the python library path since we
# will import from there.
import os.path
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import gdbremote_testcase
import signal
from lldbtest import *

class TestGdbRemoteSegFault(gdbremote_testcase.GdbRemoteTestCaseBase):
    mydir = TestBase.compute_mydir(__file__)

    GDB_REMOTE_STOP_CODE_BAD_ACCESS = 0x91

    def inferior_seg_fault_received(self, expected_signo):
        procs = self.prep_debug_monitor_and_inferior(inferior_args=["segfault"])
        self.assertIsNotNone(procs)

        self.test_sequence.add_log_lines([
            "read packet: $vCont;c#a8",
            {"direction":"send", "regex":r"^\$T([0-9a-fA-F]{2}).*#[0-9a-fA-F]{2}$", "capture":{ 1:"hex_exit_code"} },
            ], True)

        context = self.expect_gdbremote_sequence()
        self.assertIsNotNone(context)

        hex_exit_code = context.get("hex_exit_code")
        self.assertIsNotNone(hex_exit_code)
        self.assertEquals(int(hex_exit_code, 16), expected_signo)

    @debugserver_test
    @dsym_test
    def test_inferior_seg_fault_received_debugserver_dsym(self):
        self.init_debugserver_test()
        self.buildDsym()
        self.inferior_seg_fault_received(self.GDB_REMOTE_STOP_CODE_BAD_ACCESS)

    @llgs_test
    @dwarf_test
    def test_inferior_seg_fault_received_llgs_dwarf(self):
        self.init_llgs_test()
        self.buildDwarf()
        self.inferior_seg_fault_received(signal.SIGSEGV)
