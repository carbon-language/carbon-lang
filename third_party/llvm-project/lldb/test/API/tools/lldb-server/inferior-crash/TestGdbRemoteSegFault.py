import gdbremote_testcase
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil

class TestGdbRemoteSegFault(gdbremote_testcase.GdbRemoteTestCaseBase):
    mydir = TestBase.compute_mydir(__file__)

    GDB_REMOTE_STOP_CODE_BAD_ACCESS = 0x91

    def inferior_seg_fault_received(self, expected_signo):
        procs = self.prep_debug_monitor_and_inferior(
            inferior_args=["segfault"])
        self.assertIsNotNone(procs)

        self.test_sequence.add_log_lines(["read packet: $vCont;c#a8",
                                          {"direction": "send",
                                           "regex": r"^\$T([0-9a-fA-F]{2}).*#[0-9a-fA-F]{2}$",
                                           "capture": {1: "hex_exit_code"}},
                                          ],
                                         True)

        context = self.expect_gdbremote_sequence()
        self.assertIsNotNone(context)

        hex_exit_code = context.get("hex_exit_code")
        self.assertIsNotNone(hex_exit_code)
        self.assertEqual(int(hex_exit_code, 16), expected_signo)

    @skipIfWindows # No signal is sent on Windows.
    def test_inferior_seg_fault_received(self):
        self.build()
        if self.platformIsDarwin():
            self.inferior_seg_fault_received(self.GDB_REMOTE_STOP_CODE_BAD_ACCESS)
        else:
            self.inferior_seg_fault_received(lldbutil.get_signal_number('SIGSEGV'))
