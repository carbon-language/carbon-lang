import unittest2

import gdbremote_testcase
import lldbgdbserverutils

from lldbtest import *

class TestGdbRemoteKill(gdbremote_testcase.GdbRemoteTestCaseBase):
    def attach_commandline_kill_after_initial_stop(self):
        procs = self.prep_debug_monitor_and_inferior()
        self.test_sequence.add_log_lines([
            "read packet: $k#6b",
            {"direction":"send", "regex":r"^\$X[0-9a-fA-F]+([^#]*)#[0-9A-Fa-f]{2}" },
            ], True)

        if self.stub_sends_two_stop_notifications_on_kill:
            # Add an expectation for a second X result for stubs that send two of these.
            self.test_sequence.add_log_lines([
                {"direction":"send", "regex":r"^\$X[0-9a-fA-F]+([^#]*)#[0-9A-Fa-f]{2}" },
                ], True)

        self.expect_gdbremote_sequence()

        # Wait a moment for completed and now-detached inferior process to clear.
        time.sleep(1)

        # Process should be dead now.  Reap results.
        poll_result = procs["inferior"].poll()
        self.assertIsNotNone(poll_result)

        # Where possible, verify at the system level that the process is not running.
        self.assertFalse(lldbgdbserverutils.process_is_running(procs["inferior"].pid, False))

    @debugserver_test
    @dsym_test
    def test_attach_commandline_kill_after_initial_stop_debugserver_dsym(self):
        self.init_debugserver_test()
        self.buildDsym()
        self.set_inferior_startup_attach()
        self.attach_commandline_kill_after_initial_stop()

    @llgs_test
    @dwarf_test
    def test_attach_commandline_kill_after_initial_stop_llgs_dwarf(self):
        self.init_llgs_test()
        self.buildDwarf()
        self.set_inferior_startup_attach()
        self.attach_commandline_kill_after_initial_stop()

