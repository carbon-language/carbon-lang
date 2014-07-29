import gdbremote_testcase
import lldbgdbserverutils
import unittest2

from lldbtest import *

class TestGdbRemoteAttach(gdbremote_testcase.GdbRemoteTestCaseBase):

    def attach_with_vAttach(self):
        # Start the inferior, start the debug monitor, nothing is attached yet.
        procs = self.prep_debug_monitor_and_inferior(inferior_args=["sleep:60"])
        self.assertIsNotNone(procs)

        # Make sure the target process has been launched.
        inferior = procs.get("inferior")
        self.assertIsNotNone(inferior)
        self.assertTrue(inferior.pid > 0)
        self.assertTrue(lldbgdbserverutils.process_is_running(inferior.pid, True))

        # Add attach packets.
        self.test_sequence.add_log_lines([
            # Do the attach.
            "read packet: $vAttach;{:x}#00".format(inferior.pid),
            # Expect a stop notification from the attach.
            { "direction":"send", "regex":r"^\$T([0-9a-fA-F]{2})[^#]*#[0-9a-fA-F]{2}$", "capture":{1:"stop_signal_hex"} },
            ], True)
        self.add_process_info_collection_packets()

        # Run the stream
        context = self.expect_gdbremote_sequence()
        self.assertIsNotNone(context)

        # Gather process info response
        process_info = self.parse_process_info_response(context)
        self.assertIsNotNone(process_info)

        # Ensure the process id matches what we expected.
        pid_text = process_info.get('pid', None)
        self.assertIsNotNone(pid_text)
        reported_pid = int(pid_text, base=16)
        self.assertEqual(reported_pid, inferior.pid)

    @debugserver_test
    @dsym_test
    def test_attach_with_vAttach_debugserver_dsym(self):
        self.init_debugserver_test()
        self.buildDsym()
        self.set_inferior_startup_attach_manually()
        self.attach_with_vAttach()

    @llgs_test
    @dwarf_test
    def test_attach_with_vAttach_llgs_dwarf(self):
        self.init_llgs_test()
        self.buildDwarf()
        self.set_inferior_startup_attach_manually()
        self.attach_with_vAttach()


if __name__ == '__main__':
    unittest2.main()
import gdbremote_testcase
import lldbgdbserverutils
import unittest2

from lldbtest import *

class TestGdbRemoteAttach(gdbremote_testcase.GdbRemoteTestCaseBase):

    def attach_with_vAttach(self):
        # Start the inferior, start the debug monitor, nothing is attached yet.
        procs = self.prep_debug_monitor_and_inferior(inferior_args=["sleep:60"])
        self.assertIsNotNone(procs)

        # Make sure the target process has been launched.
        inferior = procs.get("inferior")
        self.assertIsNotNone(inferior)
        self.assertTrue(inferior.pid > 0)
        self.assertTrue(lldbgdbserverutils.process_is_running(inferior.pid, True))

        # Add attach packets.
        self.test_sequence.add_log_lines([
            # Do the attach.
            "read packet: $vAttach;{:x}#00".format(inferior.pid),
            # Expect a stop notification from the attach.
            { "direction":"send", "regex":r"^\$T([0-9a-fA-F]{2})[^#]*#[0-9a-fA-F]{2}$", "capture":{1:"stop_signal_hex"} },
            ], True)
        self.add_process_info_collection_packets()

        # Run the stream
        context = self.expect_gdbremote_sequence()
        self.assertIsNotNone(context)

        # Gather process info response
        process_info = self.parse_process_info_response(context)
        self.assertIsNotNone(process_info)

        # Ensure the process id matches what we expected.
        pid_text = process_info.get('pid', None)
        self.assertIsNotNone(pid_text)
        reported_pid = int(pid_text, base=16)
        self.assertEqual(reported_pid, inferior.pid)

    @debugserver_test
    @dsym_test
    def test_attach_with_vAttach_debugserver_dsym(self):
        self.init_debugserver_test()
        self.buildDsym()
        self.set_inferior_startup_attach_manually()
        self.attach_with_vAttach()

    @llgs_test
    @dwarf_test
    def test_attach_with_vAttach_llgs_dwarf(self):
        self.init_llgs_test()
        self.buildDwarf()
        self.set_inferior_startup_attach_manually()
        self.attach_with_vAttach()


if __name__ == '__main__':
    unittest2.main()
