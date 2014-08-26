import unittest2

import gdbremote_testcase
import lldbgdbserverutils

from lldbtest import *

class TestGdbRemoteProcessInfo(gdbremote_testcase.GdbRemoteTestCaseBase):

    def qProcessInfo_returns_running_process(self):
        procs = self.prep_debug_monitor_and_inferior()
        self.add_process_info_collection_packets()

        # Run the stream
        context = self.expect_gdbremote_sequence()
        self.assertIsNotNone(context)

        # Gather process info response
        process_info = self.parse_process_info_response(context)
        self.assertIsNotNone(process_info)

        # Ensure the process id looks reasonable.
        pid_text = process_info.get("pid")
        self.assertIsNotNone(pid_text)
        pid = int(pid_text, base=16)
        self.assertNotEqual(0, pid)

        # If possible, verify that the process is running.
        self.assertTrue(lldbgdbserverutils.process_is_running(pid, True))

    @debugserver_test
    @dsym_test
    def test_qProcessInfo_returns_running_process_debugserver_dsym(self):
        self.init_debugserver_test()
        self.buildDsym()
        self.qProcessInfo_returns_running_process()

    @llgs_test
    @dwarf_test
    def test_qProcessInfo_returns_running_process_llgs_dwarf(self):
        self.init_llgs_test()
        self.buildDwarf()
        self.qProcessInfo_returns_running_process()

    def attach_commandline_qProcessInfo_reports_correct_pid(self):
        procs = self.prep_debug_monitor_and_inferior()
        self.assertIsNotNone(procs)
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
        self.assertEqual(reported_pid, procs["inferior"].pid)

    @debugserver_test
    @dsym_test
    def test_attach_commandline_qProcessInfo_reports_correct_pid_debugserver_dsym(self):
        self.init_debugserver_test()
        self.buildDsym()
        self.set_inferior_startup_attach()
        self.attach_commandline_qProcessInfo_reports_correct_pid()

    @llgs_test
    @dwarf_test
    def test_attach_commandline_qProcessInfo_reports_correct_pid_llgs_dwarf(self):
        self.init_llgs_test()
        self.buildDwarf()
        self.set_inferior_startup_attach()
        self.attach_commandline_qProcessInfo_reports_correct_pid()

    def qProcessInfo_reports_valid_endian(self):
        procs = self.prep_debug_monitor_and_inferior()
        self.add_process_info_collection_packets()

        # Run the stream
        context = self.expect_gdbremote_sequence()
        self.assertIsNotNone(context)

        # Gather process info response
        process_info = self.parse_process_info_response(context)
        self.assertIsNotNone(process_info)

        # Ensure the process id looks reasonable.
        endian = process_info.get("endian")
        self.assertIsNotNone(endian)
        self.assertTrue(endian in ["little", "big", "pdp"])

    @debugserver_test
    @dsym_test
    def test_qProcessInfo_reports_valid_endian_debugserver_dsym(self):
        self.init_debugserver_test()
        self.buildDsym()
        self.qProcessInfo_reports_valid_endian()

    @llgs_test
    @dwarf_test
    def test_qProcessInfo_reports_valid_endian_llgs_dwarf(self):
        self.init_llgs_test()
        self.buildDwarf()
        self.qProcessInfo_reports_valid_endian()


if __name__ == '__main__':
    unittest2.main()
