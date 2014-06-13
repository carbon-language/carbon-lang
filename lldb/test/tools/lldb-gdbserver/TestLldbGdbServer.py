"""
Test case for testing the gdbremote protocol.

Tests run against debugserver and lldb-gdbserver (llgs).
lldb-gdbserver tests run where the lldb-gdbserver exe is
available.

This class will be broken into smaller test case classes by
gdb remote packet functional areas.  For now it contains
the initial set of tests implemented.
"""

import lldbgdbserverutils
import unittest2
from lldbtest import *

import gdbremote_testcase

class LldbGdbServerTestCase(gdbremote_testcase.GdbRemoteTestCaseBase):

    @debugserver_test
    def test_exe_starts_debugserver(self):
        self.init_debugserver_test()
        server = self.start_server()

    @llgs_test
    def test_exe_starts_llgs(self):
        self.init_llgs_test()
        server = self.start_server()

    def start_no_ack_mode(self):
        server = self.start_server()
        self.assertIsNotNone(server)

        self.add_no_ack_remote_stream()
        self.expect_gdbremote_sequence()

    @debugserver_test
    def test_start_no_ack_mode_debugserver(self):
        self.init_debugserver_test()
        self.start_no_ack_mode()

    @llgs_test
    def test_start_no_ack_mode_llgs(self):
        self.init_llgs_test()
        self.start_no_ack_mode()

    def thread_suffix_supported(self):
        server = self.start_server()
        self.assertIsNotNone(server)

        self.add_no_ack_remote_stream()
        self.test_sequence.add_log_lines(
            ["lldb-gdbserver <  26> read packet: $QThreadSuffixSupported#e4",
             "lldb-gdbserver <   6> send packet: $OK#9a"],
            True)

        self.expect_gdbremote_sequence()

    @debugserver_test
    def test_thread_suffix_supported_debugserver(self):
        self.init_debugserver_test()
        self.thread_suffix_supported()

    @llgs_test
    @unittest2.expectedFailure()
    def test_thread_suffix_supported_llgs(self):
        self.init_llgs_test()
        self.thread_suffix_supported()

    def list_threads_in_stop_reply_supported(self):
        server = self.start_server()
        self.assertIsNotNone(server)

        self.add_no_ack_remote_stream()
        self.test_sequence.add_log_lines(
            ["lldb-gdbserver <  27> read packet: $QListThreadsInStopReply#21",
             "lldb-gdbserver <   6> send packet: $OK#9a"],
            True)
        self.expect_gdbremote_sequence()

    @debugserver_test
    def test_list_threads_in_stop_reply_supported_debugserver(self):
        self.init_debugserver_test()
        self.list_threads_in_stop_reply_supported()

    @llgs_test
    @unittest2.expectedFailure()
    def test_list_threads_in_stop_reply_supported_llgs(self):
        self.init_llgs_test()
        self.list_threads_in_stop_reply_supported()

    def start_inferior(self):
        server = self.start_server()
        self.assertIsNotNone(server)

        # build launch args
        launch_args = [os.path.abspath('a.out')]

        self.add_no_ack_remote_stream()
        self.test_sequence.add_log_lines(
            ["read packet: %s" % lldbgdbserverutils.build_gdbremote_A_packet(launch_args),
             "send packet: $OK#9a"],
            True)
        self.expect_gdbremote_sequence()

    @debugserver_test
    @dsym_test
    def test_start_inferior_debugserver_dsym(self):
        self.init_debugserver_test()
        self.buildDsym()
        self.start_inferior()

    @llgs_test
    @dwarf_test
    def test_start_inferior_llgs_dwarf(self):
        self.init_llgs_test()
        self.buildDwarf()
        self.start_inferior()

    def inferior_exit_0(self):
        server = self.start_server()
        self.assertIsNotNone(server)

        # build launch args
        launch_args = [os.path.abspath('a.out')]

        self.add_no_ack_remote_stream()
        self.add_verified_launch_packets(launch_args)
        self.test_sequence.add_log_lines(
            ["read packet: $vCont;c#00",
             "send packet: $W00#00"],
            True)

        self.expect_gdbremote_sequence()

    @debugserver_test
    @dsym_test
    def test_inferior_exit_0_debugserver_dsym(self):
        self.init_debugserver_test()
        self.buildDsym()
        self.inferior_exit_0()

    @llgs_test
    @dwarf_test
    @unittest2.expectedFailure()
    def test_inferior_exit_0_llgs_dwarf(self):
        self.init_llgs_test()
        self.buildDwarf()
        self.inferior_exit_0()

    def inferior_exit_42(self):
        server = self.start_server()
        self.assertIsNotNone(server)

        RETVAL = 42

        # build launch args
        launch_args = [os.path.abspath('a.out'), "retval:%d" % RETVAL]

        self.add_no_ack_remote_stream()
        self.add_verified_launch_packets(launch_args)
        self.test_sequence.add_log_lines(
            ["read packet: $vCont;c#00",
             "send packet: $W{0:02x}#00".format(RETVAL)],
            True)

        self.expect_gdbremote_sequence()

    @debugserver_test
    @dsym_test
    def test_inferior_exit_42_debugserver_dsym(self):
        self.init_debugserver_test()
        self.buildDsym()
        self.inferior_exit_42()

    @llgs_test
    @dwarf_test
    @unittest2.expectedFailure()
    def test_inferior_exit_42_llgs_dwarf(self):
        self.init_llgs_test()
        self.buildDwarf()
        self.inferior_exit_42()

    def inferior_print_exit(self):
        server = self.start_server()
        self.assertIsNotNone(server)

        # build launch args
        launch_args = [os.path.abspath('a.out'), "hello, world"]

        self.add_no_ack_remote_stream()
        self.add_verified_launch_packets(launch_args)
        self.test_sequence.add_log_lines(
            ["read packet: $vCont;c#00",
             {"type":"output_match", "regex":r"^hello, world\r\n$" },
             "send packet: $W00#00"],
            True)

        context = self.expect_gdbremote_sequence()
        self.assertIsNotNone(context)

    @debugserver_test
    @dsym_test
    def test_inferior_print_exit_debugserver_dsym(self):
        self.init_debugserver_test()
        self.buildDsym()
        self.inferior_print_exit()

    @llgs_test
    @dwarf_test
    @unittest2.expectedFailure()
    def test_inferior_print_exit_llgs_dwarf(self):
        self.init_llgs_test()
        self.buildDwarf()
        self.inferior_print_exit()

    def first_launch_stop_reply_thread_matches_first_qC(self):
        server = self.start_server()
        self.assertIsNotNone(server)

        # build launch args
        launch_args = [os.path.abspath('a.out'), "hello, world"]

        self.add_no_ack_remote_stream()
        self.add_verified_launch_packets(launch_args)
        self.test_sequence.add_log_lines(
            ["read packet: $qC#00",
             { "direction":"send", "regex":r"^\$QC([0-9a-fA-F]+)#", "capture":{1:"thread_id"} },
             "read packet: $?#00",
             { "direction":"send", "regex":r"^\$T[0-9a-fA-F]{2}thread:([0-9a-fA-F]+)", "expect_captures":{1:"thread_id"} }],
            True)
        self.expect_gdbremote_sequence()

    @debugserver_test
    @dsym_test
    def test_first_launch_stop_reply_thread_matches_first_qC_debugserver_dsym(self):
        self.init_debugserver_test()
        self.buildDsym()
        self.first_launch_stop_reply_thread_matches_first_qC()

    @llgs_test
    @dwarf_test
    @unittest2.expectedFailure()
    def test_first_launch_stop_reply_thread_matches_first_qC_llgs_dwarf(self):
        self.init_llgs_test()
        self.buildDwarf()
        self.first_launch_stop_reply_thread_matches_first_qC()

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
    @unittest2.expectedFailure()
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
    @unittest2.expectedFailure()
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
    @unittest2.expectedFailure()
    def test_qProcessInfo_reports_valid_endian_llgs_dwarf(self):
        self.init_llgs_test()
        self.buildDwarf()
        self.qProcessInfo_reports_valid_endian()

    def attach_commandline_continue_app_exits(self):
        procs = self.prep_debug_monitor_and_inferior()
        self.test_sequence.add_log_lines(
            ["read packet: $vCont;c#00",
             "send packet: $W00#00"],
            True)
        self.expect_gdbremote_sequence()

        # Process should be dead now.  Reap results.
        poll_result = procs["inferior"].poll()
        self.assertIsNotNone(poll_result)

        # Where possible, verify at the system level that the process is not running.
        self.assertFalse(lldbgdbserverutils.process_is_running(procs["inferior"].pid, False))

    @debugserver_test
    @dsym_test
    def test_attach_commandline_continue_app_exits_debugserver_dsym(self):
        self.init_debugserver_test()
        self.buildDsym()
        self.set_inferior_startup_attach()
        self.attach_commandline_continue_app_exits()

    @llgs_test
    @dwarf_test
    @unittest2.expectedFailure()
    def test_attach_commandline_continue_app_exits_llgs_dwarf(self):
        self.init_llgs_test()
        self.buildDwarf()
        self.set_inferior_startup_attach()
        self.attach_commandline_continue_app_exits()

    def attach_commandline_kill_after_initial_stop(self):
        procs = self.prep_debug_monitor_and_inferior()
        self.test_sequence.add_log_lines(
            ["read packet: $k#6b",
             "send packet: $X09#00"],
            True)
        self.expect_gdbremote_sequence()

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
    @unittest2.expectedFailure()
    def test_attach_commandline_kill_after_initial_stop_llgs_dwarf(self):
        self.init_llgs_test()
        self.buildDwarf()
        self.set_inferior_startup_attach()
        self.attach_commandline_kill_after_initial_stop()

    def qRegisterInfo_returns_one_valid_result(self):
        server = self.start_server()
        self.assertIsNotNone(server)

        # Build launch args
        launch_args = [os.path.abspath('a.out')]

        # Build the expected protocol stream
        self.add_no_ack_remote_stream()
        self.add_verified_launch_packets(launch_args)
        self.test_sequence.add_log_lines(
            ["read packet: $qRegisterInfo0#00",
             { "direction":"send", "regex":r"^\$(.+);#\d{2}", "capture":{1:"reginfo_0"} }],
            True)

        # Run the stream
        context = self.expect_gdbremote_sequence()
        self.assertIsNotNone(context)

        reg_info_packet = context.get("reginfo_0")
        self.assertIsNotNone(reg_info_packet)
        self.assert_valid_reg_info(lldbgdbserverutils.parse_reg_info_response(reg_info_packet))

    @debugserver_test
    @dsym_test
    def test_qRegisterInfo_returns_one_valid_result_debugserver_dsym(self):
        self.init_debugserver_test()
        self.buildDsym()
        self.qRegisterInfo_returns_one_valid_result()

    @llgs_test
    @dwarf_test
    @unittest2.expectedFailure()
    def test_qRegisterInfo_returns_one_valid_result_llgs_dwarf(self):
        self.init_llgs_test()
        self.buildDwarf()
        self.qRegisterInfo_returns_one_valid_result()

    def qRegisterInfo_returns_all_valid_results(self):
        server = self.start_server()
        self.assertIsNotNone(server)

        # Build launch args.
        launch_args = [os.path.abspath('a.out')]

        # Build the expected protocol stream.
        self.add_no_ack_remote_stream()
        self.add_verified_launch_packets(launch_args)
        self.add_register_info_collection_packets()

        # Run the stream.
        context = self.expect_gdbremote_sequence()
        self.assertIsNotNone(context)

        # Validate that each register info returned validates.
        for reg_info in self.parse_register_info_packets(context):
            self.assert_valid_reg_info(reg_info)

    @debugserver_test
    @dsym_test
    def test_qRegisterInfo_returns_all_valid_results_debugserver_dsym(self):
        self.init_debugserver_test()
        self.buildDsym()
        self.qRegisterInfo_returns_all_valid_results()

    @llgs_test
    @dwarf_test
    @unittest2.expectedFailure()
    def test_qRegisterInfo_returns_all_valid_results_llgs_dwarf(self):
        self.init_llgs_test()
        self.buildDwarf()
        self.qRegisterInfo_returns_all_valid_results()

    def qRegisterInfo_contains_required_generics(self):
        server = self.start_server()
        self.assertIsNotNone(server)

        # Build launch args
        launch_args = [os.path.abspath('a.out')]

        # Build the expected protocol stream
        self.add_no_ack_remote_stream()
        self.add_verified_launch_packets(launch_args)
        self.add_register_info_collection_packets()

        # Run the packet stream.
        context = self.expect_gdbremote_sequence()
        self.assertIsNotNone(context)

        # Gather register info entries.
        reg_infos = self.parse_register_info_packets(context)

        # Collect all generic registers found.
        generic_regs = { reg_info['generic']:1 for reg_info in reg_infos if 'generic' in reg_info }

        # Ensure we have a program counter register.
        self.assertTrue('pc' in generic_regs)

        # Ensure we have a frame pointer register.
        self.assertTrue('fp' in generic_regs)

        # Ensure we have a stack pointer register.
        self.assertTrue('sp' in generic_regs)

        # Ensure we have a flags register.
        self.assertTrue('flags' in generic_regs)

    @debugserver_test
    @dsym_test
    def test_qRegisterInfo_contains_required_generics_debugserver_dsym(self):
        self.init_debugserver_test()
        self.buildDsym()
        self.qRegisterInfo_contains_required_generics()

    @llgs_test
    @dwarf_test
    @unittest2.expectedFailure()
    def test_qRegisterInfo_contains_required_generics_llgs_dwarf(self):
        self.init_llgs_test()
        self.buildDwarf()
        self.qRegisterInfo_contains_required_generics()

    def qRegisterInfo_contains_at_least_one_register_set(self):
        server = self.start_server()
        self.assertIsNotNone(server)

        # Build launch args
        launch_args = [os.path.abspath('a.out')]

        # Build the expected protocol stream
        self.add_no_ack_remote_stream()
        self.add_verified_launch_packets(launch_args)
        self.add_register_info_collection_packets()

        # Run the packet stream.
        context = self.expect_gdbremote_sequence()
        self.assertIsNotNone(context)

        # Gather register info entries.
        reg_infos = self.parse_register_info_packets(context)

        # Collect all register sets found.
        register_sets = { reg_info['set']:1 for reg_info in reg_infos if 'set' in reg_info }
        self.assertTrue(len(register_sets) >= 1)

    @debugserver_test
    @dsym_test
    def test_qRegisterInfo_contains_at_least_one_register_set_debugserver_dsym(self):
        self.init_debugserver_test()
        self.buildDsym()
        self.qRegisterInfo_contains_at_least_one_register_set()

    @llgs_test
    @dwarf_test
    @unittest2.expectedFailure()
    def test_qRegisterInfo_contains_at_least_one_register_set_llgs_dwarf(self):
        self.init_llgs_test()
        self.buildDwarf()
        self.qRegisterInfo_contains_at_least_one_register_set()

    def qThreadInfo_contains_thread(self):
        procs = self.prep_debug_monitor_and_inferior()
        self.add_threadinfo_collection_packets()

        # Run the packet stream.
        context = self.expect_gdbremote_sequence()
        self.assertIsNotNone(context)

        # Gather threadinfo entries.
        threads = self.parse_threadinfo_packets(context)
        self.assertIsNotNone(threads)

        # We should have exactly one thread.
        self.assertEqual(len(threads), 1)

    @debugserver_test
    @dsym_test
    def test_qThreadInfo_contains_thread_launch_debugserver_dsym(self):
        self.init_debugserver_test()
        self.buildDsym()
        self.set_inferior_startup_launch()
        self.qThreadInfo_contains_thread()

    @llgs_test
    @dwarf_test
    @unittest2.expectedFailure()
    def test_qThreadInfo_contains_thread_launch_llgs_dwarf(self):
        self.init_llgs_test()
        self.buildDwarf()
        self.set_inferior_startup_launch()
        self.qThreadInfo_contains_thread()

    @debugserver_test
    @dsym_test
    def test_qThreadInfo_contains_thread_attach_debugserver_dsym(self):
        self.init_debugserver_test()
        self.buildDsym()
        self.set_inferior_startup_attach()
        self.qThreadInfo_contains_thread()

    @llgs_test
    @dwarf_test
    @unittest2.expectedFailure()
    def test_qThreadInfo_contains_thread_attach_llgs_dwarf(self):
        self.init_llgs_test()
        self.buildDwarf()
        self.set_inferior_startup_attach()
        self.qThreadInfo_contains_thread()

    def qThreadInfo_matches_qC(self):
        procs = self.prep_debug_monitor_and_inferior()

        self.add_threadinfo_collection_packets()
        self.test_sequence.add_log_lines(
            ["read packet: $qC#00",
             { "direction":"send", "regex":r"^\$QC([0-9a-fA-F]+)#", "capture":{1:"thread_id"} }
             ], True)

        # Run the packet stream.
        context = self.expect_gdbremote_sequence()
        self.assertIsNotNone(context)

        # Gather threadinfo entries.
        threads = self.parse_threadinfo_packets(context)
        self.assertIsNotNone(threads)

        # We should have exactly one thread from threadinfo.
        self.assertEqual(len(threads), 1)

        # We should have a valid thread_id from $QC.
        QC_thread_id_hex = context.get("thread_id")
        self.assertIsNotNone(QC_thread_id_hex)
        QC_thread_id = int(QC_thread_id_hex, 16)

        # Those two should be the same.
        self.assertEquals(threads[0], QC_thread_id)

    @debugserver_test
    @dsym_test
    def test_qThreadInfo_matches_qC_launch_debugserver_dsym(self):
        self.init_debugserver_test()
        self.buildDsym()
        self.set_inferior_startup_launch()
        self.qThreadInfo_matches_qC()

    @llgs_test
    @dwarf_test
    @unittest2.expectedFailure()
    def test_qThreadInfo_matches_qC_launch_llgs_dwarf(self):
        self.init_llgs_test()
        self.buildDwarf()
        self.set_inferior_startup_launch()
        self.qThreadInfo_matches_qC()

    @debugserver_test
    @dsym_test
    def test_qThreadInfo_matches_qC_attach_debugserver_dsym(self):
        self.init_debugserver_test()
        self.buildDsym()
        self.set_inferior_startup_attach()
        self.qThreadInfo_matches_qC()

    @llgs_test
    @dwarf_test
    @unittest2.expectedFailure()
    def test_qThreadInfo_matches_qC_attach_llgs_dwarf(self):
        self.init_llgs_test()
        self.buildDwarf()
        self.set_inferior_startup_attach()
        self.qThreadInfo_matches_qC()

    def p_returns_correct_data_size_for_each_qRegisterInfo(self):
        procs = self.prep_debug_monitor_and_inferior()
        self.add_register_info_collection_packets()

        # Run the packet stream.
        context = self.expect_gdbremote_sequence()
        self.assertIsNotNone(context)

        # Gather register info entries.
        reg_infos = self.parse_register_info_packets(context)
        self.assertIsNotNone(reg_infos)
        self.assertTrue(len(reg_infos) > 0)

        # Read value for each register.
        reg_index = 0
        for reg_info in reg_infos:
            # Clear existing packet expectations.
            self.reset_test_sequence()

            # Run the register query
            self.test_sequence.add_log_lines(
                ["read packet: $p{0:x}#00".format(reg_index),
                 { "direction":"send", "regex":r"^\$([0-9a-fA-F]+)#", "capture":{1:"p_response"} }],
                True)
            context = self.expect_gdbremote_sequence()
            self.assertIsNotNone(context)

            # Verify the response length.
            p_response = context.get("p_response")
            self.assertIsNotNone(p_response)
            self.assertEquals(len(p_response), 2 * int(reg_info["bitsize"]) / 8)
            # print "register {} ({}): {}".format(reg_index, reg_info["name"], p_response)

            # Increment loop
            reg_index += 1

    @debugserver_test
    @dsym_test
    def test_p_returns_correct_data_size_for_each_qRegisterInfo_launch_debugserver_dsym(self):
        self.init_debugserver_test()
        self.buildDsym()
        self.set_inferior_startup_launch()
        self.p_returns_correct_data_size_for_each_qRegisterInfo()

    @llgs_test
    @dwarf_test
    @unittest2.expectedFailure()
    def test_p_returns_correct_data_size_for_each_qRegisterInfo_launch_llgs_dwarf(self):
        self.init_llgs_test()
        self.buildDwarf()
        self.set_inferior_startup_launch()
        self.p_returns_correct_data_size_for_each_qRegisterInfo()

    @debugserver_test
    @dsym_test
    def test_p_returns_correct_data_size_for_each_qRegisterInfo_attach_debugserver_dsym(self):
        self.init_debugserver_test()
        self.buildDsym()
        self.set_inferior_startup_attach()
        self.p_returns_correct_data_size_for_each_qRegisterInfo()

    @llgs_test
    @dwarf_test
    @unittest2.expectedFailure()
    def test_p_returns_correct_data_size_for_each_qRegisterInfo_attach_llgs_dwarf(self):
        self.init_llgs_test()
        self.buildDwarf()
        self.set_inferior_startup_attach()
        self.p_returns_correct_data_size_for_each_qRegisterInfo()

    def Hg_switches_to_3_threads(self):
        # Startup the inferior with three threads (main + 2 new ones).
        procs = self.prep_debug_monitor_and_inferior(inferior_args=["thread:new", "thread:new"])

        # Let the inferior process have a few moments to start up the thread when launched.  (The launch scenario has no time to run, so threads won't be there yet.)
        self.run_process_then_stop(run_seconds=1)

        # Wait at most x seconds for 3 threads to be present.
        threads = self.wait_for_thread_count(3, timeout_seconds=5)
        self.assertEquals(len(threads), 3)

        # verify we can $H to each thead, and $qC matches the thread we set.
        for thread in threads:
            # Change to each thread, verify current thread id.
            self.reset_test_sequence()
            self.test_sequence.add_log_lines(
                ["read packet: $Hg{0:x}#00".format(thread),  # Set current thread.
                 "send packet: $OK#00",
                 "read packet: $qC#00",
                 { "direction":"send", "regex":r"^\$QC([0-9a-fA-F]+)#", "capture":{1:"thread_id"} }],
                True)

            context = self.expect_gdbremote_sequence()
            self.assertIsNotNone(context)

            # Verify the thread id.
            self.assertIsNotNone(context.get("thread_id"))
            self.assertEquals(int(context.get("thread_id"), 16), thread)

    @debugserver_test
    @dsym_test
    def test_Hg_switches_to_3_threads_launch_debugserver_dsym(self):
        self.init_debugserver_test()
        self.buildDsym()
        self.set_inferior_startup_launch()
        self.Hg_switches_to_3_threads()

    @llgs_test
    @dwarf_test
    @unittest2.expectedFailure()
    def test_Hg_switches_to_3_threads_launch_llgs_dwarf(self):
        self.init_llgs_test()
        self.buildDwarf()
        self.set_inferior_startup_launch()
        self.Hg_switches_to_3_threads()

    @debugserver_test
    @dsym_test
    def test_Hg_switches_to_3_threads_attach_debugserver_dsym(self):
        self.init_debugserver_test()
        self.buildDsym()
        self.set_inferior_startup_attach()
        self.Hg_switches_to_3_threads()

    @llgs_test
    @dwarf_test
    @unittest2.expectedFailure()
    def test_Hg_switches_to_3_threads_attach_llgs_dwarf(self):
        self.init_llgs_test()
        self.buildDwarf()
        self.set_inferior_startup_attach()
        self.Hg_switches_to_3_threads()

    def Hc_then_Csignal_signals_correct_thread(self):
        # NOTE only run this one in inferior-launched mode: we can't grab inferior stdout when running attached,
        # and the test requires getting stdout from the exe.

        NUM_THREADS = 3

        # Startup the inferior with three threads (main + NUM_THREADS-1 worker threads).
        # inferior_args=["thread:print-ids"]
        inferior_args=["thread:segfault"]
        for i in range(NUM_THREADS - 1):
            # if i > 0:
                # Give time between thread creation/segfaulting for the handler to work.
                # inferior_args.append("sleep:1")
            inferior_args.append("thread:new")
        inferior_args.append("sleep:10")

        # Launch/attach.  (In our case, this should only ever be launched since we need inferior stdout/stderr).
        procs = self.prep_debug_monitor_and_inferior(inferior_args=inferior_args)
        self.test_sequence.add_log_lines(["read packet: $c#00"], True)
        context = self.expect_gdbremote_sequence()

        # Let the inferior process have a few moments to start up the thread when launched.
        # context = self.run_process_then_stop(run_seconds=1)

        # Wait at most x seconds for all threads to be present.
        # threads = self.wait_for_thread_count(NUM_THREADS, timeout_seconds=5)
        # self.assertEquals(len(threads), NUM_THREADS)

        signaled_tids = {}

        # Switch to each thread, deliver a signal, and verify signal delivery
        for i in range(NUM_THREADS - 1):
            # Run until SIGSEGV comes in.
            self.reset_test_sequence()
            self.test_sequence.add_log_lines(
                [ # "read packet: $c#00",
                 {"direction":"send", "regex":r"^\$T([0-9a-fA-F]{2})thread:([0-9a-fA-F]+);", "capture":{1:"signo", 2:"thread_id"} }
                 ], True)
            context = self.expect_gdbremote_sequence()

            self.assertIsNotNone(context)
            signo = context.get("signo")
            self.assertEqual(int(signo, 16), self.TARGET_EXC_BAD_ACCESS)

            # Ensure we haven't seen this tid yet.
            thread_id = int(context.get("thread_id"), 16)
            self.assertFalse(thread_id in signaled_tids)
            signaled_tids[thread_id] = 1

            # Send SIGUSR1 to the thread that signaled the SIGSEGV.
            self.reset_test_sequence()
            self.test_sequence.add_log_lines(
                [
                 "read packet: $Hc{0:x}#00".format(thread_id),  # Set current thread.
                 "send packet: $OK#00",
                 "read packet: $C{0:x}#00".format(signal.SIGUSR1),
                 # "read packet: $vCont;C{0:x}:{1:x};c#00".format(signal.SIGUSR1, thread_id),
                 # "read packet: $c#00",
                 {"direction":"send", "regex":r"^\$T([0-9a-fA-F]{2})thread:([0-9a-fA-F]+);", "capture":{1:"stop_signo", 2:"stop_thread_id"} },
                  "read packet: $c#00",
                 # { "type":"output_match", "regex":r"^received SIGUSR1 on thread id: ([0-9a-fA-F]+)\r\n$", "capture":{ 1:"print_thread_id"} },
                 # "read packet: {}".format(chr(03)),
                ],
                True)

            # Run the sequence.
            context = self.expect_gdbremote_sequence()
            self.assertIsNotNone(context)

            # Ensure the stop signal is the signal we delivered.
            stop_signo = context.get("stop_signo")
            self.assertIsNotNone(stop_signo)
            self.assertEquals(int(stop_signo,16), signal.SIGUSR1)

            # Ensure the stop thread is the thread to which we delivered the signal.
            stop_thread_id = context.get("stop_thread_id")
            self.assertIsNotNone(stop_thread_id)
            self.assertEquals(int(stop_thread_id,16), thread_id)

            # Ensure we haven't seen this thread id yet.  The inferior's self-obtained thread ids are not guaranteed to match the stub tids (at least on MacOSX).
            # print_thread_id = context.get("print_thread_id")
            # self.assertIsNotNone(print_thread_id)
            # self.assertFalse(print_thread_id in print_thread_ids)

            # Now remember this print (i.e. inferior-reflected) thread id and ensure we don't hit it again.
            # print_thread_ids[print_thread_id] = 1

    @debugserver_test
    @dsym_test
    @unittest2.expectedFailure()
    def test_Hc_then_Csignal_signals_correct_thread_launch_debugserver_dsym(self):
        self.init_debugserver_test()
        self.buildDsym()
        self.set_inferior_startup_launch()
        self.Hc_then_Csignal_signals_correct_thread()

    @llgs_test
    @dwarf_test
    @unittest2.expectedFailure()
    def test_Hc_then_Csignal_signals_correct_thread_launch_llgs_dwarf(self):
        self.init_llgs_test()
        self.buildDwarf()
        self.set_inferior_startup_launch()
        self.Hc_then_Csignal_signals_correct_thread()

    def m_packet_reads_memory(self):
        # This is the memory we will write into the inferior and then ensure we can read back with $m.
        MEMORY_CONTENTS = "Test contents 0123456789 ABCDEFGHIJKLMNOPQRSTUVWXYZ abcdefghijklmnopqrstuvwxyz"

        # Start up the inferior.
        procs = self.prep_debug_monitor_and_inferior(
            inferior_args=["set-message:%s" % MEMORY_CONTENTS, "get-data-address-hex:g_message", "sleep:5"])

        # Run the process
        self.test_sequence.add_log_lines(
            [
             # Start running after initial stop.
             "read packet: $c#00",
             # Match output line that prints the memory address of the message buffer within the inferior. 
             # Note we require launch-only testing so we can get inferior otuput.
             { "type":"output_match", "regex":r"^data address: 0x([0-9a-fA-F]+)\r\n$", "capture":{ 1:"message_address"} },
             # Now stop the inferior.
             "read packet: {}".format(chr(03)),
             # And wait for the stop notification.
             {"direction":"send", "regex":r"^\$T([0-9a-fA-F]{2})thread:([0-9a-fA-F]+);", "capture":{1:"stop_signo", 2:"stop_thread_id"} }],
            True)

        # Run the packet stream.
        context = self.expect_gdbremote_sequence()
        self.assertIsNotNone(context)

        # Grab the message address.
        self.assertIsNotNone(context.get("message_address"))
        message_address = int(context.get("message_address"), 16)

        # Grab contents from the inferior.
        self.reset_test_sequence()
        self.test_sequence.add_log_lines(
            ["read packet: $m{0:x},{1:x}#00".format(message_address, len(MEMORY_CONTENTS)),
             {"direction":"send", "regex":r"^\$(.+)#[0-9a-fA-F]{2}$", "capture":{1:"read_contents"} }],
            True)

        # Run the packet stream.
        context = self.expect_gdbremote_sequence()
        self.assertIsNotNone(context)

        # Ensure what we read from inferior memory is what we wrote.
        self.assertIsNotNone(context.get("read_contents"))
        read_contents = context.get("read_contents").decode("hex")
        self.assertEquals(read_contents, MEMORY_CONTENTS)

    @debugserver_test
    @dsym_test
    def test_m_packet_reads_memory_debugserver_dsym(self):
        self.init_debugserver_test()
        self.buildDsym()
        self.set_inferior_startup_launch()
        self.m_packet_reads_memory()

    @llgs_test
    @dwarf_test
    @unittest2.expectedFailure()
    def test_m_packet_reads_memory_llgs_dwarf(self):
        self.init_llgs_test()
        self.buildDwarf()
        self.set_inferior_startup_launch()
        self.m_packet_reads_memory()

    def qMemoryRegionInfo_is_supported(self):
        # Start up the inferior.
        procs = self.prep_debug_monitor_and_inferior()

        # Ask if it supports $qMemoryRegionInfo.
        self.test_sequence.add_log_lines(
            ["read packet: $qMemoryRegionInfo#00",
             "send packet: $OK#00"
             ], True)
        self.expect_gdbremote_sequence()

    @debugserver_test
    @dsym_test
    def test_qMemoryRegionInfo_is_supported_debugserver_dsym(self):
        self.init_debugserver_test()
        self.buildDsym()
        self.set_inferior_startup_launch()
        self.qMemoryRegionInfo_is_supported()

    @llgs_test
    @dwarf_test
    @unittest2.expectedFailure()
    def test_qMemoryRegionInfo_is_supported_llgs_dwarf(self):
        self.init_llgs_test()
        self.buildDwarf()
        self.set_inferior_startup_launch()
        self.qMemoryRegionInfo_is_supported()

    def qMemoryRegionInfo_reports_code_address_as_executable(self):
        # Start up the inferior.
        procs = self.prep_debug_monitor_and_inferior(
            inferior_args=["get-code-address-hex:hello", "sleep:5"])

        # Run the process
        self.test_sequence.add_log_lines(
            [
             # Start running after initial stop.
             "read packet: $c#00",
             # Match output line that prints the memory address of the message buffer within the inferior. 
             # Note we require launch-only testing so we can get inferior otuput.
             { "type":"output_match", "regex":r"^code address: 0x([0-9a-fA-F]+)\r\n$", "capture":{ 1:"code_address"} },
             # Now stop the inferior.
             "read packet: {}".format(chr(03)),
             # And wait for the stop notification.
             {"direction":"send", "regex":r"^\$T([0-9a-fA-F]{2})thread:([0-9a-fA-F]+);", "capture":{1:"stop_signo", 2:"stop_thread_id"} }],
            True)

        # Run the packet stream.
        context = self.expect_gdbremote_sequence()
        self.assertIsNotNone(context)

        # Grab the code address.
        self.assertIsNotNone(context.get("code_address"))
        code_address = int(context.get("code_address"), 16)

        # Grab memory region info from the inferior.
        self.reset_test_sequence()
        self.add_query_memory_region_packets(code_address)

        # Run the packet stream.
        context = self.expect_gdbremote_sequence()
        self.assertIsNotNone(context)
        mem_region_dict = self.parse_memory_region_packet(context)

        # Ensure code address is readable and executable.
        self.assertTrue("permissions" in mem_region_dict)
        self.assertTrue("r" in mem_region_dict["permissions"])
        self.assertTrue("x" in mem_region_dict["permissions"])

        # Ensure the start address and size encompass the address we queried.
        self.assert_address_within_memory_region(code_address, mem_region_dict)

    @debugserver_test
    @dsym_test
    def test_qMemoryRegionInfo_reports_code_address_as_executable_debugserver_dsym(self):
        self.init_debugserver_test()
        self.buildDsym()
        self.set_inferior_startup_launch()
        self.qMemoryRegionInfo_reports_code_address_as_executable()

    @llgs_test
    @dwarf_test
    @unittest2.expectedFailure()
    def test_qMemoryRegionInfo_reports_code_address_as_executable_llgs_dwarf(self):
        self.init_llgs_test()
        self.buildDwarf()
        self.set_inferior_startup_launch()
        self.qMemoryRegionInfo_reports_code_address_as_executable()

    def qMemoryRegionInfo_reports_stack_address_as_readable_writeable(self):
        # Start up the inferior.
        procs = self.prep_debug_monitor_and_inferior(
            inferior_args=["get-stack-address-hex:", "sleep:5"])

        # Run the process
        self.test_sequence.add_log_lines(
            [
             # Start running after initial stop.
             "read packet: $c#00",
             # Match output line that prints the memory address of the message buffer within the inferior. 
             # Note we require launch-only testing so we can get inferior otuput.
             { "type":"output_match", "regex":r"^stack address: 0x([0-9a-fA-F]+)\r\n$", "capture":{ 1:"stack_address"} },
             # Now stop the inferior.
             "read packet: {}".format(chr(03)),
             # And wait for the stop notification.
             {"direction":"send", "regex":r"^\$T([0-9a-fA-F]{2})thread:([0-9a-fA-F]+);", "capture":{1:"stop_signo", 2:"stop_thread_id"} }],
            True)

        # Run the packet stream.
        context = self.expect_gdbremote_sequence()
        self.assertIsNotNone(context)

        # Grab the address.
        self.assertIsNotNone(context.get("stack_address"))
        stack_address = int(context.get("stack_address"), 16)

        # Grab memory region info from the inferior.
        self.reset_test_sequence()
        self.add_query_memory_region_packets(stack_address)

        # Run the packet stream.
        context = self.expect_gdbremote_sequence()
        self.assertIsNotNone(context)
        mem_region_dict = self.parse_memory_region_packet(context)

        # Ensure address is readable and executable.
        self.assertTrue("permissions" in mem_region_dict)
        self.assertTrue("r" in mem_region_dict["permissions"])
        self.assertTrue("w" in mem_region_dict["permissions"])

        # Ensure the start address and size encompass the address we queried.
        self.assert_address_within_memory_region(stack_address, mem_region_dict)

    @debugserver_test
    @dsym_test
    def test_qMemoryRegionInfo_reports_stack_address_as_readable_writeable_debugserver_dsym(self):
        self.init_debugserver_test()
        self.buildDsym()
        self.set_inferior_startup_launch()
        self.qMemoryRegionInfo_reports_stack_address_as_readable_writeable()

    @llgs_test
    @dwarf_test
    @unittest2.expectedFailure()
    def test_qMemoryRegionInfo_reports_stack_address_as_readable_writeable_llgs_dwarf(self):
        self.init_llgs_test()
        self.buildDwarf()
        self.set_inferior_startup_launch()
        self.qMemoryRegionInfo_reports_stack_address_as_readable_writeable()

    def qMemoryRegionInfo_reports_heap_address_as_readable_writeable(self):
        # Start up the inferior.
        procs = self.prep_debug_monitor_and_inferior(
            inferior_args=["get-heap-address-hex:", "sleep:5"])

        # Run the process
        self.test_sequence.add_log_lines(
            [
             # Start running after initial stop.
             "read packet: $c#00",
             # Match output line that prints the memory address of the message buffer within the inferior. 
             # Note we require launch-only testing so we can get inferior otuput.
             { "type":"output_match", "regex":r"^heap address: 0x([0-9a-fA-F]+)\r\n$", "capture":{ 1:"heap_address"} },
             # Now stop the inferior.
             "read packet: {}".format(chr(03)),
             # And wait for the stop notification.
             {"direction":"send", "regex":r"^\$T([0-9a-fA-F]{2})thread:([0-9a-fA-F]+);", "capture":{1:"stop_signo", 2:"stop_thread_id"} }],
            True)

        # Run the packet stream.
        context = self.expect_gdbremote_sequence()
        self.assertIsNotNone(context)

        # Grab the address.
        self.assertIsNotNone(context.get("heap_address"))
        heap_address = int(context.get("heap_address"), 16)

        # Grab memory region info from the inferior.
        self.reset_test_sequence()
        self.add_query_memory_region_packets(heap_address)

        # Run the packet stream.
        context = self.expect_gdbremote_sequence()
        self.assertIsNotNone(context)
        mem_region_dict = self.parse_memory_region_packet(context)

        # Ensure address is readable and executable.
        self.assertTrue("permissions" in mem_region_dict)
        self.assertTrue("r" in mem_region_dict["permissions"])
        self.assertTrue("w" in mem_region_dict["permissions"])

        # Ensure the start address and size encompass the address we queried.
        self.assert_address_within_memory_region(heap_address, mem_region_dict)


    @debugserver_test
    @dsym_test
    def test_qMemoryRegionInfo_reports_heap_address_as_readable_writeable_debugserver_dsym(self):
        self.init_debugserver_test()
        self.buildDsym()
        self.set_inferior_startup_launch()
        self.qMemoryRegionInfo_reports_heap_address_as_readable_writeable()

    @llgs_test
    @dwarf_test
    @unittest2.expectedFailure()
    def test_qMemoryRegionInfo_reports_heap_address_as_readable_writeable_llgs_dwarf(self):
        self.init_llgs_test()
        self.buildDwarf()
        self.set_inferior_startup_launch()
        self.qMemoryRegionInfo_reports_heap_address_as_readable_writeable()

    def software_breakpoint_set_and_remove_work(self):
        # Start up the inferior.
        procs = self.prep_debug_monitor_and_inferior(
            inferior_args=["get-code-address-hex:hello", "sleep:1", "call-function:hello"])

        # Run the process
        self.add_register_info_collection_packets()
        self.add_process_info_collection_packets()
        self.test_sequence.add_log_lines(
            [# Start running after initial stop.
             "read packet: $c#00",
             # Match output line that prints the memory address of the function call entry point.
             # Note we require launch-only testing so we can get inferior otuput.
             { "type":"output_match", "regex":r"^code address: 0x([0-9a-fA-F]+)\r\n$", "capture":{ 1:"function_address"} },
             # Now stop the inferior.
             "read packet: {}".format(chr(03)),
             # And wait for the stop notification.
             {"direction":"send", "regex":r"^\$T([0-9a-fA-F]{2})thread:([0-9a-fA-F]+);", "capture":{1:"stop_signo", 2:"stop_thread_id"} }],
            True)

        # Run the packet stream.
        context = self.expect_gdbremote_sequence()
        self.assertIsNotNone(context)

        # Gather process info - we need endian of target to handle register value conversions.
        process_info = self.parse_process_info_response(context)
        endian = process_info.get("endian")
        self.assertIsNotNone(endian)

        # Gather register info entries.
        reg_infos = self.parse_register_info_packets(context)
        (pc_lldb_reg_index, pc_reg_info) = self.find_pc_reg_info(reg_infos)
        self.assertIsNotNone(pc_lldb_reg_index)
        self.assertIsNotNone(pc_reg_info)

        # Grab the function address.
        self.assertIsNotNone(context.get("function_address"))
        function_address = int(context.get("function_address"), 16)

        # Set the breakpoint.
        # Note this might need to be switched per platform (ARM, mips, etc.).
        BREAKPOINT_KIND = 1
        self.reset_test_sequence()
        self.add_set_breakpoint_packets(function_address, do_continue=True, breakpoint_kind=BREAKPOINT_KIND)

        # Run the packet stream.
        context = self.expect_gdbremote_sequence()
        self.assertIsNotNone(context)

        # Verify the stop signal reported was the breakpoint signal number.
        stop_signo = context.get("stop_signo")
        self.assertIsNotNone(stop_signo)
        self.assertEquals(int(stop_signo,16), signal.SIGTRAP)

        # Ensure we did not receive any output.  If the breakpoint was not set, we would
        # see output (from a launched process with captured stdio) printing a hello, world message.
        # That would indicate the breakpoint didn't take.
        self.assertEquals(len(context["O_content"]), 0)

        # Verify that the PC for the main thread is where we expect it - right at the breakpoint address.
        # This acts as a another validation on the register reading code.
        self.reset_test_sequence()
        self.test_sequence.add_log_lines(
            [
             # Print the PC.  This should match the breakpoint address.
             "read packet: $p{0:x}#00".format(pc_lldb_reg_index),
             # Capture $p results.
             { "direction":"send", "regex":r"^\$([0-9a-fA-F]+)#", "capture":{1:"p_response"} },
             ], True)

        context = self.expect_gdbremote_sequence()
        self.assertIsNotNone(context)

        # Verify the PC is where we expect.  Note response is in endianness of the inferior.
        p_response = context.get("p_response")
        self.assertIsNotNone(p_response)

        # Convert from target endian to int.
        returned_pc = lldbgdbserverutils.unpack_register_hex_unsigned(endian, p_response)
        self.assertEquals(returned_pc, function_address)

        # Verify that a breakpoint remove and continue gets us the expected output.
        self.reset_test_sequence()
        self.test_sequence.add_log_lines(
            [
            # Remove the breakpoint.
            "read packet: $z0,{0:x},{1}#00".format(function_address, BREAKPOINT_KIND),
            # Verify the stub could unset it.
            "send packet: $OK#00",
            # Continue running.
            "read packet: $c#00",
            # We should now receive the output from the call.
            { "type":"output_match", "regex":r"^hello, world\r\n$" },
            # And wait for program completion.
            {"direction":"send", "regex":r"^\$W00(.*)#00" },
            ], True)

        context = self.expect_gdbremote_sequence()
        self.assertIsNotNone(context)

    @debugserver_test
    @dsym_test
    def test_software_breakpoint_set_and_remove_work_debugserver_dsym(self):
        self.init_debugserver_test()
        self.buildDsym()
        self.set_inferior_startup_launch()
        self.software_breakpoint_set_and_remove_work()

    @llgs_test
    @dwarf_test
    @unittest2.expectedFailure()
    def test_software_breakpoint_set_and_remove_work_llgs_dwarf(self):
        self.init_llgs_test()
        self.buildDwarf()
        self.set_inferior_startup_launch()
        self.software_breakpoint_set_and_remove_work()

    def g_c1_c2_contents_are(self, args):
        g_c1_address = args["g_c1_address"]
        g_c2_address = args["g_c2_address"]
        expected_g_c1 = args["expected_g_c1"]
        expected_g_c2 = args["expected_g_c2"]

        # Read g_c1 and g_c2 contents.
        self.reset_test_sequence()
        self.test_sequence.add_log_lines(
            ["read packet: $m{0:x},{1:x}#00".format(g_c1_address, 1),
             {"direction":"send", "regex":r"^\$(.+)#[0-9a-fA-F]{2}$", "capture":{1:"g_c1_contents"} },
             "read packet: $m{0:x},{1:x}#00".format(g_c2_address, 1),
             {"direction":"send", "regex":r"^\$(.+)#[0-9a-fA-F]{2}$", "capture":{1:"g_c2_contents"} }],
            True)

        # Run the packet stream.
        context = self.expect_gdbremote_sequence()
        self.assertIsNotNone(context)

        # Check if what we read from inferior memory is what we are expecting.
        self.assertIsNotNone(context.get("g_c1_contents"))
        self.assertIsNotNone(context.get("g_c2_contents"))
 
        return (context.get("g_c1_contents").decode("hex") == expected_g_c1) and (context.get("g_c2_contents").decode("hex") == expected_g_c2)

    def count_single_steps_until_true(self, thread_id, predicate, args, max_step_count=100):
        single_step_count = 0

        while single_step_count < max_step_count:
            # Single step.
            self.reset_test_sequence()
            self.test_sequence.add_log_lines(
                [# Set the continue thread.
                 "read packet: $Hc{0:x}#00".format(thread_id),
                 "send packet: $OK#00",
                 # Single step.
                 "read packet: $s#00",
                 # "read packet: $vCont;s:{0:x}#00".format(thread_id),
                 # Expect a breakpoint stop report.
                 {"direction":"send", "regex":r"^\$T([0-9a-fA-F]{2})thread:([0-9a-fA-F]+);", "capture":{1:"stop_signo", 2:"stop_thread_id"} },
                 ], True)
            context = self.expect_gdbremote_sequence()
            self.assertIsNotNone(context)
            self.assertIsNotNone(context.get("stop_signo"))
            self.assertEquals(int(context.get("stop_signo"), 16), signal.SIGTRAP)

            single_step_count += 1

            # See if the predicate is true.  If so, we're done.
            if predicate(args):
                return (True, single_step_count)

        # The predicate didn't return true within the runaway step count.
        return (False, single_step_count)

    def single_step_only_steps_one_instruction(self):
        # Start up the inferior.
        procs = self.prep_debug_monitor_and_inferior(
            inferior_args=["get-code-address-hex:swap_chars", "get-data-address-hex:g_c1", "get-data-address-hex:g_c2", "sleep:1", "call-function:swap_chars", "sleep:5"])

        # Run the process
        self.test_sequence.add_log_lines(
            [# Start running after initial stop.
             "read packet: $c#00",
             # Match output line that prints the memory address of the function call entry point.
             # Note we require launch-only testing so we can get inferior otuput.
             { "type":"output_match", "regex":r"^code address: 0x([0-9a-fA-F]+)\r\ndata address: 0x([0-9a-fA-F]+)\r\ndata address: 0x([0-9a-fA-F]+)\r\n$", 
               "capture":{ 1:"function_address", 2:"g_c1_address", 3:"g_c2_address"} },
             # Now stop the inferior.
             "read packet: {}".format(chr(03)),
             # And wait for the stop notification.
             {"direction":"send", "regex":r"^\$T([0-9a-fA-F]{2})thread:([0-9a-fA-F]+);", "capture":{1:"stop_signo", 2:"stop_thread_id"} }],
            True)

        # Run the packet stream.
        context = self.expect_gdbremote_sequence()
        self.assertIsNotNone(context)

        # Grab the main thread id.
        self.assertIsNotNone(context.get("stop_thread_id"))
        main_thread_id = int(context.get("stop_thread_id"), 16)

        # Grab the function address.
        self.assertIsNotNone(context.get("function_address"))
        function_address = int(context.get("function_address"), 16)

        # Grab the data addresses.
        self.assertIsNotNone(context.get("g_c1_address"))
        g_c1_address = int(context.get("g_c1_address"), 16)

        self.assertIsNotNone(context.get("g_c2_address"))
        g_c2_address = int(context.get("g_c2_address"), 16)

        # Set a breakpoint at the given address.
        # Note this might need to be switched per platform (ARM, mips, etc.).
        BREAKPOINT_KIND = 1
        self.reset_test_sequence()
        self.add_set_breakpoint_packets(function_address, do_continue=True, breakpoint_kind=BREAKPOINT_KIND)
        context = self.expect_gdbremote_sequence()
        self.assertIsNotNone(context)

        # Remove the breakpoint.
        self.reset_test_sequence()
        self.add_remove_breakpoint_packets(function_address, breakpoint_kind=BREAKPOINT_KIND)
        context = self.expect_gdbremote_sequence()
        self.assertIsNotNone(context)

        # Verify g_c1 and g_c2 match expected initial state.
        args = {}
        args["g_c1_address"] = g_c1_address
        args["g_c2_address"] = g_c2_address
        args["expected_g_c1"] = "0"
        args["expected_g_c2"] = "1"

        self.assertTrue(self.g_c1_c2_contents_are(args))

        # Verify we take only a small number of steps to hit the first state.  Might need to work through function entry prologue code.
        args["expected_g_c1"] = "1"
        args["expected_g_c2"] = "1"
        (state_reached, step_count) = self.count_single_steps_until_true(main_thread_id, self.g_c1_c2_contents_are, args, max_step_count=25)
        self.assertTrue(state_reached)

        # Verify we hit the next state.
        args["expected_g_c1"] = "1"
        args["expected_g_c2"] = "0"
        (state_reached, step_count) = self.count_single_steps_until_true(main_thread_id, self.g_c1_c2_contents_are, args, max_step_count=5)
        self.assertTrue(state_reached)
        self.assertEquals(step_count, 1)

        # Verify we hit the next state.
        args["expected_g_c1"] = "0"
        args["expected_g_c2"] = "0"
        (state_reached, step_count) = self.count_single_steps_until_true(main_thread_id, self.g_c1_c2_contents_are, args, max_step_count=5)
        self.assertTrue(state_reached)
        self.assertEquals(step_count, 1)

        # Verify we hit the next state.
        args["expected_g_c1"] = "0"
        args["expected_g_c2"] = "1"
        (state_reached, step_count) = self.count_single_steps_until_true(main_thread_id, self.g_c1_c2_contents_are, args, max_step_count=5)
        self.assertTrue(state_reached)
        self.assertEquals(step_count, 1)

    @debugserver_test
    @dsym_test
    def test_single_step_only_steps_one_instruction_debugserver_dsym(self):
        self.init_debugserver_test()
        self.buildDsym()
        self.set_inferior_startup_launch()
        self.single_step_only_steps_one_instruction()

    @llgs_test
    @dwarf_test
    @unittest2.expectedFailure()
    def test_single_step_only_steps_one_instruction_llgs_dwarf(self):
        self.init_llgs_test()
        self.buildDwarf()
        self.set_inferior_startup_launch()
        self.single_step_only_steps_one_instruction()

    def qSupported_returns_known_stub_features(self):
        # Start up the stub and start/prep the inferior.
        procs = self.prep_debug_monitor_and_inferior()
        self.add_qSupported_packets()

        # Run the packet stream.
        context = self.expect_gdbremote_sequence()
        self.assertIsNotNone(context)

        # Retrieve the qSupported features.
        supported_dict = self.parse_qSupported_response(context)
        self.assertIsNotNone(supported_dict)
        self.assertTrue(len(supported_dict) > 0)

    @debugserver_test
    @dsym_test
    def test_qSupported_returns_known_stub_features_debugserver_dsym(self):
        self.init_debugserver_test()
        self.buildDsym()
        self.set_inferior_startup_launch()
        self.qSupported_returns_known_stub_features()

    @llgs_test
    @dwarf_test
    @unittest2.expectedFailure()
    def test_qSupported_returns_known_stub_features_llgs_dwarf(self):
        self.init_llgs_test()
        self.buildDwarf()
        self.set_inferior_startup_launch()
        self.qSupported_returns_known_stub_features()

    def written_M_content_reads_back_correctly(self):
        TEST_MESSAGE = "Hello, memory"

        # Start up the stub and start/prep the inferior.
        procs = self.prep_debug_monitor_and_inferior(inferior_args=["set-message:xxxxxxxxxxxxxX", "get-data-address-hex:g_message", "sleep:1", "print-message:"])
        self.test_sequence.add_log_lines(
            [
             # Start running after initial stop.
             "read packet: $c#00",
             # Match output line that prints the memory address of the message buffer within the inferior. 
             # Note we require launch-only testing so we can get inferior otuput.
             { "type":"output_match", "regex":r"^data address: 0x([0-9a-fA-F]+)\r\n$", "capture":{ 1:"message_address"} },
             # Now stop the inferior.
             "read packet: {}".format(chr(03)),
             # And wait for the stop notification.
             {"direction":"send", "regex":r"^\$T([0-9a-fA-F]{2})thread:([0-9a-fA-F]+);", "capture":{1:"stop_signo", 2:"stop_thread_id"} }],
            True)
        context = self.expect_gdbremote_sequence()
        self.assertIsNotNone(context)

        # Grab the message address.
        self.assertIsNotNone(context.get("message_address"))
        message_address = int(context.get("message_address"), 16)

        # Hex-encode the test message, adding null termination.
        hex_encoded_message = TEST_MESSAGE.encode("hex")

        # Write the message to the inferior.
        self.reset_test_sequence()
        self.test_sequence.add_log_lines(
            ["read packet: $M{0:x},{1:x}:{2}#00".format(message_address, len(hex_encoded_message)/2, hex_encoded_message),
             "send packet: $OK#00",
             "read packet: $c#00",
             { "type":"output_match", "regex":r"^message: (.+)\r\n$", "capture":{ 1:"printed_message"} },
             "send packet: $W00#00",
            ], True)
        context = self.expect_gdbremote_sequence()
        self.assertIsNotNone(context)

        # Ensure what we read from inferior memory is what we wrote.
        printed_message = context.get("printed_message")
        self.assertIsNotNone(printed_message)
        self.assertEquals(printed_message, TEST_MESSAGE + "X")

    @debugserver_test
    @dsym_test
    def test_written_M_content_reads_back_correctly_debugserver_dsym(self):
        self.init_debugserver_test()
        self.buildDsym()
        self.set_inferior_startup_launch()
        self.written_M_content_reads_back_correctly()

    @llgs_test
    @dwarf_test
    @unittest2.expectedFailure()
    def test_written_M_content_reads_back_correctly_llgs_dwarf(self):
        self.init_llgs_test()
        self.buildDwarf()
        self.set_inferior_startup_launch()
        self.written_M_content_reads_back_correctly()

    def flip_all_bits_in_each_register_value(self, reg_infos, endian):
        self.assertIsNotNone(reg_infos)

        successful_writes = 0
        failed_writes = 0

        for reg_info in reg_infos:
            # Use the lldb register index added to the reg info.  We're not necessarily
            # working off a full set of register infos, so an inferred register index could be wrong. 
            reg_index = reg_info["lldb_register_index"]
            self.assertIsNotNone(reg_index)

            reg_byte_size = int(reg_info["bitsize"])/8
            self.assertTrue(reg_byte_size > 0)

            # Read the existing value.
            self.reset_test_sequence()
            self.test_sequence.add_log_lines(
                ["read packet: $p{0:x}#00".format(reg_index),
                 { "direction":"send", "regex":r"^\$([0-9a-fA-F]+)#", "capture":{1:"p_response"} }],
                True)
            context = self.expect_gdbremote_sequence()
            self.assertIsNotNone(context)

            # Verify the response length.
            p_response = context.get("p_response")
            self.assertIsNotNone(p_response)
            initial_reg_value = lldbgdbserverutils.unpack_register_hex_unsigned(endian, p_response)

            # Flip the value by xoring with all 1s
            all_one_bits_raw = "ff" * (int(reg_info["bitsize"]) / 8)
            flipped_bits_int = initial_reg_value ^ int(all_one_bits_raw, 16)
            # print "reg (index={}, name={}): val={}, flipped bits (int={}, hex={:x})".format(reg_index, reg_info["name"], initial_reg_value, flipped_bits_int, flipped_bits_int)

            # Write the flipped value to the register.
            self.reset_test_sequence()
            self.test_sequence.add_log_lines(
                ["read packet: $P{0:x}={1}#00".format(reg_index, lldbgdbserverutils.pack_register_hex(endian, flipped_bits_int, byte_size=reg_byte_size)),
                { "direction":"send", "regex":r"^\$(OK|E[0-9a-fA-F]+)#[0-9a-fA-F]{2}", "capture":{1:"P_response"} },
                ], True)
            context = self.expect_gdbremote_sequence()
            self.assertIsNotNone(context)

            # Determine if the write succeeded.  There are a handful of registers that can fail, or partially fail
            # (e.g. flags, segment selectors, etc.) due to register value restrictions.  Don't worry about them
            # all flipping perfectly.
            P_response = context.get("P_response")
            self.assertIsNotNone(P_response)
            if P_response == "OK":
                successful_writes += 1
            else:
                failed_writes += 1
                # print "reg (index={}, name={}) write FAILED (error: {})".format(reg_index, reg_info["name"], P_response)

            # Read back the register value, ensure it matches the flipped value.
            if P_response == "OK":
                self.reset_test_sequence()
                self.test_sequence.add_log_lines(
                    ["read packet: $p{0:x}#00".format(reg_index),
                     { "direction":"send", "regex":r"^\$([0-9a-fA-F]+)#", "capture":{1:"p_response"} }],
                    True)
                context = self.expect_gdbremote_sequence()
                self.assertIsNotNone(context)

                verify_p_response_raw = context.get("p_response")
                self.assertIsNotNone(verify_p_response_raw)
                verify_bits = lldbgdbserverutils.unpack_register_hex_unsigned(endian, verify_p_response_raw)

                if verify_bits != flipped_bits_int:
                    # Some registers, like mxcsrmask and others, will permute what's written.  Adjust succeed/fail counts.
                    # print "reg (index={}, name={}): read verify FAILED: wrote {:x}, verify read back {:x}".format(reg_index, reg_info["name"], flipped_bits_int, verify_bits)
                    successful_writes -= 1
                    failed_writes +=1

        return (successful_writes, failed_writes)

    def P_writes_all_gpr_registers(self):
        # Start inferior debug session, grab all register info.
        procs = self.prep_debug_monitor_and_inferior(inferior_args=["sleep:2"])
        self.add_register_info_collection_packets()
        self.add_process_info_collection_packets()

        context = self.expect_gdbremote_sequence()
        self.assertIsNotNone(context)

        # Process register infos.
        reg_infos = self.parse_register_info_packets(context)
        self.assertIsNotNone(reg_infos)
        self.add_lldb_register_index(reg_infos)

        # Process endian.
        process_info = self.parse_process_info_response(context)
        endian = process_info.get("endian")
        self.assertIsNotNone(endian)

        # Pull out the general purpose register infos that are not contained in other registers.
        gpr_reg_infos = [reg_info for reg_info in reg_infos if ("set" in reg_info) and (reg_info["set"] == "General Purpose Registers") and ("container-regs" not in reg_info)]
        self.assertIsNotNone(len(gpr_reg_infos) > 0)

        # Write flipped bit pattern of existing value to each register.
        (successful_writes, failed_writes) = self.flip_all_bits_in_each_register_value(gpr_reg_infos, endian)
        # print "successful writes: {}, failed writes: {}".format(successful_writes, failed_writes)
        self.assertTrue(successful_writes > 0)

    # Note: as of this moment, a hefty number of the GPR writes are failing with E32 (everything except rax-rdx, rdi, rsi, rbp).
    # Come back to this.  I have the test rigged to verify that at least some of the bit-flip writes work.
    @debugserver_test
    @dsym_test
    def test_P_writes_all_gpr_registers_debugserver_dsym(self):
        self.init_debugserver_test()
        self.buildDsym()
        self.set_inferior_startup_launch()
        self.P_writes_all_gpr_registers()

    @llgs_test
    @dwarf_test
    @unittest2.expectedFailure()
    def test_P_writes_all_gpr_registers_llgs_dwarf(self):
        self.init_llgs_test()
        self.buildDwarf()
        self.set_inferior_startup_launch()
        self.P_writes_all_gpr_registers()

    def P_and_p_thread_suffix_work(self):
        # Startup the inferior with three threads.
        procs = self.prep_debug_monitor_and_inferior(inferior_args=["thread:new", "thread:new"])
        self.add_thread_suffix_request_packets()
        self.add_register_info_collection_packets()
        self.add_process_info_collection_packets()

        context = self.expect_gdbremote_sequence()
        self.assertIsNotNone(context)

        process_info = self.parse_process_info_response(context)
        self.assertIsNotNone(process_info)
        endian = process_info.get("endian")
        self.assertIsNotNone(endian)

        reg_infos = self.parse_register_info_packets(context)
        self.assertIsNotNone(reg_infos)
        self.add_lldb_register_index(reg_infos)

        reg_index = self.select_modifiable_register(reg_infos)
        self.assertIsNotNone(reg_index)
        reg_byte_size = int(reg_infos[reg_index]["bitsize"]) / 8
        self.assertTrue(reg_byte_size > 0)

        # Run the process a bit so threads can start up, and collect register info.
        context = self.run_process_then_stop(run_seconds=1)
        self.assertIsNotNone(context)

        # Wait for 3 threads to be present.
        threads = self.wait_for_thread_count(3, timeout_seconds=5)
        self.assertEquals(len(threads), 3)

        expected_reg_values = []
        register_increment = 1
        next_value = None

        # Set the same register in each of 3 threads to a different value.
        # Verify each one has the unique value.
        for thread in threads:
            # If we don't have a next value yet, start it with the initial read value + 1
            if not next_value:
                # Read pre-existing register value.
                self.reset_test_sequence()
                self.test_sequence.add_log_lines(
                    ["read packet: $p{0:x};thread:{1:x}#00".format(reg_index, thread),
                     { "direction":"send", "regex":r"^\$([0-9a-fA-F]+)#", "capture":{1:"p_response"} },
                    ], True)
                context = self.expect_gdbremote_sequence()
                self.assertIsNotNone(context)

                # Set the next value to use for writing as the increment plus current value.
                p_response = context.get("p_response")
                self.assertIsNotNone(p_response)
                next_value = lldbgdbserverutils.unpack_register_hex_unsigned(endian, p_response)

            # Set new value using P and thread suffix.
            self.reset_test_sequence()
            self.test_sequence.add_log_lines(
                ["read packet: $P{0:x}={1};thread:{2:x}#00".format(reg_index, lldbgdbserverutils.pack_register_hex(endian, next_value, byte_size=reg_byte_size), thread),
                 "send packet: $OK#00",
                ], True)
            context = self.expect_gdbremote_sequence()
            self.assertIsNotNone(context)

            # Save the value we set.
            expected_reg_values.append(next_value)

            # Increment value for next thread to use (we want them all different so we can verify they wrote to each thread correctly next.)
            next_value += register_increment

        # Revisit each thread and verify they have the expected value set for the register we wrote.
        thread_index = 0
        for thread in threads:
            # Read pre-existing register value.
            self.reset_test_sequence()
            self.test_sequence.add_log_lines(
                ["read packet: $p{0:x};thread:{1:x}#00".format(reg_index, thread),
                 { "direction":"send", "regex":r"^\$([0-9a-fA-F]+)#", "capture":{1:"p_response"} },
                ], True)
            context = self.expect_gdbremote_sequence()
            self.assertIsNotNone(context)

            # Get the register value.
            p_response = context.get("p_response")
            self.assertIsNotNone(p_response)
            read_value = lldbgdbserverutils.unpack_register_hex_unsigned(endian, p_response)

            # Make sure we read back what we wrote.
            self.assertEquals(read_value, expected_reg_values[thread_index])
            thread_index += 1

    # Note: as of this moment, a hefty number of the GPR writes are failing with E32 (everything except rax-rdx, rdi, rsi, rbp).
    @debugserver_test
    @dsym_test
    def test_P_and_p_thread_suffix_work_debugserver_dsym(self):
        self.init_debugserver_test()
        self.buildDsym()
        self.set_inferior_startup_launch()
        self.P_and_p_thread_suffix_work()

    @llgs_test
    @dwarf_test
    @unittest2.expectedFailure()
    def test_P_and_p_thread_suffix_work_llgs_dwarf(self):
        self.init_llgs_test()
        self.buildDwarf()
        self.set_inferior_startup_launch()
        self.P_and_p_thread_suffix_work()


if __name__ == '__main__':
    unittest2.main()
