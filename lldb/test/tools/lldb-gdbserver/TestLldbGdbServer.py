"""
Test lldb-gdbserver operation
"""

import unittest2
import pexpect
import socket
import subprocess
import sys
import time
from lldbtest import *
from lldbgdbserverutils import *
import logging
import os.path

class LldbGdbServerTestCase(TestBase):

    mydir = TestBase.compute_mydir(__file__)

    port = 12345

    _TIMEOUT_SECONDS = 5

    _GDBREMOTE_KILL_PACKET = "$k#6b"

    _LOGGING_LEVEL = logging.WARNING
    # _LOGGING_LEVEL = logging.DEBUG

    def setUp(self):
        TestBase.setUp(self)
        FORMAT = '%(asctime)-15s %(levelname)-8s %(message)s'
        logging.basicConfig(format=FORMAT)
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(self._LOGGING_LEVEL)
        self.test_sequence = GdbRemoteTestSequence(self.logger)

    def init_llgs_test(self):
        self.debug_monitor_exe = get_lldb_gdbserver_exe()
        if not self.debug_monitor_exe:
            self.skipTest("lldb_gdbserver exe not found")

    def init_debugserver_test(self):
        self.debug_monitor_exe = get_debugserver_exe()
        if not self.debug_monitor_exe:
            self.skipTest("debugserver exe not found")

    def create_socket(self):
        sock = socket.socket()
        logger = self.logger

        def shutdown_socket():
            if sock:
                try:
                    # send the kill packet so lldb-gdbserver shuts down gracefully
                    sock.sendall(LldbGdbServerTestCase._GDBREMOTE_KILL_PACKET)
                except:
                    logger.warning("failed to send kill packet to debug monitor: {}; ignoring".format(sys.exc_info()[0]))

                try:
                    sock.close()
                except:
                    logger.warning("failed to close socket to debug monitor: {}; ignoring".format(sys.exc_info()[0]))

        self.addTearDownHook(shutdown_socket)

        sock.connect(('localhost', self.port))
        return sock

    def start_server(self, attach_pid=None):
        # Create the command line
        commandline = "{} localhost:{}".format(self.debug_monitor_exe, self.port)
        if attach_pid:
            commandline += " --attach=%d" % attach_pid
            
        # start the server
        server = pexpect.spawn(commandline)

        # Turn on logging for what the child sends back.
        if self.TraceOn():
            server.logfile_read = sys.stdout

        # Schedule debug monitor to be shut down during teardown.
        logger = self.logger
        def shutdown_debug_monitor():
            try:
                server.close()
            except:
                logger.warning("failed to close pexpect server for debug monitor: {}; ignoring".format(sys.exc_info()[0]))

        self.addTearDownHook(shutdown_debug_monitor)

        # Wait until we receive the server ready message before continuing.
        server.expect_exact('Listening to port {} for a connection from localhost'.format(self.port))

        # Create a socket to talk to the server
        self.sock = self.create_socket()

        return server

    def launch_process_for_attach(self,sleep_seconds=3):
        # We're going to start a child process that the debug monitor stub can later attach to.
        # This process needs to be started so that it just hangs around for a while.  We'll
        # have it sleep.
        exe_path = os.path.abspath("a.out")
        args = [exe_path]
        if sleep_seconds:
            args.append("sleep:%d" % sleep_seconds)
            
        return subprocess.Popen(args)

    def add_no_ack_remote_stream(self):
        self.test_sequence.add_log_lines(
            ["read packet: +",
             "read packet: $QStartNoAckMode#b0",
             "send packet: +",
             "send packet: $OK#9a",
             "read packet: +"],
            True)

    def add_verified_launch_packets(self, launch_args):
        self.test_sequence.add_log_lines(
            ["read packet: %s" % build_gdbremote_A_packet(launch_args),
             "send packet: $OK#00",
             "read packet: $qLaunchSuccess#a5",
             "send packet: $OK#00"],
            True)

    def add_get_pid(self):
        self.test_sequence.add_log_lines(
            ["read packet: $qProcessInfo#00",
              { "direction":"send", "regex":r"^\$pid:([0-9a-fA-F]+);", "capture":{1:"pid"} }],
            True)

    def add_register_info_collection_packets(self):
        self.test_sequence.add_log_lines(
            [ { "type":"multi_response", "query":"qRegisterInfo", "append_iteration_suffix":True,
              "end_regex":re.compile(r"^\$(E\d+)?#[0-9a-fA-F]{2}$"),
              "save_key":"reg_info_responses" } ],
            True)

    def parse_register_info_packets(self, context):
        """Return an array of register info dictionaries, one per register info."""
        reg_info_responses = context.get("reg_info_responses")
        self.assertIsNotNone(reg_info_responses)

        # Parse register infos.
        return [parse_reg_info_response(reg_info_response) for reg_info_response in reg_info_responses]

    def expect_gdbremote_sequence(self):
        return expect_lldb_gdbserver_replay(self, self.sock, self.test_sequence, self._TIMEOUT_SECONDS, self.logger)

    _KNOWN_REGINFO_KEYS = [
        "name",
        "alt-name",
        "bitsize",
        "offset",
        "encoding",
        "format",
        "set",
        "gcc",
        "dwarf",
        "generic",
        "container-regs",
        "invalidate-regs"
    ]

    def assert_valid_reg_info(self, reg_info):
        # Assert we know about all the reginfo keys parsed.
        for key in reg_info:
            self.assertTrue(key in self._KNOWN_REGINFO_KEYS)
        
        # Check the bare-minimum expected set of register info keys.
        self.assertTrue("name" in reg_info)
        self.assertTrue("bitsize" in reg_info)
        self.assertTrue("offset" in reg_info)
        self.assertTrue("encoding" in reg_info)
        self.assertTrue("format" in reg_info)


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
            ["read packet: %s" % build_gdbremote_A_packet(launch_args),
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
             "send packet: $O{}#00".format(gdbremote_hex_encode_string("hello, world\r\n")),
             "send packet: $W00#00"],
            True)
        self.expect_gdbremote_sequence()

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
        server = self.start_server()
        self.assertIsNotNone(server)

        # Build launch args
        launch_args = [os.path.abspath('a.out'), "hello, world"]

        # Build the expected protocol stream
        self.add_no_ack_remote_stream()
        self.add_verified_launch_packets(launch_args)
        self.test_sequence.add_log_lines(
            ["read packet: $qProcessInfo#00",
             { "direction":"send", "regex":r"^\$pid:([0-9a-fA-F]+);", "capture":{1:"pid"} }],
            True)

        # Run the stream
        context = self.expect_gdbremote_sequence()
        self.assertIsNotNone(context)

        # Ensure the process id looks reasonable.
        pid_text = context.get('pid', None)
        self.assertIsNotNone(pid_text)
        pid = int(pid_text, base=16)
        self.assertNotEqual(0, pid)

        # If possible, verify that the process is running.
        self.assertTrue(process_is_running(pid, True))

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

    def attach_commandline_qProcessInfo_reports_pid(self):
        # Launch the process that we'll use as the inferior.
        inferior = self.launch_process_for_attach()
        self.assertIsNotNone(inferior)
        self.assertTrue(inferior.pid > 0)
        
        # Launch the debug monitor stub, attaching to the inferior.
        server = self.start_server(attach_pid=inferior.pid)
        self.assertIsNotNone(server)

        # Check that the stub reports attachment to the inferior.
        self.add_no_ack_remote_stream()
        self.add_get_pid()
        context = self.expect_gdbremote_sequence()

        # Ensure the process id matches what we expected.
        pid_text = context.get('pid', None)
        self.assertIsNotNone(pid_text)
        reported_pid = int(pid_text, base=16)
        self.assertEqual(reported_pid, inferior.pid)

    @debugserver_test
    @dsym_test
    def test_attach_commandline_qProcessInfo_reports_pid_debugserver_dsym(self):
        self.init_debugserver_test()
        self.buildDsym()
        self.attach_commandline_qProcessInfo_reports_pid()

    @llgs_test
    @dwarf_test
    @unittest2.expectedFailure()
    def test_attach_commandline_qProcessInfo_reports_pid_llgs_dwarf(self):
        self.init_llgs_test()
        self.buildDwarf()
        self.attach_commandline_qProcessInfo_reports_pid()

    def attach_commandline_continue_app_exits(self):
        # Launch the process that we'll use as the inferior.
        inferior = self.launch_process_for_attach(sleep_seconds=1)
        self.assertIsNotNone(inferior)
        self.assertTrue(inferior.pid > 0)

        # Launch the debug monitor stub, attaching to the inferior.
        server = self.start_server(attach_pid=inferior.pid)
        self.assertIsNotNone(server)

        # Check that the stub reports attachment to the inferior.
        self.add_no_ack_remote_stream()
        self.add_get_pid()
        self.test_sequence.add_log_lines(
            ["read packet: $vCont;c#00",
             "send packet: $W00#00"],
            True)
        self.expect_gdbremote_sequence()

        # Process should be dead now.  Reap results.
        poll_result = inferior.poll()
        self.assertIsNotNone(poll_result)

        # Where possible, verify at the system level that the process is not running.
        self.assertFalse(process_is_running(inferior.pid, False))

    @debugserver_test
    @dsym_test
    def test_attach_commandline_continue_app_exits_debugserver_dsym(self):
        self.init_debugserver_test()
        self.buildDsym()
        self.attach_commandline_continue_app_exits()

    @llgs_test
    @dwarf_test
    @unittest2.expectedFailure()
    def test_attach_commandline_continue_app_exits_llgs_dwarf(self):
        self.init_llgs_test()
        self.buildDwarf()
        self.attach_commandline_continue_app_exits()

    def attach_commandline_kill_after_initial_stop(self):
        # Launch the process that we'll use as the inferior.
        inferior = self.launch_process_for_attach(sleep_seconds=10)
        self.assertIsNotNone(inferior)
        self.assertTrue(inferior.pid > 0)

        # Launch the debug monitor stub, attaching to the inferior.
        server = self.start_server(attach_pid=inferior.pid)
        self.assertIsNotNone(server)

        # Check that the stub reports attachment to the inferior.
        self.add_no_ack_remote_stream()
        self.add_get_pid()
        self.test_sequence.add_log_lines(
            ["read packet: $k#6b",
             "send packet: $X09#00"],
            True)
        self.expect_gdbremote_sequence()

        # Process should be dead now.  Reap results.
        poll_result = inferior.poll()
        self.assertIsNotNone(poll_result)

        # Where possible, verify at the system level that the process is not running.
        self.assertFalse(process_is_running(inferior.pid, False))

    @debugserver_test
    @dsym_test
    def test_attach_commandline_kill_after_initial_stop_debugserver_dsym(self):
        self.init_debugserver_test()
        self.buildDsym()
        self.attach_commandline_kill_after_initial_stop()

    @llgs_test
    @dwarf_test
    @unittest2.expectedFailure()
    def test_attach_commandline_kill_after_initial_stop_llgs_dwarf(self):
        self.init_llgs_test()
        self.buildDwarf()
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
        self.assert_valid_reg_info(parse_reg_info_response(reg_info_packet))

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

        # Collect all generics found.
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

        # Collect all generics found.
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


if __name__ == '__main__':
    unittest2.main()
