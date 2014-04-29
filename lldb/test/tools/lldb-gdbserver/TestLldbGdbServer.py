"""
Test lldb-gdbserver operation
"""

import unittest2
import pexpect
import socket
import sys
from lldbtest import *
from lldbgdbserverutils import *
import logging
import os.path

class LldbGdbServerTestCase(TestBase):

    mydir = TestBase.compute_mydir(__file__)

    port = 12345

    _TIMEOUT_SECONDS = 5

    _GDBREMOTE_KILL_PACKET = "$k#6b"

    # _LOGGING_LEVEL = logging.WARNING
    _LOGGING_LEVEL = logging.DEBUG

    def setUp(self):
        TestBase.setUp(self)

        FORMAT = '%(asctime)-15s %(levelname)-8s %(message)s'
        logging.basicConfig(format=FORMAT)
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(self._LOGGING_LEVEL)

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

    def start_server(self):
        # start the server
        server = pexpect.spawn("{} localhost:{}".format(self.debug_monitor_exe, self.port))

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

    def create_no_ack_remote_stream(self):
       return [
            "lldb-gdbserver <  19> read packet: +",
            "lldb-gdbserver <  19> read packet: $QStartNoAckMode#b0",
            "lldb-gdbserver <   1> send packet: +",
            "lldb-gdbserver <   6> send packet: $OK#9a",
            "lldb-gdbserver <   1> read packet: +"]

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

        log_lines = self.create_no_ack_remote_stream()

        expect_lldb_gdbserver_replay(self, self.sock, log_lines, True,
                                     self._TIMEOUT_SECONDS, self.logger)

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

        log_lines = self.create_no_ack_remote_stream()
        log_lines.extend([
            "lldb-gdbserver <  26> read packet: $QThreadSuffixSupported#e4",
            "lldb-gdbserver <   6> send packet: $OK#9a"])

        expect_lldb_gdbserver_replay(self, self.sock, log_lines, True,
                                     self._TIMEOUT_SECONDS, self.logger)

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

        log_lines = self.create_no_ack_remote_stream()
        log_lines.extend([
            "lldb-gdbserver <  27> read packet: $QListThreadsInStopReply#21",
            "lldb-gdbserver <   6> send packet: $OK#9a"])

        expect_lldb_gdbserver_replay(self, self.sock, log_lines, True,
                                     self._TIMEOUT_SECONDS, self.logger)

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

        log_lines = self.create_no_ack_remote_stream()
        log_lines.extend([
            "lldb-gdbserver <   0> read packet: %s" % build_gdbremote_A_packet(launch_args),
            "lldb-gdbserver <   6> send packet: $OK#9a"])

        expect_lldb_gdbserver_replay(self, self.sock, log_lines, True,
                                     self._TIMEOUT_SECONDS, self.logger)

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

        log_lines = self.create_no_ack_remote_stream()
        log_lines.extend([
            "lldb-gdbserver <   0> read packet: %s" % build_gdbremote_A_packet(launch_args),
            "lldb-gdbserver <   6> send packet: $OK#00",
            "lldb-gdbserver <  18> read packet: $qLaunchSuccess#a5",
            "lldb-gdbserver <   6> send packet: $OK#00",
            "lldb-gdbserver <   5> read packet: $vCont;c#00",
            "lldb-gdbserver <   7> send packet: $W00#00"])

        expect_lldb_gdbserver_replay(self, self.sock, log_lines, True,
                                     self._TIMEOUT_SECONDS, self.logger)

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

        log_lines = self.create_no_ack_remote_stream()
        log_lines.extend([
            "lldb-gdbserver <   0> read packet: %s" % build_gdbremote_A_packet(launch_args),
            "lldb-gdbserver <   6> send packet: $OK#00",
            "lldb-gdbserver <  18> read packet: $qLaunchSuccess#a5",
            "lldb-gdbserver <   6> send packet: $OK#00",
            "lldb-gdbserver <   5> read packet: $vCont;c#00",
            "lldb-gdbserver <   7> send packet: $W{0:02x}#00".format(RETVAL)])

        expect_lldb_gdbserver_replay(self, self.sock, log_lines, True,
                                     self._TIMEOUT_SECONDS, self.logger)

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

        log_lines = self.create_no_ack_remote_stream()
        log_lines.extend([
            "lldb-gdbserver <   0> read packet: %s" % build_gdbremote_A_packet(launch_args),
            "lldb-gdbserver <   6> send packet: $OK#00",
            "lldb-gdbserver <  18> read packet: $qLaunchSuccess#a5",
            "lldb-gdbserver <   6> send packet: $OK#00",
            "lldb-gdbserver <   5> read packet: $vCont;c#00",
            "lldb-gdbserver <   7> send packet: $O{}#00".format(gdbremote_hex_encode_string("hello, world\r\n")),
            "lldb-gdbserver <   7> send packet: $W00#00"])

        expect_lldb_gdbserver_replay(self, self.sock, log_lines, True,
                                     self._TIMEOUT_SECONDS, self.logger)

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


if __name__ == '__main__':
    unittest2.main()
